# Hire Right Agent

An AI-powered hiring assistant built on the Databricks platform for Jackson & Jackson HR Digital. This reference implementation demonstrates how to combine generative AI agents, vector search, ML models, and data analytics into a production-ready hiring evaluation application.

## Overview

Hire Right Agent helps HR leaders evaluate candidates across four open positions. It provides:

- A **Databricks App** frontend for browsing candidates, viewing scores, and reading resumes
- A **ResponsesAgent** that answers questions about candidates using tools for data retrieval, ML scoring, and email
- A **Genie Space** for natural language analytics over HR data
- A **data pipeline** that ingests, transforms, and scores candidates end-to-end
- A **Lakeview dashboard** for hiring analytics

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Databricks App                        │
│  ┌─────────────────┐        ┌─────────────────────────┐ │
│  │  index.html      │◄──────►│  app.py (FastAPI)       │ │
│  │  (SPA frontend)  │        │  /api/candidates        │ │
│  └─────────────────┘        │  /api/jobs              │ │
│                             │  /api/chat              │ │
│                             │  /api/genie             │ │
│                             │  /api/resume-pdf/{id}   │ │
│                             └──────────┬──────────────┘ │
└────────────────────────────────────────┼────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
     ┌────────▼────────┐      ┌──────────▼──────────┐   ┌──────────▼──────────┐
     │  Hire Right     │      │   Genie Space        │   │   Unity Catalog     │
     │  Agent          │      │   (HR Analytics)     │   │   bx4.hrd_2030      │
     │  (ResponsesAgent│      └─────────────────────┘   │   - candidates      │
     │   on Model      │                                 │   - jobs            │
     │   Serving)      │      ┌─────────────────────┐   │   - resumes         │
     │                 │      │   Vector Search      │   └─────────────────────┘
     │  Tools:         ├─────►│   Resume Index       │
     │  - query_genie  │      └─────────────────────┘
     │  - search_resume│
     │  - predict_score│      ┌─────────────────────┐
     │  - send_email   ├─────►│   ML Model Serving  │
     └─────────────────┘      │   (hire_right)       │
                              └─────────────────────┘
```

## Repository Structure

```
.
├── app/
│   ├── app.py          # FastAPI backend (proxies to Databricks APIs)
│   ├── app.yaml        # Databricks App resource declarations
│   └── index.html      # Single-page application frontend
├── agent_src/
│   ├── hire_right_agent.py   # MLflow ResponsesAgent definition
│   ├── config.yaml           # Agent configuration
│   ├── config_helper.py      # Centralized env var loading
│   └── tools/
│       ├── tool_query_genie.py    # Genie Space REST API client
│       ├── tool_search_resume.py  # Vector Search retrieval
│       ├── tool_predict_score.py  # ML model endpoint invocation
│       └── tool_send_email.py     # Mailgun email integration
├── notebooks/
│   ├── 00_load_bronze.ipynb                          # Raw data ingestion
│   ├── 01_load_silver.ipynb                          # Cleaning & transformation
│   ├── 01b_build_vector_search.ipynb                 # Resume embedding index
│   ├── 02_apply_data_quality_and_classification.ipynb
│   ├── 03_build_gold.ipynb                           # Candidate scoring table
│   ├── 03b_apply_business_semantics.ipynb
│   ├── 04_create_genie_space.ipynb                   # HR Analytics Genie Space
│   ├── 05_train_ml_model.ipynb                       # Hire/no-hire classifier
│   ├── 05b_create_drift_monitor.ipynb
│   ├── 06_evaluate_register_agent.ipynb              # Agent eval & registration
│   ├── 08_refresh_dashboard.ipynb
│   └── run_all.ipynb                                 # Master orchestration
├── dashboard/
│   └── hiring_analytics.lvdash.json  # Lakeview AI/BI dashboard
├── images/                           # Architecture diagrams
├── data/                             # Seed data
├── slides/                           # Presentation materials
├── databricks.yml                    # Databricks Bundle config
└── env.template                      # Environment variable template
```

## Data Pipeline

The pipeline runs as a Databricks job (`jj-hr-digital-pipeline`) with conditional task execution:

| Step | Notebook | Description |
|------|----------|-------------|
| Bronze | `00_load_bronze` | Ingest raw candidate and job data |
| Silver | `01_load_silver` | Clean, transform, PII mask |
| Vector Search | `01b_build_vector_search` | Embed resumes and build VS index |
| Data Quality | `02_apply_data_quality_and_classification` | DQM checks and classification |
| Gold | `03_build_gold` | Aggregated candidate scoring (8 factors) |
| Semantics | `03b_apply_business_semantics` | Business metrics and context |
| Genie | `04_create_genie_space` | Create HR Analytics Genie Space |
| ML Training | `05_train_ml_model` | Train and register hire/no-hire model |
| Drift Monitor | `05b_create_drift_monitor` | Set up model monitoring |
| Agent | `06_evaluate_register_agent` | Evaluate and register the agent |
| Dashboard | `08_refresh_dashboard` | Refresh Lakeview dashboard |

## Candidate Scoring

Each candidate is scored on 8 weighted factors:

| Factor | Description |
|--------|-------------|
| `education` | Degree and field relevance |
| `experience` | Years and domain fit |
| `leadership` | Management and team experience |
| `certifications` | Relevant credentials |
| `skills_match` | Technical and functional skill alignment |
| `industry` | Industry background match |
| `interview` | Interview performance |
| `culture_fit` | Values and team alignment |

The composite score (0–100) drives the overall recommendation. Scores ≥ 80 are flagged as recommended.

## Agent Tools

The `ResponsesAgent` uses four tools:

- **`query_genie`** — Natural language queries against the HR Analytics Genie Space (e.g., "How many candidates have a score above 80?")
- **`search_resumes`** — Semantic vector search over resume content (top-3 results)
- **`predict_hiring_score`** — Calls the `hire_right` ML model endpoint for a hire/no-hire prediction
- **`send_email`** — Sends candidate summaries via Mailgun

## Setup

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- Databricks CLI configured (`databricks configure`)
- Python 3.10+

### Environment Configuration

```bash
cp env.template .env
# Edit .env with your values
```

Key variables:

| Variable | Description |
|----------|-------------|
| `DATABRICKS_AGENT_ENDPOINT` | Model serving endpoint name |
| `GENIE_SPACE_ID` | HR Analytics Genie Space ID |
| `TARGET_CATALOG` | Unity Catalog catalog name |
| `TARGET_SCHEMA` | Unity Catalog schema name |
| `DATABRICKS_WAREHOUSE_ID` | SQL warehouse for job queries |
| `LLM_ENDPOINT_NAME` | Databricks LLM endpoint |
| `VS_ENDPOINT_NAME` | Vector Search endpoint |
| `VS_INDEX` | Vector Search index name |
| `MAILGUN_API_KEY` | Mailgun API key for email |
| `MAILGUN_API_URL` | Mailgun API URL |
| `SENDER` | Sender email address |

### Deploy with Databricks Bundles

```bash
# Validate the bundle
databricks bundle validate

# Deploy to your workspace
databricks bundle deploy

# Run the full pipeline
databricks bundle run jj-hr-digital-pipeline
```

### Local Development

```bash
pip install -r app/requirements.txt
cd app && uvicorn app:app --reload
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent framework | MLflow 3.0 `ResponsesAgent` |
| LLM | Databricks Model Serving (GPT-5-4) |
| App backend | FastAPI + Uvicorn |
| App frontend | HTML5 / Vanilla JS / marked.js |
| Data platform | Databricks Unity Catalog |
| Vector search | Databricks Vector Search |
| ML tracking | MLflow |
| IaC | Databricks Asset Bundles |
| Email | Mailgun |

## Data Model

- **Catalog:** `bx4`
- **Schema:** `hrd_2030`
- **Candidates:** 20 profiles — 10 historical (hired/rejected) + 10 active pipeline
- **Jobs:** 4 open positions (JR001–JR004)
- **ML Model:** Registered as `hire_right` in MLflow Model Registry
