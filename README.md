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

## Installation

### Prerequisites

- Databricks workspace with Unity Catalog and Vector Search enabled
- Databricks CLI installed and configured (`databricks configure`)
- Python 3.10+
- A Mailgun account for email (free tier: 3,000 emails/month)

---

### Step 1 — Configure `.env` (notebooks)

The `.env` file is the single source of truth used by the notebooks when they run on Databricks. It is never committed to git.

```bash
cp env.template .env
```

Then fill in your values:

| Variable | Where to find it | Notes |
|----------|-----------------|-------|
| `TARGET_CATALOG` | Your Unity Catalog catalog name | e.g. `my_catalog` |
| `TARGET_SCHEMA` | Schema to create for this project | e.g. `hrd_demo` |
| `GENIE_SPACE_ID` | Created by notebook `04_create_genie_space` — copy the ID from the Genie Space URL after running it | Leave blank initially; fill in after step 3 |
| `LLM_ENDPOINT_NAME` | Databricks Serving Endpoints page | e.g. `databricks-claude-3-7-sonnet` |
| `VS_ENDPOINT_NAME` | Databricks Vector Search page | Must exist before running notebooks |
| `VS_INDEX` | Name for the resume index — keep default or rename | `hr_resumes_vs_index` |
| `MODEL_ENDPOINT_NAME` | Created by notebook `05_train_ml_model` | Keep default: `hr-predictive-hiring-endpoint` |
| `AGENT_NAME` | Registered model name in Unity Catalog | Keep default: `hire_right` |
| `AGENT_ENDPOINT_NAME` | Created by notebook `06_evaluate_register_agent` | Keep default: `hire-right-agent-endpoint` |
| `APP_NAME` | Databricks App name | Keep default: `hire-right-agent` |
| `DATABRICKS_WAREHOUSE_ID` | Databricks SQL Warehouses page — copy the ID from any running warehouse | Required for Genie and dashboard queries |
| `MAILGUN_API_URL` | Your Mailgun dashboard → Sending → Domains | Format: `https://api.mailgun.net/v3/<your-domain>/messages` |
| `MAILGUN_API_KEY` | Your Mailgun dashboard → API Keys | Keep secret |
| `SENDER` | Email address you send from — must match your Mailgun domain | e.g. `noreply@yourdomain.com` |
| `RECIPIENT` | Default recipient for agent emails | e.g. `you@yourcompany.com` |

---

### Step 2 — Update `databricks.yml` (bundle deployment)

`databricks.yml` drives the full deployment — job, app, dashboard. Update two things:

**Your workspace target** (bottom of the file) — only `host` needs updating, `root_path` resolves automatically to the deploying user:
```yaml
targets:
  prod:
    workspace:
      host: https://<your-workspace>.cloud.databricks.com
```

**Variable defaults** — these default to the reference deployment values. Override any that differ in your workspace:
```yaml
variables:
  catalog:
    default: my_catalog           # must match TARGET_CATALOG in .env
  schema:
    default: hrd_demo             # must match TARGET_SCHEMA in .env
  warehouse_id:
    default: "<your-warehouse-id>"  # must match DATABRICKS_WAREHOUSE_ID in .env
  vs_endpoint_name:
    default: "<your-vs-endpoint>"   # must match VS_ENDPOINT_NAME in .env
  llm_endpoint_name:
    default: "<your-llm-endpoint>"  # must match LLM_ENDPOINT_NAME in .env
  genie_space_id:
    default: ""                     # fill in after notebook 04 runs
```

The remaining variables (`agent_name`, `agent_endpoint_name`, `app_name`, `model_endpoint_name`) can be left as defaults unless you want different resource names.

---

### Step 3 — Update `app/app.yaml` (Databricks App runtime)

`app.yaml` configures the running Databricks App. Most env vars are injected automatically from the `resources:` block via `valueFrom:` — you only need to update the `resources:` section and two env vars:

```yaml
env:
  - name: TARGET_CATALOG
    value: "my_catalog"     # update to match your catalog
  - name: TARGET_SCHEMA
    value: "hrd_demo"       # update to match your schema

resources:
  - name: agent_endpoint
    serving_endpoint:
      name: hire-right-agent-endpoint   # update if you changed AGENT_ENDPOINT_NAME
  - name: sql_warehouse
    sql_warehouse:
      id: "<your-warehouse-id>"          # update to your warehouse ID
  - name: genie_space
    genie_space:
      name: "Jackson and Johnson HR Digital — Hiring Analytics Genie"  # update if you renamed it
      space_id: "<your-genie-space-id>"  # fill in after notebook 04 runs
```

`DATABRICKS_AGENT_ENDPOINT`, `DATABRICKS_WAREHOUSE_ID`, and `GENIE_SPACE_ID` are injected automatically from the resources above — no need to set them as literal values.

---

### Step 4 — Deploy

```bash
# Validate everything looks correct
databricks bundle validate

# Deploy the job, app, and dashboard
databricks bundle deploy

# Run the full pipeline (builds Bronze → Gold → trains model → deploys agent)
databricks bundle run jj-hr-digital-pipeline
```

After the pipeline completes:
1. Copy the Genie Space ID from the Genie Space URL
2. Update `GENIE_SPACE_ID` in `.env`, `genie_space_id` default in `databricks.yml`, and `space_id` in `app/app.yaml`
3. Redeploy: `databricks bundle deploy`

---

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
