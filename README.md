# Hire Right — AI-Powered Predictive Hiring on Databricks

**Jackson & Jackson HR Digital** (a fictional pharmaceutical company) — an end-to-end AI build that
takes raw, synthetic applicant data through a governed lakehouse, a predictive ML model, a
data-grounded GenAI agent, natural-language analytics, and a low-latency operational serving
layer, and delivers it all to HR leaders in a single Databricks App.

It began as a **customer demo** supporting an HR-Modernization program where Databricks was being
evaluated as the Data & AI platform — specifically the **ML, MLOps, and GenAI** capabilities. It
was later extended for the **FE Bar** (added the Lakebase operational serving layer, the AI offer
letter, live streaming tool calls, and inline HR notes).

> New here? Read this file top-to-bottom, then open `slides/hire-right-fe-bar.pdf` for the value
> story and architecture, and click **Data Flow** in the app for the interactive diagram.

---

## What is deployed vs. illustrative

| Deployed & live (real Databricks resources) | Illustrative / demo-only |
|---|---|
| Bronze→Silver→Gold Delta tables in `bldemos.hrd_2030` (Unity Catalog) | The 20 candidates / 4 jobs are **synthetic** (self-generated) |
| ML model `hr-predictive-hiring-endpoint` (Model Serving) | ML is trained on ~10 labeled rows — a demo-scale model, not a validated one |
| Agent `hire-right-agent-endpoint` (MLflow 3.0 ResponsesAgent) | The "Composite / Total Score" is a **computed formula**, not the model (see note below) |
| Vector Search `bx4_hrd_vs_endpoint` + `hr_resumes_vs_index` | KPI figures in the deck (60% faster, etc.) are **projected/illustrative** |
| Genie Space (HR Analytics, NL→SQL) | Mailgun sender/recipient are demo values in `.env` |
| **Lakebase** instance `hire-right-lb` — synced tables + `candidate_annotations` | |
| Databricks App `hire-right-agent-app` (FastAPI + single-page UI) | |
| Job `jj-hr-digital-pipeline` (the end-to-end pipeline) + Lakeview dashboard | |

**Score note:** the candidate "Total Score" shown in the UI is the **average** of the 8 category
scores (computed in `03_build_gold`). The agent tool's "Composite Score" is a **weighted sum**
(in `agent_src/tools/tool_predict_score.py`). Neither is the ML model — the model returns only the
binary hire / no-hire label. This is intentional for the demo narrative.

## Live resources (default target)

- **Workspace:** `https://fevm-bobertx3-aws-fevm.cloud.databricks.com` (CLI profile `DEFAULT`)
- **App:** `https://hire-right-agent-app-7474645374628060.aws.databricksapps.com`
- **Catalog / schema:** `bldemos.hrd_2030`
- **Lakebase:** instance `hire-right-lb`, database `databricks_postgres`, UC catalog `hrd_2030_lakebase`
- **LLM:** `databricks-gpt-5-4` (via AI Gateway)

---

## Repository structure — what's where

The repo is intentionally lean (~120 tracked files) — build/tooling artifacts
(`node_modules/`, vendored `.agents/` skills, local `mlflow.db`/`mlruns/`, `.venv/`)
are gitignored, so everything you see below is the actual work.

```
.
├── databricks.yml                       # DAB bundle: variables, pipeline Job, dashboard, 2 targets
├── notebooks/                           # Build pipeline — run in order (also the Job's tasks)
│   ├── 00a_setup_schema_volume.ipynb    # Create UC schema + raw_data volume
│   ├── 00_load_bronze.ipynb             # Ingest synthetic CSVs via read_files (Lakeflow)
│   ├── 01_load_silver.ipynb             # Type, clean, mask PII
│   ├── 01b_build_vector_search.ipynb    # Resume embedding index (RAG)
│   ├── 02_apply_data_quality_and_classification.ipynb
│   ├── 03_build_gold.ipynb              # candidate_scoring_summary (join, total_score, stage)
│   ├── 03b_apply_business_semantics.ipynb   # Metric view / definitions for Genie
│   ├── 04_create_genie_space.ipynb      # HR Analytics Genie Space (+ 04_genie_space.json)
│   ├── 04b_create_uc_functions.ipynb    # UC SQL functions (agent tools)
│   ├── 05_train_ml_model.ipynb          # Train 3 sklearn models → MLflow → @champion → Serving
│   ├── 05b_create_drift_monitor.ipynb   # Lakehouse Monitoring on the model
│   ├── 06_evaluate_register_agent.ipynb # Log/eval (LLM judge) + register + deploy the agent
│   ├── 07_deploy_app.ipynb              # Deploy the (bundle-managed) app's latest source
│   ├── 08_refresh_dashboard.ipynb       # Refresh the Lakeview dashboard
│   ├── 09_grant_app_permissions.ipynb   # Grant app SP: UC + endpoint + Genie access
│   ├── 10_setup_lakebase.ipynb          # Create Lakebase, sync tables, candidate_annotations
│   └── run_all.ipynb                    # Manual orchestrator (not the Job DAG)
├── agent_src/                           # MLflow 3.0 ResponsesAgent (deployed by notebook 06)
│   ├── hire_right_agent.py              # ResponsesAgent: predict + streaming predict_stream
│   ├── config_helper.py                 # os.getenv-based config
│   └── tools/
│       ├── tool_query_genie.py          # Genie Conversation API
│       ├── tool_search_resume.py        # Vector Search RAG
│       ├── tool_predict_score.py        # ML serving endpoint (+ weighted-sum fallback)
│       ├── tool_send_email.py           # Mailgun mailer (AI offer letter)
│       └── tool_query_hr_data.py        # Direct SQL helper
├── app/                                 # Databricks App (deployed by notebook 07)
│   ├── app.py                           # FastAPI: candidates/jobs (Lakebase), chat SSE stream,
│   │                                    #   Genie proxy, resume PDF, offer letter, notes CRUD
│   ├── db.py                            # Lakebase (Postgres) via OAuth credential injection
│   ├── index.html                       # Single-file UI: cockpit, tool cards, notes, composer
│   └── requirements.txt                 # (app config + resource bindings live in databricks.yml)
├── dashboard/
│   └── hiring_analytics.lvdash.json     # Lakeview AI/BI dashboard
├── scripts/
│   ├── deploy.sh                        # One command: bundle deploy + run the workflow
│   └── generate_resumes.py              # Synthetic resume generator
├── slides/                              # Decks (HTML + PDF)
│   ├── hire-right-fe-bar.html / .pdf    # FE Bar value + architecture deck (submit this)
│   ├── hire-right-workshop.html / .pdf  # Longer technical workshop deck
│   └── screenshots/                     # Product screenshots used in the deck
├── solution-brief/                      # Printable customer leave-behind (PDF)
├── reference/                           # Reference material collected before building
├── README.md
└── BUILD.md                             # How this was built (workflow, AI tools, decisions)
```

## Architecture — one connected journey

```
          Synthetic data  (resumes PDF + candidate / job CSVs)
                                    │  Lakeflow · read_files / Auto Loader
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ Unity Catalog — bldemos.hrd_2030                       │
       │ Bronze → Silver → Gold                                 │
       │ PII masking · tags · data quality · lineage            │
       └────────────────────────────────────────────────────────┘
                                    │  synced  (Delta → Postgres)
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ Lakebase (Postgres)                                    │
       │ synced candidate / job tables                          │
       │ + candidate_annotations  (transactional HR notes)      │
       └────────────────────────────────────────────────────────┘
                                    │  low-latency reads · note writes
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ Databricks App   (FastAPI + single-page UI)            │
       │ cockpit · streaming tool cards · Genie panel ·         │
       │ HR notes · AI offer letter                             │
       └────────────────────────────────────────────────────────┘
             │ chat (SSE stream)          │ NL analytics
             ▼                            ▼
    ┌────────────────────────────┐       ┌──────────────────────┐
    │ Hire Right Agent           │       │ Genie Space          │
    │ MLflow 3.0 ResponsesAgent  │       │ NL → SQL analytics   │
    └────────────────────────────┘       └──────────────────────┘
                    │  tools
          ┌─────────┼───────────────┬────────────────────┐
          ▼         ▼               ▼                    ▼
   query_genie  search_resumes  predict_score        send_email
   (→ Genie)    (→ Vector       (→ ML Model          (→ Mailgun
                Search RAG)      Serving:             offer letter)
                                 hire / no-hire)
```

- **Lakeflow + Unity Catalog** — synthetic data ingested with `read_files` (Auto Loader), refined
  through the medallion, governed with PII tags, column masking, data quality, and lineage.
- **Lakebase** — the Gold candidate & job tables are synced Delta→Postgres for millisecond,
  app-facing reads; HR notes are written to a native `candidate_annotations` table.
- **Predictive ML** — three sklearn models trained, tracked in MLflow, best promoted to
  `@champion` in UC and served for real-time hire/no-hire prediction.
- **AI Agent** — an MLflow 3.0 ResponsesAgent orchestrates tools (Genie, resume RAG, ML scoring,
  email); tool calls **stream live** into the app.
- **Genie** — natural-language analytics over the governed Gold layer (in-app panel + agent tool).
- **App** — a single HR cockpit: scoring, DS recommendation, resume viewer, Genie Q&A, HR notes,
  and one-click **AI-drafted offer letters** sent via the agent's mailer tool.

See the interactive **Data Flow** modal in the app, or the architecture slide in the FE Bar deck.

---

## How to run / deploy

Everything is a **Declarative Automation Bundle** (DAB). Prereqs: Databricks CLI authenticated to
the workspace (`databricks auth login --profile DEFAULT --host <workspace>`), and a `.env`
(copy `env.template`).

### One command — deploy + wire everything
```bash
./scripts/deploy.sh              # target=default, profile=DEFAULT
# ./scripts/deploy.sh prod MYPROFILE
```
Two phases:
1. **`bundle deploy`** provisions the Job, dashboard, and the **Databricks App** — attaching the
   app's service-principal **resource bindings** (agent endpoint, warehouse, Genie, Lakebase)
   declaratively from `databricks.yml`.
2. **`bundle run hrd_setup_job`** runs the end-to-end **workflow**, which does all the dynamic
   wiring: Bronze→Gold, Lakebase setup, Genie space, ML train + serve, agent deploy, app-source
   deploy, and — in `grant_app_permissions` (notebook 09) — **grants the current app SP** its UC
   (`USE CATALOG`/`SCHEMA`, `SELECT`, `READ VOLUME`), **Genie `CAN_RUN`** (on the live space),
   warehouse, and endpoint access. That's the piece that makes it turnkey: the workflow grants
   whatever SP the app currently has.

**Task DAG (high level):** `setup → load_bronze → (load_silver ∥ build_vector_index) →
classify_and_quality → build_gold → setup_lakebase → {genie_space, create_uc_functions,
apply_business_semantics} → [train]→ml_model → [agent]→agent → deploy_app → grant_app_permissions
→ drift_monitor / refresh_dashboard`.

### Running the pieces individually
```bash
databricks bundle deploy -t default --profile DEFAULT          # provision
databricks bundle run    hrd_setup_job -t default --profile DEFAULT \
  --params train_model=false,deploy_agent=false                # data-only refresh (no ML/agent)
databricks bundle run    hire_right_app -t default --profile DEFAULT   # (re)deploy app source only
```
`train_model` / `deploy_agent` gate the expensive ML-retrain and agent re-deploy/eval steps.

### Notes & gotchas
- **App SP grants are automatic in the workflow.** A (re)created app gets a **new service
  principal**; `grant_app_permissions` (notebook 09) grants whatever SP the app currently has
  (UC + Genie + warehouse + endpoint), so as long as you deploy via `scripts/deploy.sh` (or run the
  job after deploy) it's turnkey. If you only `bundle deploy` + `bundle run hire_right_app` (app,
  no job), run notebook 09 once to grant the SP.
- **First-time app adoption** — if an app with the same `name` already exists but wasn't created by
  this bundle, `bundle deploy` errors ("app already exists"). Delete it once
  (`databricks apps delete <name>`) so the bundle can own it, then redeploy.
- **New environment** — `warehouse_id`, the target `workspace.host`, and (for a freshly-created
  Genie space) `genie_space_id` are env-specific vars in `databricks.yml`; set them for a new
  workspace. Notebook 04 creates the Genie space and the workflow grants the SP `CAN_RUN` on it
  dynamically; set the `genie_space_id` var to the created id so the app queries the right space.
- **Secret scanner** flags base64 image blobs and Postgres connection-string placeholders
  (`postgresql://<user>:<pass>@<host>`) as false positives — the deck references screenshots by
  relative path and docs use angle-bracket placeholders.

## Local development
```bash
cd app && pip install -r requirements.txt
# .env must have TARGET_CATALOG/SCHEMA, LAKEBASE_*, DATABRICKS_* etc.
uvicorn app:app --reload --port 8000
```
`db.py` mints short-lived Lakebase OAuth credentials from your CLI profile locally, and from the
app service principal when deployed.
