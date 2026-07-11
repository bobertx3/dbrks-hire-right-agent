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

```
databricks.yml                # DAB bundle: variables, the pipeline Job, dashboard, 2 targets
notebooks/                    # The build pipeline (run in order; also the Job's tasks)
  00a_setup_schema_volume     # Create UC schema + raw_data volume
  00_load_bronze              # Ingest synthetic CSVs via read_files (Lakeflow/Auto Loader)
  01_load_silver              # Type, clean, mask PII
  01b_build_vector_search     # Resume embedding index (RAG)
  02_apply_data_quality_and_classification
  03_build_gold               # candidate_scoring_summary (join, total_score avg, stage)
  03b_apply_business_semantics# Metric view / business definitions for Genie
  04_create_genie_space       # HR Analytics Genie Space   (+ 04_genie_space.json config)
  04b_create_uc_functions     # UC SQL functions (agent tools)
  05_train_ml_model           # Train 3 sklearn models -> MLflow -> @champion -> Model Serving
  05b_create_drift_monitor    # Lakehouse Monitoring on the model
  06_evaluate_register_agent  # Log/eval (LLM judge) + register + deploy the ResponsesAgent
  07_deploy_app               # Build app.yaml, deploy the App, PATCH resource bindings
  08_refresh_dashboard        # Refresh the Lakeview dashboard
  09_grant_app_permissions    # Grant the app SP UC + endpoint + Genie access
  10_setup_lakebase           # Create Lakebase, sync tables, create candidate_annotations
  run_all                     # Manual orchestrator (not the Job DAG)

agent_src/                    # The MLflow 3.0 ResponsesAgent (deployed by notebook 06)
  hire_right_agent.py         # ToolCalling ResponsesAgent; predict + streaming predict_stream
  config_helper.py            # os.getenv-based config (no ModelConfig)
  tools/
    tool_query_genie.py       # Genie Conversation API
    tool_search_resume.py     # Vector Search RAG
    tool_predict_score.py     # Calls the ML serving endpoint (+ weighted-sum fallback)
    tool_send_email.py        # Mailgun mailer (used by the AI offer letter)
    tool_query_hr_data.py     # Direct SQL helper

app/                          # The Databricks App (deployed by notebook 07)
  app.py                      # FastAPI: candidates/jobs from Lakebase, chat (SSE streaming),
                              #   Genie proxy, resume PDF, offer-letter draft+send, annotations CRUD
  db.py                       # Lakebase (Postgres) access via OAuth credential injection
  index.html                  # Single-file UI: candidate cockpit, streaming tool cards,
                              #   HR notes, AI offer-letter composer, architecture modal
  app.yaml                    # App config + resource bindings (serving, warehouse, genie, database)
  requirements.txt

dashboard/hiring_analytics.lvdash.json   # Lakeview AI/BI dashboard
scripts/generate_resumes.py              # Synthetic resume generator
slides/                                  # FE Bar deck + workshop deck (HTML + PDF) + screenshots
  hire-right-fe-bar.html / .pdf          # The FE Bar business+architecture deck (submit this)
  hire-right-workshop.html / .pdf        # Longer technical workshop deck
  screenshots/                           # Product screenshots used in the deck
solution-brief/                          # Printable customer leave-behind (PDF)
reference/                               # Reference material collected before building
obo-test-app/                            # Diagnostic app used to debug app->endpoint auth
BUILD.md                                 # How this was built (workflow, AI tools, decisions)
```

## Architecture — one connected journey

`Ingest → Unity Catalog (Bronze→Silver→Gold) → Lakebase → ML + AI Agent → Genie → App`

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

### 1. Deploy the bundle
```bash
databricks bundle deploy -t default --profile DEFAULT
```
Targets: `default` (fevm-bobertx3, the live demo — `default: true`) and `prod`.

### 2. Run the pipeline end-to-end
```bash
# Full run incl. ML retrain + agent redeploy + app deploy:
databricks bundle run hrd_setup_job -t default --profile DEFAULT \
  --params train_model=true,deploy_agent=true
```
Job parameters gate the expensive steps: `train_model` (retrain the ML model) and `deploy_agent`
(re-log/eval/deploy the agent). Default both to `false` for a data-only refresh.

**Task DAG (high level):** `setup → load_bronze → (load_silver ∥ build_vector_index) →
classify_and_quality → build_gold → setup_lakebase → {genie_space, create_uc_functions,
apply_business_semantics} → [train]→ml_model → [agent]→agent → deploy_app → grant_app_permissions
→ drift_monitor / refresh_dashboard`.

### 3. Or run notebooks individually
Run in numeric order (`00a → 10`). Minimum path to a working app after data exists:
`03_build_gold → 10_setup_lakebase → 07_deploy_app`.

### Deploy gotchas (learned the hard way)
- **App resource bindings** declared in `app.yaml` don't always attach on deploy (app shows
  `resources: []`). Notebook `07` PATCHes `/api/2.0/apps/{name}` with the resource list after
  deploy so the app SP actually gets its serving/warehouse/Genie/**Lakebase** grants.
- **Redeploying the agent endpoint resets its ACLs** — re-grant the app SP `CAN_QUERY` (notebook
  `09` does this in the full pipeline).
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
