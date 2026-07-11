# BUILD.md — How this was built

A short account of the workflow, the AI tooling, the key decisions, and where AI made the biggest
difference. (Optional per the FE Bar rubric — included for transparency.)

## Origin & purpose

This started as a **customer demo** for an **HR-Modernization program** where Databricks was being
evaluated as the customer's Data & AI platform. The brief was to show the platform's **ML, MLOps,
and GenAI** capabilities working together on a realistic HR use case — not as isolated features,
but as one connected journey from raw data to a decision an HR leader would actually make.

It was later **extended for the FE Bar**, where the main addition was the **Lakebase** operational
serving layer (so the app reads from a low-latency Postgres synced from the governed lakehouse, and
persists HR notes transactionally), plus the AI-drafted **offer letter**, **live streaming tool
calls** in the app, and inline **HR notes** (add/delete).

## Workflow — how it was actually built

1. **Installed the AI Dev Kit.** The Databricks FE AI Dev Kit (`ai-dev-kit`) set up the local
   environment: the `databricks-*` skills (bundles, apps, model serving, genie, lakebase, vector
   search, mlflow-evaluation, unity-catalog, etc.), the Databricks MCP server, and the supporting
   plugins.
2. **Collected reference material** into `reference/` (architecture patterns, the medallion/ML-Ops
   charter) so the assistant had grounding for the target design before any code was written.
3. **Drove the build with Isaac** — the AI Dev Kit's Claude Code-based pair-programmer — through
   iterative natural-language prompts. Isaac scaffolded the DAB bundle, wrote the medallion
   notebooks, the ResponsesAgent and its tools, the Genie space, the ML training + serving, and the
   Databricks App, then deployed and verified each piece against the live workspace via the
   Databricks MCP tools.
4. **Iterated against reality.** Each layer was deployed and exercised end-to-end (run the job,
   query the endpoint, click through the app) and fixed based on what actually happened — several of
   the "deploy gotchas" in the README were discovered this way.

> The exact original prompt wasn't captured — this was first developed back in March/April. It
> included PII requirements and data masking / attribute-based access control (ABAC) to demonstrate
> how to secure sensitive HR data such as phone numbers, salary, etc.

## AI tools & prompts used

- **Isaac / Claude Code** (the AI Dev Kit agent) — primary author of the notebooks, agent, app, and
  bundle; also ran deploys and live verification.
- **AI Dev Kit skills** — `databricks-bundles/dabs`, `databricks-apps`, `databricks-model-serving`,
  `databricks-genie`, `databricks-lakebase`, `databricks-vector-search`,
  `databricks-mlflow-evaluation`, `databricks-unity-catalog`, `databricks-synthetic-data-gen`, and
  `db-presentation` (for the slide decks).
- **Databricks MCP** — `execute_sql`, `manage_*`, serving/endpoint and app management used to build
  and verify resources against the live workspace.
- **Prompting style** — outcome-first prompts ("build/verify X end-to-end"), then tight
  correction loops when a deploy or query behaved differently than expected.

## Key decisions

- **Medallion + Unity Catalog first.** Governance (PII tags, `mask_phone` column masking, data
  quality, lineage) is baked into Silver so the whole story is auditable — important for a regulated
  (pharma) buyer.
- **ResponsesAgent (MLflow 3.0), not LangGraph.** A simple, transparent Python tool-calling loop
  that's easy to trace and to stream. `predict_stream` emits each tool call/result incrementally so
  the UI can render live tool cards.
- **Lakebase for the operational layer.** The app serves candidate/job reads from Postgres synced
  from the governed Gold tables (snappy), and writes HR notes to a native transactional table that
  joins back to candidate data — the analytical/operational split without a second system to govern.
- **Computed score vs. the model, on purpose.** The UI "Total Score" is a transparent average of
  the 8 category scores; the model returns the hire/no-hire decision. Keeping them separate makes
  the demo explainable (and honest about what the model does).
- **App deployed the declarative way.** The Databricks App is a first-class DAB `apps` resource in
  `databricks.yml` with its `resources:` block, so `bundle deploy` attaches the service-principal
  bindings (serving endpoint, warehouse, Genie, Lakebase) automatically — no post-deploy PATCH.

## Where AI made the biggest difference

- **The agent + streaming UX.** Building the ResponsesAgent, its four tools, the SSE streaming
  backend route, and the live tool-call cards in vanilla JS — end to end, deployed and verified — was
  dramatically faster with AI, including diagnosing the "burst vs. progressive" streaming behavior.
- **Lakebase integration.** Standing up the instance, synced tables, the transactional annotations
  table, OAuth credential injection, and the app rewrite — and finding the app-binding / ACL-reset
  gotchas — was largely AI-driven against the live workspace.
- **Governance + pipeline plumbing.** The medallion notebooks, PII masking/tagging, UC functions,
  Genie space config, and the DAB job DAG were scaffolded and wired up by AI.
