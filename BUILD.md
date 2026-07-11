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
calls** in the app, and inline **HR notes** (add/delete). The FE Bar prep also produced the
**value + architecture deck** (`slides/hire-right-fe-bar.*`, generated from a Python builder using
the `db-presentation` design system with real product screenshots), a candidate-page redesign
(score ring, centered composer), and a final **repo cleanup** (removed ~5.7k build/tooling
artifacts so the repo contains only the actual work).

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

> The exact original seed prompt isn't captured verbatim, but it was of the form:
> *"Build an end-to-end predictive-hiring demo on Databricks for a fictional pharma HR org —
> medallion pipeline in Unity Catalog with PII governance, a hire/no-hire ML model served on
> Model Serving, a Genie space for NL analytics, an MLflow ResponsesAgent with tools for Genie,
> resume RAG, ML scoring and email, and a Databricks App front end — all deployable as a DAB."*
> The FE Bar work was driven by follow-up prompts (e.g., *"add a Lakebase serving layer synced
> from the gold tables and a transactional notes table,"* *"stream the agent's tool calls into the
> app,"* *"add an AI-powered offer letter that sends via the agent's mailer tool"*).

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
- **App deploy hardening.** App resource bindings are PATCHed post-deploy (they don't reliably
  attach from `app.yaml`), and the agent-endpoint ACL is re-granted after redeploys.
- **Decks as code.** The FE Bar deck is generated by a Python builder using the `db-presentation`
  design system, with real product screenshots referenced by relative path (avoids base64 that trips
  the secret scanner) and exported to PDF via a headless-Chrome script.

## Where AI made the biggest difference

- **The agent + streaming UX.** Building the ResponsesAgent, its four tools, the SSE streaming
  backend route, and the live tool-call cards in vanilla JS — end to end, deployed and verified — was
  dramatically faster with AI, including diagnosing the "burst vs. progressive" streaming behavior.
- **Lakebase integration.** Standing up the instance, synced tables, the transactional annotations
  table, OAuth credential injection, and the app rewrite — and finding the app-binding / ACL-reset
  gotchas — was largely AI-driven against the live workspace.
- **Governance + pipeline plumbing.** The medallion notebooks, PII masking/tagging, UC functions,
  Genie space config, and the DAB job DAG were scaffolded and wired up by AI.
- **The decks.** Both the workshop and FE Bar decks (design system, architecture diagram with
  animated connectors, screenshot-led slides, PDF export) were produced with AI.
