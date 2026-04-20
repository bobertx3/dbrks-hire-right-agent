"""
Tools: HR Analytics UC Functions
Replaces the Genie tool with direct calls to registered UC SQL table-valued functions.
Each @tool maps to one function in bx4.hrd_2030 and executes via the SQL warehouse.
"""
import json
import time
import mlflow
from langchain_core.tools import tool
from config_helper import cfg_get


def _sql(query: str) -> str:
    """Execute a SQL statement via the Databricks Statement Execution API and return results as JSON."""
    from databricks.sdk import WorkspaceClient

    catalog  = cfg_get("target_catalog", "TARGET_CATALOG", "bx4")
    schema   = cfg_get("target_schema",  "TARGET_SCHEMA",  "hrd_2030")
    wh_id    = cfg_get("warehouse_id",   "DATABRICKS_WAREHOUSE_ID", "")

    w = WorkspaceClient()

    if not wh_id:
        warehouses = list(w.warehouses.list())
        if not warehouses:
            return "Error: no SQL warehouse available."
        wh_id = warehouses[0].id

    resp = w.api_client.do(
        "POST",
        "/api/2.0/sql/statements",
        body={
            "statement":  query,
            "warehouse_id": wh_id,
            "catalog":    catalog,
            "schema":     schema,
            "wait_timeout": "30s",
            "on_wait_timeout": "CONTINUE",
        },
    )

    stmt_id = resp.get("statement_id")
    state   = resp.get("status", {}).get("state", "")

    for _ in range(20):
        if state in ("SUCCEEDED", "FAILED", "CANCELED", "CLOSED"):
            break
        time.sleep(3)
        resp  = w.api_client.do("GET", f"/api/2.0/sql/statements/{stmt_id}")
        state = resp.get("status", {}).get("state", "")

    if state != "SUCCEEDED":
        err = resp.get("status", {}).get("error", {})
        return f"Query failed ({state}): {err.get('message', 'unknown error')}"

    result    = resp.get("result", {})
    schema_   = resp.get("manifest", {}).get("schema", {}).get("columns", [])
    col_names = [c["name"] for c in schema_]
    rows      = result.get("data_array", [])

    if not rows:
        return "No results found."

    records = [dict(zip(col_names, row)) for row in rows]
    return json.dumps(records, indent=2)


@tool
@mlflow.trace(span_type="TOOL")
def get_candidate(candidate_id: str) -> str:
    """Look up the full profile and all feature scores for a single candidate by their ID.
    Returns education, experience, leadership, certification, skills, industry relevance,
    interview, and culture fit scores, plus job assignment and hire outcome.
    Use this when the user asks about a specific candidate (e.g. 'Tell me about C004').
    candidate_id: candidate identifier, e.g. 'C001' through 'C020'."""
    catalog = cfg_get("target_catalog", "TARGET_CATALOG", "bx4")
    schema  = cfg_get("target_schema",  "TARGET_SCHEMA",  "hrd_2030")
    return _sql(f"SELECT * FROM {catalog}.{schema}.get_candidate('{candidate_id}')")


@tool
@mlflow.trace(span_type="TOOL")
def get_top_candidates(job_id: str, top_n: int = 5) -> str:
    """Return the top N candidates for a given job role, ranked by total_score descending.
    Use this when the user asks 'who are the best candidates for JR002' or similar ranking questions.
    job_id: job requisition ID, e.g. 'JR001', 'JR002', 'JR003', or 'JR004'.
    top_n: number of candidates to return (default 5)."""
    catalog = cfg_get("target_catalog", "TARGET_CATALOG", "bx4")
    schema  = cfg_get("target_schema",  "TARGET_SCHEMA",  "hrd_2030")
    return _sql(f"SELECT * FROM {catalog}.{schema}.get_top_candidates('{job_id}', {int(top_n)})")


@tool
@mlflow.trace(span_type="TOOL")
def get_candidates_by_job(job_id: str) -> str:
    """Return all candidates assigned to a specific job role with their scores and key qualifications.
    Use this to see every candidate for a role, not just the top ones.
    job_id: job requisition ID, e.g. 'JR001', 'JR002', 'JR003', or 'JR004'."""
    catalog = cfg_get("target_catalog", "TARGET_CATALOG", "bx4")
    schema  = cfg_get("target_schema",  "TARGET_SCHEMA",  "hrd_2030")
    return _sql(f"SELECT * FROM {catalog}.{schema}.get_candidates_by_job('{job_id}')")


@tool
@mlflow.trace(span_type="TOOL")
def get_pipeline_candidates() -> str:
    """Return all active pipeline candidates — those currently in the hiring process
    with no hire decision yet (C011–C020, hired IS NULL).
    Use this when the user asks about open pipeline candidates or candidates awaiting evaluation."""
    catalog = cfg_get("target_catalog", "TARGET_CATALOG", "bx4")
    schema  = cfg_get("target_schema",  "TARGET_SCHEMA",  "hrd_2030")
    return _sql(f"SELECT * FROM {catalog}.{schema}.get_pipeline_candidates()")


@tool
@mlflow.trace(span_type="TOOL")
def get_hire_analytics() -> str:
    """Return aggregate hiring statistics by job role: hire rate, average scores, and candidate counts.
    Only includes historical candidates with a completed hire decision (C001–C010).
    Use this for questions about hire rates, average performance by role, or historical trends."""
    catalog = cfg_get("target_catalog", "TARGET_CATALOG", "bx4")
    schema  = cfg_get("target_schema",  "TARGET_SCHEMA",  "hrd_2030")
    return _sql(f"SELECT * FROM {catalog}.{schema}.get_hire_analytics()")
