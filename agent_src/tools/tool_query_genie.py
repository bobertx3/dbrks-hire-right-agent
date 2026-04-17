"""
Tool: query_genie
Wraps the Databricks Genie Space as a LangChain tool.

Uses WorkspaceClient REST polling as the primary path — it auto-uses endpoint
credentials (SP token or on_behalf_of_user forwarded token) without needing
explicit DATABRICKS_TOKEN env var.

GenieTool from databricks_langchain is tried first; if it returns an error
string or raises, we fall back to WorkspaceClient REST.
"""
import time
import mlflow
from langchain_core.tools import tool
from config_helper import cfg_get

_ERROR_HINTS = ("error", "exception", "failed", "denied", "blocked", "not found", "invalid", "permission")


@tool
@mlflow.trace(span_type="TOOL")
def query_genie(question: str) -> str:
    """Query the Jackson and Jackson HR Digital HR Analytics Genie Space to answer data questions about
    candidates, hiring scores, ML predictions, job requirements, and HR metrics.
    Use this tool whenever the user asks for data about candidates, comparisons,
    hire rates, scores, certifications, or any structured HR analytics.
    The Genie Space has access to: candidates, job_requirements, training_data,
    and candidate_scoring_summary tables including all ML model scores."""
    sid = cfg_get("genie_space_id", "GENIE_SPACE_ID")

    if not sid:
        return "Error: GENIE_SPACE_ID is not configured on this endpoint."

    # ── Attempt 1: databricks_langchain GenieTool ────────────────────────────
    try:
        from databricks_langchain import GenieTool as _GenieTool
        genie  = _GenieTool(space_id=sid)
        result = genie._run(question)
        result_str = str(result)
        # If GenieTool returned an error string, fall through to REST
        lower = result_str.lower()
        if not any(hint in lower for hint in _ERROR_HINTS):
            return result_str
        import logging
        logging.getLogger(__name__).warning("GenieTool returned error response, using REST fallback: %s", result_str[:200])
    except Exception as _genie_tool_err:
        import logging
        logging.getLogger(__name__).warning("GenieTool raised, using REST fallback: %s", _genie_tool_err)

    # ── Attempt 2: WorkspaceClient REST polling ──────────────────────────────
    # WorkspaceClient() on a serving endpoint automatically uses endpoint
    # credentials — no explicit token needed. With on_behalf_of_user=True in
    # the model resources, it forwards the calling user's token to Genie.
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        start = w.api_client.do(
            "POST",
            f"/api/2.0/genie/spaces/{sid}/start-conversation",
            body={"content": question},
        )
        conv_id = start["conversation_id"]
        msg_id  = start["message_id"]

        for _ in range(30):
            time.sleep(3)
            msg    = w.api_client.do(
                "GET",
                f"/api/2.0/genie/spaces/{sid}/conversations/{conv_id}/messages/{msg_id}",
            )
            status = msg.get("status", "PENDING")

            if status == "COMPLETED":
                parts = []
                for att in msg.get("attachments", []):
                    if att.get("text"):
                        parts.append(att["text"]["content"])
                    elif att.get("query"):
                        q = att["query"]
                        if q.get("description"):
                            parts.append(q["description"])
                        if q.get("query"):
                            parts.append(f"SQL: {q['query']}")
                return "\n".join(parts) or "Query completed with no text response."

            if status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
                return f"Genie query {status.lower()}: {msg.get('error', 'Unknown error')}"

        return "Genie query timed out after 90 seconds."

    except Exception as e:
        return f"Error querying Genie: {str(e)}"
