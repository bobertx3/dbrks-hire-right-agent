"""
Tool: query_genie
Wraps the Databricks Genie Space as a LangChain tool.

Preferred: databricks_langchain.GenieTool (native SDK, available in >= 0.3)
Fallback:  WorkspaceClient().api_client.do() — handles endpoint credential
           passthrough automatically (no DATABRICKS_TOKEN env var needed).

All instantiation is deferred to invocation time — nothing runs at import.
"""
import time
import mlflow
from langchain_core.tools import tool
from config_helper import cfg_get


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

    # ── Preferred: databricks_langchain GenieTool ────────────────────────────
    try:
        from databricks_langchain import GenieTool as _GenieTool
        genie  = _GenieTool(space_id=sid)
        result = genie._run(question)
        return str(result)
    except Exception:
        pass  # fall through to SDK REST fallback

    # ── Fallback: WorkspaceClient REST polling ───────────────────────────────
    # WorkspaceClient() automatically uses endpoint credentials — no token env var needed.
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
