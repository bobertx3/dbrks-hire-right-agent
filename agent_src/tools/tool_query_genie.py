"""
Tool: query_genie
Wraps the Databricks Genie Space as a LangChain tool.

Preferred: databricks_langchain.GenieTool (native SDK, available in >= 0.3)
Fallback:  REST API polling implementation

All instantiation is deferred to invocation time — nothing runs at import.
"""
import os
import time
import requests
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
    sid   = cfg_get("genie_space_id", "GENIE_SPACE_ID")
    host  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN", "")

    if not sid:
        return "Error: GENIE_SPACE_ID is not configured on this endpoint."

    # ── Preferred: databricks_langchain GenieTool ────────────────────────────
    try:
        from databricks_langchain import GenieTool as _GenieTool  # noqa: PLC0415
        genie  = _GenieTool(space_id=sid)
        result = genie._run(question)
        return str(result)
    except Exception:
        pass  # fall through to REST fallback

    # ── Fallback: REST API polling ───────────────────────────────────────────
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        start = requests.post(
            f"{host}/api/2.0/genie/spaces/{sid}/start-conversation",
            headers=headers,
            json={"content": question},
            timeout=30,
        )
        start.raise_for_status()
        data    = start.json()
        conv_id = data["conversation_id"]
        msg_id  = data["message_id"]

        for _ in range(30):
            time.sleep(3)
            poll = requests.get(
                f"{host}/api/2.0/genie/spaces/{sid}/conversations/{conv_id}/messages/{msg_id}",
                headers=headers,
                timeout=30,
            )
            poll.raise_for_status()
            msg    = poll.json()
            status = msg.get("status", "PENDING")

            if status == "COMPLETED":
                parts = []
                for att in msg.get("attachments", []):
                    if att.get("text"):
                        parts.append(att["text"]["content"])
                    elif att.get("query"):
                        parts.append(f"SQL: {att['query'].get('query', '')}")
                return "\n".join(parts) or "Query completed with no text response."

            if status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
                return f"Genie query {status.lower()}: {msg.get('error', 'Unknown error')}"

        return "Genie query timed out after 90 seconds."

    except requests.HTTPError as e:
        return f"Genie API error (HTTP {e.response.status_code}): {e.response.text[:300]}"
    except Exception as e:
        return f"Error querying Genie: {str(e)}"
