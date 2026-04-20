"""
OBO Test App — diagnoses whether x-forwarded-access-token reaches
the agent endpoint and whether on_behalf_of_user Genie calls succeed.
"""
import os
import logging

import requests as _requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OBO Test", version="1.0.0")

AGENT_ENDPOINT = os.getenv("DATABRICKS_AGENT_ENDPOINT", "hire-right-agent-endpoint")


def _host() -> str:
    h = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    return h if h.startswith("http") else f"https://{h}"


class AskRequest(BaseModel):
    question: str
    use_user_token: bool = True


GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f13a0f6a081fabbea933cfb0db1d01")


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html><head><style>
      body{font-family:monospace;max-width:800px;margin:40px auto;padding:0 20px}
      button{padding:10px 20px;margin:8px 4px;cursor:pointer;font-size:14px}
      pre{background:#f4f4f4;padding:16px;white-space:pre-wrap;word-break:break-all}
    </style></head><body>
    <h2>OBO / Genie Debug App</h2>
    <h3>Agent endpoint tests</h3>
    <button onclick="test(true)">Ask agent (user token)</button>
    <button onclick="test(false)">Ask agent (app SP token)</button>
    <h3>Genie direct tests</h3>
    <button onclick="genie(true)">Ask Genie (user token)</button>
    <button onclick="genie(false)">Ask Genie (app SP token)</button>
    <button onclick="location.href='/debug'">Check /debug</button>
    <pre id="out">Click a button to test...</pre>
    <script>
    async function post(url, body) {
      document.getElementById('out').textContent = 'Calling... (may take 10-30s)';
      try {
        const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        const data = await r.json();
        document.getElementById('out').textContent = JSON.stringify(data, null, 2);
      } catch(e) { document.getElementById('out').textContent = 'Error: ' + e; }
    }
    function test(u) { post('/ask', {question:'Who are the top 3 candidates by total score?', use_user_token:u}); }
    function genie(u) { post('/genie', {question:'Who are the top 3 candidates by total score?', use_user_token:u}); }
    </script>
    </body></html>
    """


@app.get("/debug")
def debug(request: Request):
    user_token = request.headers.get("x-forwarded-access-token")
    relevant = {
        k: (v[:20] + "...") if k.lower() in ("authorization", "x-forwarded-access-token") and v else v
        for k, v in request.headers.items()
        if k.lower().startswith("x-") or k.lower() == "authorization"
    }

    result = {
        "x_forwarded_access_token_present": bool(user_token),
        "x_forwarded_access_token_preview": (user_token[:20] + "...") if user_token else None,
        "relevant_headers": relevant,
    }

    try:
        me = WorkspaceClient(config=Config()).current_user.me()
        result["app_sp_identity"] = {"user_name": me.user_name, "id": me.id}
    except Exception as e:
        result["app_sp_identity_error"] = str(e)

    if user_token:
        try:
            resp = _requests.get(
                f"{_host()}/api/2.0/preview/scim/v2/Me",
                headers={"Authorization": f"Bearer {user_token}"},
                timeout=10,
            )
            me = resp.json()
            result["user_token_identity"] = {"user_name": me.get("userName"), "id": me.get("id")}
        except Exception as e:
            result["user_token_identity_error"] = str(e)

    return result


@app.post("/ask")
def ask(body: AskRequest, request: Request):
    user_token = request.headers.get("x-forwarded-access-token")
    host = _host()

    if body.use_user_token and user_token:
        token_used = "user_token (x-forwarded-access-token)"
        try:
            resp = _requests.post(
                f"{host}/serving-endpoints/{AGENT_ENDPOINT}/invocations",
                headers={"Authorization": f"Bearer {user_token}", "Content-Type": "application/json"},
                json={"input": [{"role": "user", "content": body.question}]},
                timeout=120,
            )
            if not resp.text.strip():
                return {"token_used": token_used, "http_status": resp.status_code, "error": "empty response body"}
            try:
                result = resp.json()
            except Exception:
                return {"token_used": token_used, "http_status": resp.status_code, "raw": resp.text[:500]}
            output = result.get("output", [])
            reply = next(
                (item.get("content", "") for item in reversed(output)
                 if isinstance(item, dict) and item.get("content")),
                str(result)[:500],
            )
            return {"token_used": token_used, "http_status": resp.status_code, "reply": reply}
        except Exception as e:
            raise HTTPException(status_code=500, detail={"token_used": token_used, "error": str(e)})
    else:
        token_used = "app_sp_token"
        if body.use_user_token and not user_token:
            token_used += " [user token unavailable, fell back]"
        try:
            result = WorkspaceClient(config=Config()).api_client.do(
                "POST",
                f"/serving-endpoints/{AGENT_ENDPOINT}/invocations",
                body={"input": [{"role": "user", "content": body.question}]},
            )
            output = result.get("output", [])
            reply = next(
                (item.get("content", "") for item in reversed(output)
                 if isinstance(item, dict) and item.get("content")),
                str(result)[:500],
            )
            return {"token_used": token_used, "reply": reply}
        except Exception as e:
            raise HTTPException(status_code=500, detail={"token_used": token_used, "error": str(e)})


@app.post("/genie")
def genie(body: AskRequest, request: Request):
    import time
    sid = GENIE_SPACE_ID
    user_token = request.headers.get("x-forwarded-access-token")
    host = _host()

    if body.use_user_token and user_token:
        token = user_token
        token_used = "user_token (x-forwarded-access-token)"
    else:
        token_used = "app_sp_token"
        if body.use_user_token and not user_token:
            token_used += " [user token unavailable, fell back]"
        client_id = os.getenv("DATABRICKS_CLIENT_ID", "")
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            return {"token_used": token_used, "error": "DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET not set"}
        tok_resp = _requests.post(
            f"{host}/oidc/v1/token",
            data={"grant_type": "client_credentials", "scope": "all-apis"},
            auth=(client_id, client_secret),
            timeout=15,
        )
        if not tok_resp.ok:
            return {"token_used": token_used, "error": f"OAuth token fetch failed ({tok_resp.status_code}): {tok_resp.text[:300]}"}
        token = tok_resp.json().get("access_token", "")
        if not token:
            return {"token_used": token_used, "error": f"No access_token in response: {tok_resp.text[:300]}"}

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        start_resp = _requests.post(
            f"{host}/api/2.0/genie/spaces/{sid}/start-conversation",
            headers=headers,
            json={"content": body.question},
            timeout=30,
        )
        if not start_resp.ok:
            return {"token_used": token_used, "http_status": start_resp.status_code, "error": start_resp.text[:500]}
        start = start_resp.json()
        conv_id = start["conversation_id"]
        msg_id = start["message_id"]
    except Exception as e:
        return {"token_used": token_used, "error": f"start-conversation failed: {e}"}

    for i in range(30):
        time.sleep(3)
        try:
            msg_resp = _requests.get(
                f"{host}/api/2.0/genie/spaces/{sid}/conversations/{conv_id}/messages/{msg_id}",
                headers=headers,
                timeout=15,
            )
            if not msg_resp.ok:
                return {"token_used": token_used, "poll_attempt": i, "http_status": msg_resp.status_code, "error": msg_resp.text[:500]}
            msg = msg_resp.json()
        except Exception as e:
            return {"token_used": token_used, "error": f"poll failed: {e}"}

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
            return {"token_used": token_used, "status": "COMPLETED", "result": "\n".join(parts) or "no text"}
        if status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
            return {"token_used": token_used, "status": status, "error": msg.get("error"), "full_msg": msg}

    return {"token_used": token_used, "error": "timed out after 90s"}
