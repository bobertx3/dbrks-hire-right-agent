"""
Hire Right Agent — Databricks App Backend (v4)
Jackson and Jackson HR Digital
FastAPI: serves candidate & job data from Lakebase (low latency), persists HR
annotations to a transactional Lakebase table, proxies chat to the agent
endpoint, and calls the Genie Conversation API with multi-turn support.
"""
import os
import re
import json
import time
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

from db import execute_query, execute_write, ensure_annotations_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hire Right Agent", version="4.0.0")

# ── Config ─────────────────────────────────────────────────────────────────────
AGENT_ENDPOINT = os.getenv("DATABRICKS_AGENT_ENDPOINT", "hire-right-agent-endpoint")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f170b7d5dc143995f2df49ee1fbc22")
TARGET_CATALOG = os.getenv("TARGET_CATALOG", "bx4")
TARGET_SCHEMA  = os.getenv("TARGET_SCHEMA", "hrd_2030")
WAREHOUSE_ID   = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
LLM_ENDPOINT   = os.getenv("LLM_ENDPOINT", "databricks-gpt-5-4")


def get_client() -> WorkspaceClient:
    return WorkspaceClient(config=Config())


@app.on_event("startup")
def _on_startup():
    # Belt-and-suspenders: the setup notebook creates this table, but ensure it
    # exists so annotations work even if the app is deployed first. Run it off
    # the startup path in a daemon thread so a slow/unavailable Lakebase never
    # delays the app coming up (matters on scale-from-zero cold starts).
    import threading
    threading.Thread(target=ensure_annotations_table, daemon=True).start()


# ── Models ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    conversation_history: list

class GenieRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None   # pass to continue a multi-turn session

class GenieResponse(BaseModel):
    answer: str
    sql: Optional[str] = None
    suggested_questions: list = []
    conversation_id: Optional[str] = None

class AnnotationRequest(BaseModel):
    note: str
    author: Optional[str] = None

class OfferDraftRequest(BaseModel):
    candidate_id: str

class OfferSendRequest(BaseModel):
    to: str
    subject: str
    body: str


# ── Response parsing ───────────────────────────────────────────────────────────
def _extract_agent_reply(pred) -> str:
    """
    Extract the last text reply from a ResponsesAgent endpoint prediction.
    Handles multiple serialisation formats emitted by MLflow / Databricks serving.
    """
    if not isinstance(pred, dict):
        return str(pred) if pred is not None else ""

    # Format 1 — ResponsesAgent: {"output": [{type, content, role, ...}]}
    output = pred.get("output", [])
    for item in reversed(output):
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            texts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "output_text"
            ]
            joined = " ".join(t for t in texts if t)
            if joined:
                return joined

    # Format 2 — OpenAI chat completions: {"choices": [{message: {content}}]}
    choices = pred.get("choices", [])
    if choices:
        c = choices[0].get("message", {}).get("content", "")
        if c:
            return c

    # Format 3 — messages list
    msgs = pred.get("messages", [])
    if msgs:
        c = msgs[-1].get("content", "")
        if c:
            return c

    # Format 4 — direct content field
    if "content" in pred:
        return str(pred["content"])

    return str(pred)


# ── Genie response formatter ───────────────────────────────────────────────────
def _reformat_as_markdown(w: WorkspaceClient, text: str) -> str:
    """Reformat a Genie prose answer into clean, scannable markdown."""
    try:
        result = w.api_client.do(
            "POST",
            f"/serving-endpoints/{LLM_ENDPOINT}/invocations",
            body={
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a markdown formatter for HR analytics responses. "
                            "Convert the user's prose response into clean, scannable markdown. "
                            "CRITICAL RULES: "
                            "1) ONLY include information that is explicitly present in the input — never infer, add, or show fields with N/A or unknown values. "
                            "2) When individual scores are listed, format as a two-column markdown table: | Metric | Score |. "
                            "3) When listing multiple candidates with a single value (e.g. total score), use a simple ranked list: **Name** — Score X, Decision. "
                            "4) Use **bold** for candidate names and hiring decisions. "
                            "5) Keep intro text to one short line. "
                            "6) Return only the formatted markdown — no preamble."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "max_tokens": 512,
                "temperature": 0,
            },
        )
        choices = result.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content:
                return content
    except Exception as e:
        logger.warning("Genie reformat skipped: %s", e)
    return text


# ── Candidate Data (served from Lakebase synced table) ──────────────────────────
def _to_bool(v):
    """hired: int 1/0 -> True/False; NULL -> None (UI uses `hired !== null`)."""
    if v is None:
        return None
    return bool(v)


def _shape_candidate(r: dict) -> dict:
    """Map a candidate_scoring_summary row to the shape the UI expects."""
    return {
        "id":            r.get("candidate_id"),
        "name":          r.get("full_name"),
        "title":         r.get("current_title"),
        "company":       r.get("current_company"),
        "location":      r.get("location"),
        "education":     r.get("education_level"),
        "certifications": r.get("certifications"),
        "job_id":        r.get("job_id"),
        "job_title":     r.get("job_title"),
        "total_score":   r.get("total_score"),
        "stage":         r.get("stage"),
        "hired":         _to_bool(r.get("hired")),
        "scores": {
            "education":      r.get("education_score"),
            "experience":     r.get("experience_score"),
            "leadership":     r.get("leadership_score"),
            "certifications": r.get("certification_score"),
            "skills_match":   r.get("skills_match_score"),
            "industry":       r.get("industry_relevance_score"),
            "interview":      r.get("interview_score"),
            "culture_fit":    r.get("culture_fit"),
        },
    }


# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "healthy", "agent_endpoint": AGENT_ENDPOINT}


@app.get("/api/candidates")
async def get_candidates():
    """Return all candidates from the Lakebase synced scoring-summary table."""
    try:
        rows = execute_query("""
            SELECT candidate_id, full_name, current_title, current_company, location,
                   education_level, certifications, job_id, job_title,
                   total_score, stage, hired,
                   education_score, experience_score, leadership_score, certification_score,
                   skills_match_score, industry_relevance_score, interview_score, culture_fit
            FROM candidate_scoring_summary
            ORDER BY candidate_id
        """)
        return {"candidates": [_shape_candidate(r) for r in rows]}
    except Exception as e:
        logger.error("Candidates error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Candidate Annotations (transactional Lakebase table) ────────────────────────
@app.get("/api/candidates/{candidate_id}/annotations")
async def get_annotations(candidate_id: str):
    """List HR annotations for a candidate, joined to candidate info."""
    try:
        # LEFT JOIN so a note is never hidden if the candidate row is briefly
        # absent from the synced table (e.g. sync lag); candidate_name may be NULL.
        rows = execute_query("""
            SELECT a.id, a.candidate_id, a.note, a.author, a.created_at,
                   c.full_name AS candidate_name, c.job_title
            FROM candidate_annotations a
            LEFT JOIN candidate_scoring_summary c ON c.candidate_id = a.candidate_id
            WHERE a.candidate_id = :cid
            ORDER BY a.created_at DESC
        """, {"cid": candidate_id})
        return {"annotations": rows}
    except Exception as e:
        logger.error("Get annotations error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/candidates/{candidate_id}/annotations")
async def add_annotation(candidate_id: str, req: AnnotationRequest):
    """Add an HR annotation (note) to a candidate record."""
    note = (req.note or "").strip()
    if not note:
        raise HTTPException(status_code=400, detail="Note cannot be empty.")
    try:
        # Guard against notes on unknown candidates (keeps the FK-style join valid).
        exists = execute_query(
            "SELECT 1 FROM candidate_scoring_summary WHERE candidate_id = :cid LIMIT 1",
            {"cid": candidate_id},
        )
        if not exists:
            raise HTTPException(status_code=404, detail=f"Unknown candidate {candidate_id}")

        rows = execute_write("""
            INSERT INTO candidate_annotations (candidate_id, note, author)
            VALUES (:cid, :note, :author)
            RETURNING id, candidate_id, note, author, created_at
        """, {"cid": candidate_id, "note": note, "author": req.author or "HR Manager"})
        return {"annotation": rows[0] if rows else None}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Add annotation error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── AI-Powered Offer Letter ─────────────────────────────────────────────────────
def _llm_complete(w: WorkspaceClient, system: str, user: str, max_tokens: int = 900) -> str:
    """Single-shot completion against the LLM serving endpoint."""
    result = w.api_client.do(
        "POST",
        f"/serving-endpoints/{LLM_ENDPOINT}/invocations",
        body={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.4,
        },
    )
    choices = result.get("choices", []) if isinstance(result, dict) else []
    if choices:
        return choices[0].get("message", {}).get("content", "") or ""
    return ""


def _offer_salary(row: dict) -> int:
    smin, smax = row.get("salary_min"), row.get("salary_max")
    try:
        if smin and smax:
            return int(round(((float(smin) + float(smax)) / 2) / 1000.0)) * 1000
        if smax:
            return int(float(smax))
        if smin:
            return int(float(smin))
    except (TypeError, ValueError):
        pass
    return 185000


@app.post("/api/offer-letter/draft")
async def offer_letter_draft(req: OfferDraftRequest):
    """AI-draft an offer letter (HTML body + subject) for a candidate."""
    cid = (req.candidate_id or "").upper().strip()
    try:
        rows = execute_query("""
            SELECT candidate_id, full_name, first_name, email, job_title, department,
                   current_company, salary_min, salary_max
            FROM candidate_scoring_summary WHERE candidate_id = :cid LIMIT 1
        """, {"cid": cid})
        if not rows:
            raise HTTPException(status_code=404, detail=f"Unknown candidate {cid}")
        r = rows[0]
        name = r.get("full_name") or cid
        first = r.get("first_name") or name.split(" ")[0]
        job_title = r.get("job_title") or "the role"
        dept = r.get("department") or "Human Resources"
        salary = _offer_salary(r)

        w = get_client()
        system = (
            "You are an HR talent-acquisition assistant writing a formal, warm job offer letter "
            "for Jackson & Jackson, a leading pharmaceutical company. "
            "Return ONLY the letter body as clean semantic HTML using <p>, <strong>, and <ul><li> tags. "
            "Do NOT include <html>, <head>, <body>, markdown, or code fences. "
            "Keep it professional and concise (about 200-280 words)."
        )
        user = (
            f"Write an offer letter to {first} for the position of {job_title} in the {dept} "
            f"organization at Jackson & Jackson. Annual base salary: ${salary:,}. "
            "Include: warm congratulations addressing them by first name; the role title and base salary; "
            "a brief mention of benefits (comprehensive health coverage, 401(k) match, and equity); "
            "a note that employment is at-will; and a closing line asking them to sign and return the letter "
            "within two weeks. Sign off from 'The Jackson & Jackson Talent Acquisition Team'."
        )
        body = _llm_complete(w, system, user, max_tokens=900).strip()
        # strip accidental code fences
        if body.startswith("```"):
            body = body.split("```", 2)[1] if body.count("```") >= 2 else body.replace("```", "")
            body = body.lstrip("html").strip()
        subject = f"Your Offer from Jackson & Jackson — {job_title}"
        return {"to": r.get("email") or "", "subject": subject, "body": body,
                "salary": salary, "candidate_name": name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Offer draft error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/offer-letter/send")
async def offer_letter_send(req: OfferSendRequest):
    """Send the offer letter via the agent's send_email (Mailer) tool.

    The Mailgun credentials live on the agent serving endpoint, so we route the
    send through the agent and instruct it to call send_email verbatim.
    """
    to = (req.to or "").strip()
    subject = (req.subject or "").strip()
    body = (req.body or "").strip()
    if "@" not in to:
        raise HTTPException(status_code=400, detail="A valid recipient email is required.")
    if not body:
        raise HTTPException(status_code=400, detail="The offer letter body is empty.")
    try:
        w = get_client()
        instruction = (
            "Call the send_email tool exactly once, then reply only with the tool's result. "
            "Use these arguments verbatim — do NOT edit, summarize, translate, or reformat the body; "
            "pass the HTML between the <<<BODY>>> markers exactly as-is (excluding the markers).\n\n"
            f"to = {to}\n"
            f"subject = {subject}\n"
            "body =\n<<<BODY>>>\n" + body + "\n<<<BODY>>>"
        )
        result = w.api_client.do(
            "POST",
            f"/serving-endpoints/{AGENT_ENDPOINT}/invocations",
            body={"input": [{"role": "user", "content": instruction}]},
        )
        reply = _extract_agent_reply(result) if isinstance(result, dict) else str(result)
        low = (reply or "").lower()
        ok = ("sent successfully" in low) or ("✅" in (reply or "")) or ("mailgun id" in low)
        return {"ok": ok, "detail": reply or "No response from agent."}
    except Exception as e:
        logger.error("Offer send error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Forward a message to the Hire Right agent serving endpoint."""
    try:
        w = get_client()
        history = list(request.conversation_history)
        history.append({"role": "user", "content": request.message})

        result = w.api_client.do(
            "POST",
            f"/serving-endpoints/{AGENT_ENDPOINT}/invocations",
            body={"input": history},
        )

        reply = "No response from the agent."
        if isinstance(result, dict):
            reply = _extract_agent_reply(result) or reply

        history.append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply, conversation_history=history)

    except Exception as e:
        logger.error("Chat error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming chat (Server-Sent Events) ─────────────────────────────────────────
def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _item_text(item: dict) -> str:
    """Pull display text out of a Responses 'message'/text output item."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") in ("output_text", "text") and item.get("text"):
        return item["text"]
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type") in ("output_text", "text")
        ]
        joined = " ".join(p for p in parts if p)
        if joined:
            return joined
    return item.get("text") or ""


def _normalize_item(item: dict) -> Optional[dict]:
    """Map a Responses output item to the compact event shape the browser renders."""
    if not isinstance(item, dict):
        return None
    itype = item.get("type", "")
    if itype == "function_call":
        return {"type": "tool_call", "name": item.get("name"),
                "call_id": item.get("call_id"), "arguments": item.get("arguments")}
    if itype == "function_call_output":
        return {"type": "tool_result", "call_id": item.get("call_id"),
                "output": item.get("output")}
    text = _item_text(item)
    if text:
        return {"type": "text", "text": text}
    return None


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the agent's tool calls + answer to the browser as SSE.

    Invokes the ResponsesAgent endpoint in streaming mode and forwards each
    `response.output_item.done` event (function call, tool result, text) as a
    compact SSE event. Falls back to a single JSON invocation if the endpoint
    does not return an event-stream.
    """
    w = get_client()
    host = w.config.host.rstrip("/")
    history = list(request.conversation_history)
    history.append({"role": "user", "content": request.message})

    url = f"{host}/serving-endpoints/{AGENT_ENDPOINT}/invocations"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    headers.update(w.config.authenticate())  # Authorization: Bearer ...
    payload = {"input": history, "stream": True}

    async def gen():
        try:
            timeout = httpx.Timeout(300.0, connect=30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    ctype = resp.headers.get("content-type", "")
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        low = body.lower()
                        msg = ("The agent is warming up (cold start can take a minute or two). "
                               "Please try again shortly.") if ("upstream" in low or "timeout" in low) \
                              else body[:400]
                        yield _sse({"type": "error", "message": msg})
                        return

                    if "text/event-stream" in ctype:
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                evt = json.loads(data)
                            except Exception:
                                continue
                            if evt.get("type") == "response.output_item.done":
                                out = _normalize_item(evt.get("item") or {})
                                if out:
                                    yield _sse(out)
                    else:
                        # Non-streaming fallback: parse the whole JSON and replay items.
                        full = json.loads((await resp.aread()).decode("utf-8", "replace"))
                        for item in (full.get("output") or []):
                            out = _normalize_item(item)
                            if out:
                                yield _sse(out)
            yield _sse({"type": "done"})
        except Exception as e:
            logger.error("Stream error: %s", str(e), exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so events flush live
        },
    )


@app.post("/api/genie", response_model=GenieResponse)
def ask_genie(request: GenieRequest):
    """
    Ask the Genie Space directly via the Conversation API.
    Pass conversation_id to continue an existing multi-turn conversation.
    """
    try:
        w = get_client()
        conv_id = request.conversation_id

        if conv_id:
            # Continue existing conversation
            resp = w.api_client.do(
                "POST",
                f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conv_id}/messages",
                body={"content": request.question},
            )
            msg_id = resp["message_id"]
        else:
            # Start new conversation
            resp = w.api_client.do(
                "POST",
                f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation",
                body={"content": request.question},
            )
            conv_id = resp["conversation_id"]
            msg_id  = resp["message_id"]

        # Poll for completion
        for _ in range(30):
            time.sleep(3)
            msg    = w.api_client.do(
                "GET",
                f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conv_id}/messages/{msg_id}",
            )
            status = msg.get("status", "PENDING")

            if status == "COMPLETED":
                parts = []
                sql_query = None
                suggested_questions = []
                for att in msg.get("attachments", []):
                    if att.get("text"):
                        parts.append(att["text"]["content"])
                    elif att.get("query"):
                        q = att["query"]
                        if q.get("description"):
                            parts.append(q["description"])
                        if q.get("query"):
                            sql_query = q["query"]
                    elif att.get("suggested_questions"):
                        suggested_questions = att["suggested_questions"].get("questions", [])
                answer = "\n\n".join(parts) or "Query completed with no text response."
                answer = _reformat_as_markdown(w, answer)
                return GenieResponse(
                    answer=answer,
                    sql=sql_query,
                    suggested_questions=suggested_questions,
                    conversation_id=conv_id,
                )

            if status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
                error_detail = msg.get("error", {})
                error_msg = ""
                if isinstance(error_detail, dict):
                    error_msg = error_detail.get("message", "") or error_detail.get("detail", "")
                elif isinstance(error_detail, str):
                    error_msg = error_detail
                logger.error("Genie query %s: %s | full msg: %s", status, error_msg, msg)
                answer = f"Genie query {status.lower()}."
                if error_msg:
                    answer += f" Error: {error_msg}"
                return GenieResponse(answer=answer, conversation_id=conv_id)

        return GenieResponse(answer="Genie query timed out after 90 seconds.", conversation_id=conv_id)

    except Exception as e:
        logger.error("Genie error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Job descriptions (static, table has no description column) ─────────────────
JOB_DESCRIPTIONS = {
    "JR001": """\
## Director of Human Resources

Jackson & Jackson is seeking an experienced **Director of Human Resources** to lead people operations across our New Brunswick headquarters. Reporting directly to the Chief People Officer, this role partners with senior leadership to drive talent strategy, employee engagement, and organizational effectiveness across a pharma/life sciences environment.

### Key Responsibilities
- Lead and develop a team of HR Business Partners, Talent Acquisition specialists, and HR Operations staff
- Own full-cycle HR for assigned business units: workforce planning, performance management, compensation reviews, and succession planning
- Drive DEI initiatives and embed inclusive practices across the employee lifecycle
- Serve as a trusted advisor to VP- and Director-level leaders on complex people matters
- Partner with Legal and Compliance on employee relations, investigations, and policy governance
- Champion HR technology adoption (Workday, ServiceNow HR) and data-driven decision making

### What We're Looking For
- 10+ years of progressive HR experience, with at least 4 years in a leadership role
- Deep expertise in employment law, HR policy, and organizational design
- SPHR or SHRM-SCP certification strongly preferred
- Prior experience in pharma, biotech, or a regulated industry is a significant plus
- Strong executive presence and ability to influence without authority
""",
    "JR002": """\
## VP of Talent Acquisition

Jackson & Jackson is looking for a strategic **VP of Talent Acquisition** to transform how we attract and hire world-class talent. This is a high-impact leadership role responsible for modernizing our TA function — from executive recruiting to campus programs — at a critical moment of growth for the organization.

### Key Responsibilities
- Build and lead a high-performing TA team of 15+ across sourcing, recruiting, and coordination
- Define and execute a multi-year talent acquisition strategy aligned to J&J's workforce plan
- Establish employer branding, candidate experience, and market intelligence capabilities
- Own recruiting metrics: time-to-fill, quality of hire, offer acceptance rate, and diversity pipeline
- Lead executive and board-level search partnerships
- Evaluate and implement ATS, CRM, and AI-assisted sourcing tools

### What We're Looking For
- 12+ years in talent acquisition, with 5+ years leading TA at scale (500+ hires/year)
- Track record modernizing TA operations in a Fortune 500 or large enterprise environment
- SHRM-SCP or equivalent certification preferred
- Experience with data-driven recruiting and predictive hiring tools
- Prior pharma, medical device, or life sciences TA leadership is a plus
""",
    "JR003": """\
## Director of Compensation & Benefits

Jackson & Jackson is hiring a **Director of Compensation & Benefits** to design, deliver, and evolve our total rewards strategy. This role partners closely with Finance, Legal, and HR Leadership to ensure our compensation programs attract top talent, retain key contributors, and remain compliant across all geographies.

### Key Responsibilities
- Lead the design and administration of base pay, short-term incentive, and equity programs
- Own annual compensation review cycles, market benchmarking (Radford, Mercer), and job architecture
- Manage health, welfare, retirement, and leave benefit programs for 5,000+ employees
- Ensure compliance with FLSA, ACA, ERISA, and multi-state pay equity laws
- Partner with Talent Acquisition on offer strategy and competitive pay positioning
- Lead a team of Compensation Analysts and Benefits Specialists

### What We're Looking For
- 10+ years in compensation and/or benefits, with 4+ years in a leadership role
- CCP (Certified Compensation Professional) strongly preferred
- Deep knowledge of executive compensation, equity plan administration (RSU/ESOP), and global benefits
- Experience with Workday Compensation or equivalent HRIS
- Strong analytical skills and advanced Excel/data modeling proficiency
""",
    "JR004": """\
## Chief People Officer

Jackson & Jackson is conducting a confidential search for an exceptional **Chief People Officer** to serve as a key member of the Executive Committee. Reporting to the CEO, the CPO will shape the company's human capital strategy, culture, and organizational capability as J&J enters its next phase of growth.

### Key Responsibilities
- Set the vision and strategy for all People functions: TA, Total Rewards, L&D, HR Ops, DEI, and Employee Relations
- Partner with the Board and CEO on succession planning, executive compensation, and org design
- Build a high-performance, inclusive culture that attracts and retains the best talent in pharma/life sciences
- Lead enterprise-wide transformation initiatives including digital HR, workforce planning, and change management
- Represent J&J externally as an employer brand ambassador and thought leader
- Manage an HR organization of 80+ across multiple sites and geographies

### What We're Looking For
- 20+ years of progressive HR experience, including 5+ years as a CHRO or CPO in a large enterprise
- Demonstrated success driving culture transformation and large-scale organizational change
- SPHR and/or SHRM-SCP; advanced degree (MBA, PhD in Org. Psychology) strongly preferred
- Executive presence and board-level credibility
- Prior pharma, biotech, or Fortune 500 life sciences experience highly valued
""",
}


# ── Jobs ───────────────────────────────────────────────────────────────────────
@app.get("/api/jobs")
async def get_jobs():
    """Fetch job requirements from the Lakebase synced table (low latency)."""
    try:
        jobs = execute_query("""
            SELECT job_id, title, department, location,
                   min_years_experience, required_education,
                   preferred_certifications, required_skills, preferred_skills,
                   salary_min, salary_max, team_size, reporting_to, description
            FROM job_requirements
            ORDER BY job_id
        """)
        # Overlay curated descriptions where the table has none.
        for j in jobs:
            if not j.get("description"):
                j["description"] = JOB_DESCRIPTIONS.get(j.get("job_id"), "")
        return {"jobs": jobs}
    except Exception as e:
        logger.error("Jobs error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Resume ─────────────────────────────────────────────────────────────────────
@app.get("/api/resume-pdf/{candidate_id}")
async def get_resume_pdf(candidate_id: str):
    """Stream the resume PDF for a candidate from the UC volume."""
    try:
        match = re.match(r'^C(\d+)$', candidate_id, re.IGNORECASE)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid candidate ID")
        num = int(match.group(1))
        filename = f"resume_{num:02d}.pdf"
        volume_path = f"/Volumes/{TARGET_CATALOG}/{TARGET_SCHEMA}/raw_data/resumes/{filename}"
        w = get_client()
        dl = w.files.download(volume_path)
        pdf_bytes = dl.contents.read()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Resume PDF error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Frontend ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/{path:path}")
async def catch_all(path: str):
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    return FileResponse("index.html")
