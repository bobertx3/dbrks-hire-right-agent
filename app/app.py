"""
Hire Right Agent — Databricks App Backend (v3)
Jackson and Jackson HR Digital
FastAPI: proxies chat to agent endpoint, Genie Conversation API with multi-turn support.
"""
import os
import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hire Right Agent", version="3.0.0")

# ── Config ─────────────────────────────────────────────────────────────────────
AGENT_ENDPOINT = os.getenv("DATABRICKS_AGENT_ENDPOINT", "hire-right-agent-endpoint")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f1388a821b1e42a9d579bb45510abf")
TARGET_CATALOG = os.getenv("TARGET_CATALOG", "bx4")
TARGET_SCHEMA  = os.getenv("TARGET_SCHEMA", "hrd_2030")


def get_client() -> WorkspaceClient:
    return WorkspaceClient(config=Config())


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


# ── Candidate Data ─────────────────────────────────────────────────────────────
CANDIDATES = [
    # JR001 — Director of Human Resources (historical cohort C001–C010)
    {"id":"C001","name":"Sarah Chen","title":"VP of HR","company":"Novartis","job_id":"JR001","job_title":"Director of Human Resources","location":"Chicago, IL","education":"MBA, Northwestern","certifications":"SPHR","total_score":90.1,"stage":"Hired","hired":True,"scores":{"education":88,"experience":92,"leadership":90,"certifications":85,"skills_match":90,"industry":90,"interview":88,"culture_fit":92}},
    {"id":"C002","name":"Michael Torres","title":"Director, People & Culture","company":"Boston Scientific","job_id":"JR001","job_title":"Director of Human Resources","location":"Boston, MA","education":"MA in HR Mgmt","certifications":"SHRM-SCP","total_score":83.5,"stage":"Hired","hired":True,"scores":{"education":80,"experience":85,"leadership":84,"certifications":82,"skills_match":84,"industry":83,"interview":82,"culture_fit":85}},
    {"id":"C003","name":"Jennifer Williams","title":"HR Business Partner","company":"Target","job_id":"JR001","job_title":"Director of Human Resources","location":"Atlanta, GA","education":"BA in Business","certifications":"None","total_score":47.5,"stage":"Rejected","hired":False,"scores":{"education":60,"experience":45,"leadership":42,"certifications":30,"skills_match":50,"industry":48,"interview":45,"culture_fit":52}},
    {"id":"C004","name":"David Kim","title":"Chief People Officer","company":"Merck","job_id":"JR001","job_title":"Director of Human Resources","location":"Philadelphia, PA","education":"MBA, Wharton","certifications":"SPHR, PHR","total_score":93.2,"stage":"Hired","hired":True,"scores":{"education":95,"experience":96,"leadership":94,"certifications":92,"skills_match":92,"industry":93,"interview":94,"culture_fit":91}},
    {"id":"C005","name":"Amanda Rodriguez","title":"Head of HR","company":"Gilead Sciences","job_id":"JR001","job_title":"Director of Human Resources","location":"San Diego, CA","education":"PhD Org. Psychology","certifications":"SHRM-SCP","total_score":85.4,"stage":"Hired","hired":True,"scores":{"education":90,"experience":87,"leadership":86,"certifications":85,"skills_match":84,"industry":84,"interview":82,"culture_fit":88}},
    {"id":"C006","name":"Robert Johnson","title":"HR Generalist","company":"Amazon","job_id":"JR001","job_title":"Director of Human Resources","location":"Seattle, WA","education":"BA in Psychology","certifications":"SHRM-CP","total_score":40.5,"stage":"Rejected","hired":False,"scores":{"education":55,"experience":38,"leadership":35,"certifications":42,"skills_match":48,"industry":40,"interview":38,"culture_fit":45}},
    {"id":"C007","name":"Lisa Park","title":"HR Manager","company":"Ford Motor","job_id":"JR001","job_title":"Director of Human Resources","location":"Detroit, MI","education":"MA in Business Admin","certifications":"SHRM-CP","total_score":55.2,"stage":"Rejected","hired":False,"scores":{"education":65,"experience":52,"leadership":50,"certifications":55,"skills_match":58,"industry":55,"interview":56,"culture_fit":60}},
    {"id":"C008","name":"James Wilson","title":"Senior HR Business Partner","company":"Goldman Sachs","job_id":"JR001","job_title":"Director of Human Resources","location":"New York, NY","education":"MBA","certifications":"PHR","total_score":64.5,"stage":"Rejected","hired":False,"scores":{"education":75,"experience":65,"leadership":62,"certifications":65,"skills_match":64,"industry":60,"interview":62,"culture_fit":68}},
    {"id":"C009","name":"Maria Gonzalez","title":"HR Supervisor","company":"Publix","job_id":"JR001","job_title":"Director of Human Resources","location":"Miami, FL","education":"BA in Human Resources","certifications":"SHRM-CP","total_score":56.7,"stage":"Rejected","hired":False,"scores":{"education":62,"experience":54,"leadership":52,"certifications":58,"skills_match":55,"industry":55,"interview":56,"culture_fit":62}},
    {"id":"C010","name":"Thomas Brown","title":"HR Consultant","company":"Deloitte","job_id":"JR001","job_title":"Director of Human Resources","location":"Chicago, IL","education":"MA in Human Resources","certifications":"SPHR","total_score":66.5,"stage":"Rejected","hired":False,"scores":{"education":72,"experience":66,"leadership":64,"certifications":70,"skills_match":65,"industry":62,"interview":65,"culture_fit":70}},
    # JR001 — Active Pipeline
    {"id":"C019","name":"Sophia Nguyen","title":"HR Director","company":"Amgen","job_id":"JR001","job_title":"Director of Human Resources","location":"Thousand Oaks, CA","education":"MBA","certifications":"SHRM-SCP","total_score":None,"stage":"Awaiting Interview","hired":None,"scores":{"education":84,"experience":80,"leadership":78,"certifications":82,"skills_match":80,"industry":82,"interview":None,"culture_fit":None}},
    {"id":"C020","name":"William Foster","title":"HR Business Partner","company":"Eli Lilly","job_id":"JR001","job_title":"Director of Human Resources","location":"Indianapolis, IN","education":"MA in HR","certifications":"PHR","total_score":None,"stage":"Awaiting Interview","hired":None,"scores":{"education":75,"experience":72,"leadership":68,"certifications":72,"skills_match":74,"industry":70,"interview":None,"culture_fit":None}},
    # JR002 — VP of Talent Acquisition (active pipeline)
    {"id":"C011","name":"Elena Vasquez","title":"Director of Talent Acquisition","company":"Microsoft","job_id":"JR002","job_title":"VP of Talent Acquisition","location":"Seattle, WA","education":"MBA, University of Washington","certifications":"SHRM-SCP","total_score":None,"stage":"Interview","hired":None,"scores":{"education":88,"experience":91,"leadership":85,"certifications":87,"skills_match":89,"industry":88,"interview":None,"culture_fit":None}},
    {"id":"C012","name":"Kevin O'Brien","title":"VP of People Operations","company":"Salesforce","job_id":"JR002","job_title":"VP of Talent Acquisition","location":"San Francisco, CA","education":"BA in Communications","certifications":"PHR","total_score":None,"stage":"Interview","hired":None,"scores":{"education":72,"experience":84,"leadership":80,"certifications":68,"skills_match":78,"industry":82,"interview":None,"culture_fit":None}},
    {"id":"C013","name":"Priya Sharma","title":"Head of Global Talent","company":"Cognizant","job_id":"JR002","job_title":"VP of Talent Acquisition","location":"New York, NY","education":"MBA, Columbia","certifications":"SHRM-SCP, SPHR","total_score":None,"stage":"Interview","hired":None,"scores":{"education":92,"experience":88,"leadership":86,"certifications":90,"skills_match":85,"industry":87,"interview":None,"culture_fit":None}},
    # JR003 — Director of Compensation & Benefits (active pipeline)
    {"id":"C014","name":"Marcus Thompson","title":"Sr. Compensation Manager","company":"Johnson Controls","job_id":"JR003","job_title":"Director of Compensation & Benefits","location":"Milwaukee, WI","education":"MS in Finance","certifications":"CCP","total_score":None,"stage":"Screening","hired":None,"scores":{"education":85,"experience":82,"leadership":76,"certifications":88,"skills_match":84,"industry":80,"interview":None,"culture_fit":None}},
    {"id":"C015","name":"Rachel Kim","title":"Compensation & Benefits Lead","company":"Pfizer","job_id":"JR003","job_title":"Director of Compensation & Benefits","location":"New York, NY","education":"MBA, NYU Stern","certifications":"CCP, SHRM-CP","total_score":None,"stage":"Screening","hired":None,"scores":{"education":88,"experience":86,"leadership":80,"certifications":90,"skills_match":87,"industry":89,"interview":None,"culture_fit":None}},
    {"id":"C016","name":"Daniel Park","title":"Total Rewards Manager","company":"3M","job_id":"JR003","job_title":"Director of Compensation & Benefits","location":"Minneapolis, MN","education":"BA in Accounting","certifications":"PHR","total_score":None,"stage":"Screening","hired":None,"scores":{"education":70,"experience":72,"leadership":65,"certifications":72,"skills_match":74,"industry":70,"interview":None,"culture_fit":None}},
    # JR004 — Chief People Officer (active pipeline)
    {"id":"C017","name":"Victoria Santos","title":"Chief HR Officer","company":"Moderna","job_id":"JR004","job_title":"Chief People Officer","location":"Cambridge, MA","education":"PhD, Org. Leadership","certifications":"SPHR","total_score":None,"stage":"Final Round","hired":None,"scores":{"education":96,"experience":94,"leadership":95,"certifications":92,"skills_match":90,"industry":93,"interview":None,"culture_fit":None}},
    {"id":"C018","name":"Jonathan Reed","title":"Global Head of People","company":"Unilever","job_id":"JR004","job_title":"Chief People Officer","location":"New York, NY","education":"MBA, Harvard","certifications":"SHRM-SCP, SPHR","total_score":None,"stage":"Final Round","hired":None,"scores":{"education":93,"experience":92,"leadership":91,"certifications":88,"skills_match":88,"industry":90,"interview":None,"culture_fit":None}},
]


# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "healthy", "agent_endpoint": AGENT_ENDPOINT}


@app.get("/api/candidates")
async def get_candidates():
    return {"candidates": CANDIDATES}


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


# ── Frontend ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/{path:path}")
async def catch_all(path: str):
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    return FileResponse("index.html")
