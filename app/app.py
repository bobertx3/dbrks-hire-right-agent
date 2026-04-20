"""
Hire Right Agent — Databricks App Backend (v3)
Jackson and Jackson HR Digital
FastAPI: proxies chat to agent endpoint, Genie Conversation API with multi-turn support.
"""
import os
import re
import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hire Right Agent", version="3.0.0")

# ── Config ─────────────────────────────────────────────────────────────────────
AGENT_ENDPOINT = os.getenv("DATABRICKS_AGENT_ENDPOINT", "hire-right-agent-endpoint")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f13a0f6a081fabbea933cfb0db1d01")
TARGET_CATALOG = os.getenv("TARGET_CATALOG", "bx4")
TARGET_SCHEMA  = os.getenv("TARGET_SCHEMA", "hrd_2030")
WAREHOUSE_ID   = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
LLM_ENDPOINT   = os.getenv("LLM_ENDPOINT", "databricks-gpt-5-4")


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


# ── Candidate Data ─────────────────────────────────────────────────────────────
CANDIDATES = [
    # JR001 — Director of Human Resources (historical cohort C001–C010)
    {"id":"C001","name":"Sarah Chen","title":"VP of Human Resources","company":"Novartis Pharmaceuticals","job_id":"JR001","job_title":"Director of Human Resources","location":"Chicago, IL","education":"MBA, Northwestern","certifications":"SPHR","total_score":89.3,"stage":"Hired","hired":True,"scores":{"education":85,"experience":90,"leadership":85,"certifications":95,"skills_match":92,"industry":95,"interview":88,"culture_fit":88}},
    {"id":"C002","name":"Michael Torres","title":"Director of People & Culture","company":"Boston Scientific","job_id":"JR001","job_title":"Director of Human Resources","location":"Boston, MA","education":"MA in HR Mgmt","certifications":"SHRM-SCP","total_score":83.7,"stage":"Hired","hired":True,"scores":{"education":75,"experience":85,"leadership":80,"certifications":90,"skills_match":85,"industry":90,"interview":82,"culture_fit":85}},
    {"id":"C003","name":"Jennifer Williams","title":"HR Business Partner","company":"Target Corporation","job_id":"JR001","job_title":"Director of Human Resources","location":"Atlanta, GA","education":"BA in Business","certifications":"None","total_score":45.3,"stage":"Rejected","hired":False,"scores":{"education":60,"experience":55,"leadership":40,"certifications":0,"skills_match":50,"industry":40,"interview":58,"culture_fit":55}},
    {"id":"C004","name":"David Kim","title":"Chief People Officer","company":"Merck & Co","job_id":"JR001","job_title":"Director of Human Resources","location":"Philadelphia, PA","education":"MBA, Wharton","certifications":"SPHR, PHR","total_score":92.6,"stage":"Hired","hired":True,"scores":{"education":85,"experience":95,"leadership":92,"certifications":95,"skills_match":95,"industry":95,"interview":90,"culture_fit":92}},
    {"id":"C005","name":"Amanda Rodriguez","title":"Head of Human Resources","company":"Gilead Sciences","job_id":"JR001","job_title":"Director of Human Resources","location":"San Diego, CA","education":"PhD Org. Psychology","certifications":"SHRM-SCP","total_score":68.7,"stage":"Hired","hired":True,"scores":{"education":90,"experience":82,"leadership":50,"certifications":90,"skills_match":50,"industry":80,"interview":58,"culture_fit":55}},
    {"id":"C006","name":"Robert Johnson","title":"HR Generalist","company":"Amazon Web Services","job_id":"JR001","job_title":"Director of Human Resources","location":"Seattle, WA","education":"BA in Psychology","certifications":"SHRM-CP","total_score":42.0,"stage":"Rejected","hired":False,"scores":{"education":60,"experience":35,"leadership":25,"certifications":65,"skills_match":40,"industry":45,"interview":45,"culture_fit":45}},
    {"id":"C007","name":"Lisa Park","title":"HR Manager","company":"Ford Motor Company","job_id":"JR001","job_title":"Director of Human Resources","location":"Detroit, MI","education":"MA in Business Admin","certifications":"SHRM-CP","total_score":55.7,"stage":"Rejected","hired":False,"scores":{"education":75,"experience":50,"leadership":45,"certifications":65,"skills_match":55,"industry":50,"interview":60,"culture_fit":62}},
    {"id":"C008","name":"James Wilson","title":"Senior HR Business Partner","company":"Goldman Sachs","job_id":"JR001","job_title":"Director of Human Resources","location":"New York, NY","education":"MBA","certifications":"PHR","total_score":65.0,"stage":"Rejected","hired":False,"scores":{"education":85,"experience":65,"leadership":55,"certifications":75,"skills_match":60,"industry":55,"interview":65,"culture_fit":70}},
    {"id":"C009","name":"Maria Gonzalez","title":"HR Supervisor","company":"Publix Super Markets","job_id":"JR001","job_title":"Director of Human Resources","location":"Miami, FL","education":"BA in Human Resources","certifications":"SHRM-CP","total_score":58.5,"stage":"Rejected","hired":False,"scores":{"education":60,"experience":70,"leadership":45,"certifications":65,"skills_match":58,"industry":45,"interview":62,"culture_fit":65}},
    {"id":"C010","name":"Thomas Brown","title":"HR Consultant","company":"Deloitte Consulting","job_id":"JR001","job_title":"Director of Human Resources","location":"Chicago, IL","education":"MA in Human Resources","certifications":"SPHR","total_score":67.5,"stage":"Rejected","hired":False,"scores":{"education":75,"experience":75,"leadership":40,"certifications":95,"skills_match":70,"industry":60,"interview":70,"culture_fit":75}},
    # JR001 — Active Pipeline
    {"id":"C019","name":"Sophia Nguyen","title":"HR Director","company":"Sanofi","job_id":"JR001","job_title":"Director of Human Resources","location":"Thousand Oaks, CA","education":"MBA","certifications":"SHRM-SCP","total_score":82.8,"stage":"Awaiting Interview","hired":None,"scores":{"education":85,"experience":84,"leadership":76,"certifications":90,"skills_match":78,"industry":95,"interview":80,"culture_fit":80}},
    {"id":"C020","name":"William Foster","title":"Senior HR Business Partner","company":"Cigna","job_id":"JR001","job_title":"Director of Human Resources","location":"Indianapolis, IN","education":"MA in HR","certifications":"SHRM-CP","total_score":84.3,"stage":"Awaiting Interview","hired":None,"scores":{"education":100,"experience":98,"leadership":40,"certifications":95,"skills_match":95,"industry":95,"interview":92,"culture_fit":90}},
    # JR002 — VP of Talent Acquisition (active pipeline)
    {"id":"C011","name":"Elena Vasquez","title":"Director of Talent Acquisition","company":"AbbVie","job_id":"JR002","job_title":"VP of Talent Acquisition","location":"Seattle, WA","education":"MBA, University of Washington","certifications":"SHRM-SCP","total_score":None,"stage":"Interview","hired":None,"scores":{"education":90,"experience":88,"leadership":82,"certifications":90,"skills_match":None,"industry":95,"interview":None,"culture_fit":None}},
    {"id":"C012","name":"Kevin O'Brien","title":"Global Talent Acquisition Lead","company":"Pfizer","job_id":"JR002","job_title":"VP of Talent Acquisition","location":"San Francisco, CA","education":"BA in Communications","certifications":"SHRM-CP","total_score":None,"stage":"Interview","hired":None,"scores":{"education":85,"experience":82,"leadership":72,"certifications":75,"skills_match":None,"industry":95,"interview":None,"culture_fit":None}},
    {"id":"C013","name":"Priya Sharma","title":"Talent Acquisition Manager","company":"Accenture","job_id":"JR002","job_title":"VP of Talent Acquisition","location":"New York, NY","education":"MBA, Columbia","certifications":"SHRM-SCP","total_score":None,"stage":"Interview","hired":None,"scores":{"education":90,"experience":78,"leadership":68,"certifications":90,"skills_match":None,"industry":55,"interview":None,"culture_fit":None}},
    # JR003 — Director of Compensation & Benefits (active pipeline)
    {"id":"C014","name":"Marcus Thompson","title":"Senior Director of Compensation","company":"Eli Lilly","job_id":"JR003","job_title":"Director of Compensation & Benefits","location":"Milwaukee, WI","education":"MS in Finance","certifications":"CCP","total_score":None,"stage":"Screening","hired":None,"scores":{"education":90,"experience":92,"leadership":86,"certifications":80,"skills_match":None,"industry":95,"interview":None,"culture_fit":None}},
    {"id":"C015","name":"Rachel Kim","title":"Director of Total Rewards","company":"UnitedHealth Group","job_id":"JR003","job_title":"Director of Compensation & Benefits","location":"New York, NY","education":"MBA, NYU Stern","certifications":"SHRM-SCP, CCP","total_score":None,"stage":"Screening","hired":None,"scores":{"education":85,"experience":84,"leadership":76,"certifications":90,"skills_match":None,"industry":88,"interview":None,"culture_fit":None}},
    {"id":"C016","name":"Daniel Park","title":"Compensation & Benefits Manager","company":"Apple","job_id":"JR003","job_title":"Director of Compensation & Benefits","location":"Minneapolis, MN","education":"BA in Accounting","certifications":"PHR","total_score":None,"stage":"Screening","hired":None,"scores":{"education":90,"experience":75,"leadership":62,"certifications":70,"skills_match":None,"industry":40,"interview":None,"culture_fit":None}},
    # JR004 — Chief People Officer (active pipeline)
    {"id":"C017","name":"Victoria Santos","title":"EVP of Human Resources","company":"Bristol-Myers Squibb","job_id":"JR004","job_title":"Chief People Officer","location":"Cambridge, MA","education":"PhD, Org. Leadership","certifications":"SPHR, SHRM-SCP","total_score":None,"stage":"Final Round","hired":None,"scores":{"education":100,"experience":100,"leadership":100,"certifications":100,"skills_match":None,"industry":100,"interview":None,"culture_fit":None}},
    {"id":"C018","name":"Jonathan Reed","title":"Chief Human Resources Officer","company":"Biogen","job_id":"JR004","job_title":"Chief People Officer","location":"New York, NY","education":"MBA, Harvard","certifications":"SPHR","total_score":None,"stage":"Final Round","hired":None,"scores":{"education":90,"experience":95,"leadership":94,"certifications":95,"skills_match":None,"industry":95,"interview":None,"culture_fit":None}},
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
    """Fetch job requirements with candidate counts from the UC table."""
    try:
        w = get_client()
        # Use try_extract_value pattern: fall back to NULL if description col absent
        # DESCRIBE TABLE returns data_array rows as lists [col_name, data_type, comment]
        desc_rows = w.api_client.do(
            "POST", "/api/2.0/sql/statements",
            body={"warehouse_id": WAREHOUSE_ID, "wait_timeout": "30s",
                  "statement": f"DESCRIBE TABLE {TARGET_CATALOG}.{TARGET_SCHEMA}.job_requirements"},
        ).get("result", {}).get("data_array", [])
        has_desc = any(r[0] == "description" for r in desc_rows if r)
        desc_expr = "j.description" if has_desc else "NULL AS description"
        sql = f"""
            SELECT j.job_id, j.title, j.department, j.location,
                   j.min_years_experience, j.required_education,
                   j.preferred_certifications, j.required_skills, j.preferred_skills,
                   j.salary_min, j.salary_max, j.team_size, j.reporting_to,
                   {desc_expr}
            FROM {TARGET_CATALOG}.{TARGET_SCHEMA}.job_requirements j
            ORDER BY j.job_id
        """
        result = w.api_client.do(
            "POST",
            "/api/2.0/sql/statements",
            body={"warehouse_id": WAREHOUSE_ID, "statement": sql, "wait_timeout": "30s"},
        )
        cols = [c["name"] for c in (result.get("manifest") or {}).get("schema", {}).get("columns", [])]
        rows = (result.get("result") or {}).get("data_array", [])
        jobs = [dict(zip(cols, row)) for row in rows]
        # Overlay static descriptions (table has no description column)
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
