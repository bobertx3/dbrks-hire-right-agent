"""
Hire Right Agent — Databricks App Backend
FastAPI server that proxies chat to the agent serving endpoint.
"""
import os
import logging
import asyncio
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hire Right Agent", version="1.0.0")

# ── Config ───────────────────────────────────────────────────────────────────
AGENT_ENDPOINT = os.getenv("DATABRICKS_AGENT_ENDPOINT", "hire-right-agent-endpoint")
WAREHOUSE_ID   = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
TARGET_CATALOG = os.getenv("TARGET_CATALOG", "bx4")
TARGET_SCHEMA  = os.getenv("TARGET_SCHEMA", "hrd_2030")


def get_client() -> WorkspaceClient:
    return WorkspaceClient(config=Config())


# ── Request / Response Models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []


class ChatResponse(BaseModel):
    reply: str
    conversation_history: list


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "healthy", "agent_endpoint": AGENT_ENDPOINT}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the Hire Right agent and return the response."""
    try:
        w = get_client()

        # Build message history
        history = list(request.conversation_history)
        history.append({"role": "user", "content": request.message})

        payload = {"messages": history}

        # Query the agent serving endpoint
        response = w.serving_endpoints.query(name=AGENT_ENDPOINT, request=payload)

        # Extract reply
        if response.predictions:
            pred = response.predictions[0]
            if isinstance(pred, dict):
                messages = pred.get("messages", [])
                if messages:
                    reply = messages[-1].get("content", str(pred))
                else:
                    reply = pred.get("content", str(pred))
            else:
                reply = str(pred)
        else:
            reply = "No response from the agent."

        # Append assistant reply to history
        history.append({"role": "assistant", "content": reply})

        return ChatResponse(reply=reply, conversation_history=history)

    except Exception as e:
        logger.error("Chat error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/candidates")
async def get_candidates():
    """Return the list of candidates with their scores for the dashboard."""
    candidates = [
        {"id": "C001", "name": "Sarah Chen",        "title": "VP of HR",              "company": "Novartis",      "total_score": 90, "hired": True,  "education": "MBA",      "certifications": "SPHR",      "location": "Chicago IL"},
        {"id": "C002", "name": "Michael Torres",    "title": "Director, People & Culture", "company": "Boston Scientific", "total_score": 84, "hired": True,  "education": "MA in HR", "certifications": "SHRM-SCP", "location": "Boston MA"},
        {"id": "C003", "name": "Jennifer Williams", "title": "HR Business Partner",   "company": "Target",        "total_score": 42, "hired": False, "education": "BA",       "certifications": "None",      "location": "Atlanta GA"},
        {"id": "C004", "name": "David Kim",         "title": "Chief People Officer",  "company": "Merck",         "total_score": 93, "hired": True,  "education": "MBA",      "certifications": "SPHR, PHR", "location": "Philadelphia PA"},
        {"id": "C005", "name": "Amanda Rodriguez",  "title": "Head of HR",            "company": "Gilead",        "total_score": 85, "hired": True,  "education": "PhD",      "certifications": "SHRM-SCP",  "location": "San Diego CA"},
        {"id": "C006", "name": "Robert Johnson",    "title": "HR Generalist",         "company": "Amazon",        "total_score": 42, "hired": False, "education": "BA",       "certifications": "SHRM-CP",   "location": "Seattle WA"},
        {"id": "C007", "name": "Lisa Park",         "title": "HR Manager",            "company": "Ford",          "total_score": 55, "hired": False, "education": "MA in BA", "certifications": "SHRM-CP",   "location": "Detroit MI"},
        {"id": "C008", "name": "James Wilson",      "title": "Senior HR BP",          "company": "Goldman Sachs", "total_score": 65, "hired": False, "education": "MBA",      "certifications": "PHR",       "location": "New York NY"},
        {"id": "C009", "name": "Maria Gonzalez",    "title": "HR Supervisor",         "company": "Publix",        "total_score": 58, "hired": False, "education": "BA in HR", "certifications": "SHRM-CP",   "location": "Miami FL"},
        {"id": "C010", "name": "Thomas Brown",      "title": "HR Consultant",         "company": "Deloitte",      "total_score": 67, "hired": False, "education": "MA in HR", "certifications": "SPHR",      "location": "Chicago IL"},
    ]
    return {"candidates": candidates}


# ── Serve Frontend ────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/{path:path}")
async def catch_all(path: str):
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    return FileResponse("index.html")
