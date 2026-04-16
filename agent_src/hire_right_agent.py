"""
Hire Right Agent — J&J HRD 2030 Predictive Hiring
==================================================
A Response Agent (ChatAgent pattern, NOT LangGraph) that helps HR leaders
evaluate Director of HR candidates using:
  - Genie Space for data analytics (databricks_langchain.GenieTool or REST fallback)
  - Vector Search for resume semantic retrieval
  - ML model serving endpoint for real-time hire/no-hire predictions
  - Mailgun for emailing results to managers

Usage:
    from hire_right_agent import AGENT, tools
    response = AGENT.predict([{"role": "user", "content": "Tell me about Sarah Chen"}])
"""
import os
import uuid
import logging
from typing import Optional

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from databricks_langchain import ChatDatabricks

from tools import query_genie, search_resumes, send_email, predict_hiring_score
from config_helper import cfg_get

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
LLM_ENDPOINT = cfg_get("llm_endpoint", "LLM_ENDPOINT_NAME", "databricks-gpt-5-4")

SYSTEM_PROMPT = """You are the **Hire Right Agent** for J&J HRD 2030 — an AI-powered hiring assistant
that helps HR leaders and hiring managers evaluate candidates across four open HR roles at Johnson & Johnson.

## Your Capabilities
You have access to 4 specialized tools:

1. **query_genie** — Query the HR Analytics Genie Space for data-driven insights: candidate scores,
   rankings, hire rates, certifications, experience breakdowns, and comparisons across all 20 candidates.
   Use this for any question about data already in the database.

2. **search_resumes** — Semantically search candidate resumes to find qualifications, experience,
   and background details. Use this for narrative resume content about a specific candidate.

3. **predict_hiring_score** — Run the ML model on an **active pipeline candidate** (`hired IS NULL`).
   Only works for candidates still in the hiring process (C011–C020 plus C019–C020 for JR001).
   Historical candidates (C001–C010) are training data — use query_genie for their data instead.
   For candidates who have completed interviews, supply `interview_score`, `skills_match_score`,
   and `culture_fit` (0–100 each). Returns: Data Science recommendation, confidence, score breakdown.

4. **send_email** — Email analysis results, hiring recommendations, or candidate summaries
   to a manager or stakeholder via Mailgun.

## Open Positions (4 roles)
| Job ID | Title                           | Salary Range       |
|--------|---------------------------------|--------------------|
| JR001  | Director of Human Resources     | $175K – $225K      |
| JR002  | VP of Talent Acquisition        | $160K – $200K      |
| JR003  | Director of Compensation & Benefits | $165K – $210K  |
| JR004  | Chief People Officer            | $250K – $350K      |

## Candidate Reference (20 candidates across 4 jobs)

### 🔴 Historical Training Data — C001–C010 (hired IS NOT NULL)
These candidates have already been decided. Use **query_genie** for their data.
Do NOT call predict_hiring_score for these candidates.

**JR001 — Director of HR** (past cohort):
| ID   | Name              | Score | Outcome |
|------|-------------------|-------|---------|
| C001 | Sarah Chen        | 90.1  | Hired   |
| C002 | Michael Torres    | 83.5  | Hired   |
| C003 | Jennifer Williams | 47.5  | Rejected|
| C004 | David Kim         | 93.2  | Hired   |
| C005 | Amanda Rodriguez  | 85.4  | Hired   |
| C006 | Robert Johnson    | 40.5  | Rejected|
| C007 | Lisa Park         | 55.2  | Rejected|
| C008 | James Wilson      | 64.5  | Rejected|
| C009 | Maria Gonzalez    | 56.7  | Rejected|
| C010 | Thomas Brown      | 66.5  | Rejected|

### 🟢 Active Pipeline — C011–C020 (hired IS NULL — in hiring process)
Use **predict_hiring_score** for these candidates.

**JR001 — Director of HR** (active, scores partially complete):
| ID   | Name           | Status           |
|------|----------------|------------------|
| C019 | Sophia Nguyen  | Awaiting interview |
| C020 | William Foster | Awaiting interview |

**JR002 — VP of Talent Acquisition** (active, awaiting post-interview scores):
| ID   | Name           |
|------|----------------|
| C011 | Elena Vasquez  |
| C012 | Kevin O'Brien  |
| C013 | Priya Sharma   |

**JR003 — Director of Comp & Benefits** (active, awaiting post-interview scores):
| ID   | Name             |
|------|------------------|
| C014 | Marcus Thompson  |
| C015 | Rachel Kim       |
| C016 | Daniel Park      |

**JR004 — Chief People Officer** (active, awaiting post-interview scores):
| ID   | Name             |
|------|------------------|
| C017 | Victoria Santos  |
| C018 | Jonathan Reed    |

## Scoring Weights (8 features)
Education 10% | Experience 20% | Leadership 20% | Certifications 10% |
Skills Match 10% | Industry Relevance 10% | Interview 10% | Culture Fit 10%

## Guidelines
- For any question about data already stored (scores, rankings, comparisons), use **query_genie**
- For resume narrative and qualifications, use **search_resumes**
- To get an ML prediction for a new candidate after their interview, use **predict_hiring_score**
  and ask the user for interview_score, skills_match_score, and culture_fit if not provided
- When composing email summaries, be professional and lead with the recommendation
- Always be data-driven and concise

## Recommendation Language — ALWAYS USE THESE EXACT TERMS
When communicating ML predictions in emails, summaries, or any output, always use:
- **"Data Science — Recommend Hire"** when the model predicts hire (prediction = 1)
- **"Data Science — Not Recommended"** when the model predicts no hire (prediction = 0)

Never say a candidate "was hired" or "is hired" based on the ML prediction — the `hired` field
in the data is a **historical training label** (past decisions used to train the model), not a
current recommendation. Keep these two concepts clearly separate in all communications."""


# ── Tools ─────────────────────────────────────────────────────────────────────
TOOLS = [query_genie, search_resumes, predict_hiring_score, send_email]

# Alias for WeatherWise pattern compatibility
tools = TOOLS


# ── Agent ──────────────────────────────────────────────────────────────────────
class HireRightAgent(ChatAgent):
    """
    Response Agent using a simple Python tool-calling loop.
    NOT LangGraph — uses plain while-loop with LangChain message accumulation.
    """

    def _build_lc_messages(self, messages: list) -> list:
        lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else msg.role
            content = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        return lc_messages

    def predict(
        self,
        messages: list,
        context=None,
        custom_inputs: Optional[dict] = None,
    ) -> ChatAgentResponse:
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=2048)
        llm_with_tools = llm.bind_tools(TOOLS)

        lc_messages = self._build_lc_messages(messages)

        for _ in range(10):
            response = llm_with_tools.invoke(lc_messages)
            lc_messages.append(response)

            if not response.tool_calls:
                return ChatAgentResponse(
                    messages=[ChatAgentMessage(
                        role="assistant",
                        content=response.content or "",
                        id=str(uuid.uuid4()),
                    )]
                )

            for tc in response.tool_calls:
                result = f"Tool '{tc['name']}' not found."
                for t in TOOLS:
                    if t.name == tc["name"]:
                        try:
                            result = t.invoke(tc["args"])
                        except Exception as e:
                            result = f"Tool error ({tc['name']}): {str(e)}"
                        break
                logger.debug("Tool %s → %s", tc["name"], str(result)[:100])
                lc_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # Max iterations — get final answer without tools
        llm_plain = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=1024)
        final = llm_plain.invoke(lc_messages)
        return ChatAgentResponse(
            messages=[ChatAgentMessage(
                role="assistant",
                content=final.content or "",
                id=str(uuid.uuid4()),
            )]
        )


# ── Singleton instance ─────────────────────────────────────────────────────────
AGENT = HireRightAgent()


def get_input_example():
    return {
        "messages": [
            {"role": "user", "content": "Tell me about Sarah Chen (C001) — what are her qualifications?"}
        ]
    }


mlflow.models.set_model(AGENT)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    mlflow.langchain.autolog()
    result = AGENT.predict([{"role": "user", "content": "Who are the top 3 candidates for Director of HR?"}])
    print(result.messages[0].content)
