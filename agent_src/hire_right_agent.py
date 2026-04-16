"""
Hire Right Agent — Jackson and Jackson HR Digital
==================================================
A ResponsesAgent (MLflow 3.0 standard, NOT LangGraph) that helps HR leaders
evaluate Director of HR candidates using:
  - Genie Space for data analytics (databricks_langchain.GenieTool or REST fallback)
  - Vector Search for resume semantic retrieval
  - ML model serving endpoint for real-time hire/no-hire predictions
  - Mailgun for emailing results to managers

Usage:
    from hire_right_agent import AGENT
    from mlflow.types.responses import ResponsesAgentRequest
    request = ResponsesAgentRequest(input=[{"role": "user", "content": "Tell me about Sarah Chen"}])
    response = AGENT.predict(request)
"""
import os
import json
import uuid
import logging
from typing import Generator

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from databricks_langchain import ChatDatabricks

from tools import query_genie, search_resumes, send_email, predict_hiring_score
from config_helper import cfg_get

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
LLM_ENDPOINT = cfg_get("llm_endpoint", "LLM_ENDPOINT_NAME", "databricks-gpt-5-4")

SYSTEM_PROMPT = """You are the **Hire Right Agent** for Jackson and Jackson HR Digital — an AI-powered hiring assistant
that helps HR leaders and hiring managers evaluate candidates across four open HR roles at Johnson & Johnson.

## Your Capabilities
You have access to 4 specialized tools:

1. **query_genie** — Query the HR Analytics Genie Space for data-driven insights: candidate scores,
   rankings, hire rates, certifications, experience breakdowns, and comparisons across all 20 candidates.
   Use this for any question about data already in the database.

2. **search_resumes** — Semantically search candidate resumes to find qualifications, experience,
   and background details. Use this for narrative resume content about a specific candidate.

3. **predict_hiring_score** — Run the ML model on any candidate to get a hire/no-hire prediction.
   Pass all known scores from conversation context directly as arguments — this avoids a database
   lookup. If scores are available in context (e.g. seeded from the profile view), always pass them.
   **CRITICAL: NEVER fabricate, estimate, or assume a value for any score marked as "pending".
   If interview_score or culture_fit (or any score) is "pending" in context, you MUST ask the user
   to provide those specific values before calling this tool. Do NOT invent placeholder numbers.**
   Returns: Data Science recommendation, confidence, score breakdown.

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

### C001–C010 — Historical Cohort (JR001, decisions already made)
Use **query_genie** for data questions. **predict_hiring_score** can also be called on these.

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

### C011–C020 — Active Pipeline (in hiring process, some scores pending)

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

## Guidelines
- For any question about data already stored (scores, rankings, comparisons), use **query_genie**
- For resume narrative and qualifications, use **search_resumes**
- To get an ML prediction, use **predict_hiring_score** — pass all known scores from context,
  and explicitly ask the user for any scores that are "pending". Never substitute a made-up number.
- If a user asks a hypothetical ("what if culture_fit were X"), only run it if you have a real
  baseline to compare against — do not invent the baseline
- Always be data-driven and concise

## CRITICAL — Never Compute Scores Manually
**NEVER calculate a hiring prediction or composite score yourself using any formula or weighted sum.**
The only valid way to produce a hiring recommendation is to call the **predict_hiring_score** tool.
The ML model is a trained classifier — it does not use a simple weighted average.
Any manual calculation you perform will be wrong and misleading. Always call the tool.

## Email Composition Guidelines
When sending a candidate summary email, use this structure:

**Opening line:** "For Job ID [JR00X] — [Role Title] — the following candidates are predicted to be the best fit:"

**Per candidate, write a 2–3 sentence narrative** (do NOT show raw total scores). Cover:
  1. Qualifications strength — reference their experience and/or skills match score in plain language
  2. Interview and culture fit — describe their interview performance and cultural alignment
  3. Data Science verdict — close with "Based on our Data Science prediction, [First Name] is a **Data Science — Recommend Hire**."

**Example narrative format:**
> "1. **David Kim (C004)** — David brings exceptional HR leadership experience with a strong skills match for this role. His interview performance was outstanding and he demonstrated excellent cultural alignment with the organization. Based on our Data Science prediction, David is a **Data Science — Recommend Hire**."

Do NOT include a "Total Score" line. Do NOT include the disclaimer about historical training labels in the email body.

## Recommendation Language — ALWAYS USE THESE EXACT TERMS
When communicating ML predictions in emails, summaries, or any output, always use:
- **"Data Science — Recommend Hire"** when the model predicts hire (prediction = 1)
- **"Data Science — Not Recommended"** when the model predicts no hire (prediction = 0)

Never say a candidate "was hired" or "is hired" based on the ML prediction — the `hired` field
in the data is a **historical training label** (past decisions used to train the model), not a
current recommendation. Keep these two concepts clearly separate in all communications."""


# ── Tools ─────────────────────────────────────────────────────────────────────
TOOLS = [query_genie, search_resumes, predict_hiring_score, send_email]

# Alias for compatibility
tools = TOOLS


# ── Agent ──────────────────────────────────────────────────────────────────────
class HireRightAgent(ResponsesAgent):
    """
    MLflow 3.0 ResponsesAgent using a simple Python tool-calling loop.
    NOT LangGraph — uses a plain while-loop with LangChain message accumulation.
    Tool calls are emitted as structured output items for full MLflow trace visibility.
    """

    def _build_lc_messages(self, input_messages) -> list:
        lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in input_messages:
            role    = msg.role    if hasattr(msg, "role")    else msg.get("role", "user")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")

            # Responses API delivers content as a list of typed params (e.g. ResponseInputTextParam).
            # Extract plain text from each item so LangChain gets a str, not a pydantic object.
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                    elif isinstance(item, dict):
                        parts.append(item.get("text", str(item)))
                    else:
                        parts.append(str(item))
                content = " ".join(parts)

            content = content or ""
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        return lc_messages

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        llm            = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=2048)
        llm_with_tools = llm.bind_tools(TOOLS)

        lc_messages  = self._build_lc_messages(request.input)
        output_items = []

        for _ in range(10):
            response = llm_with_tools.invoke(lc_messages)
            lc_messages.append(response)

            if not response.tool_calls:
                output_items.append(
                    self.create_text_output_item(
                        text=response.content or "",
                        id=str(uuid.uuid4()),
                    )
                )
                return ResponsesAgentResponse(output=output_items)

            for tc in response.tool_calls:
                # Emit the function call so MLflow traces it as a span
                output_items.append(
                    self.create_function_call_item(
                        id=str(uuid.uuid4()),
                        call_id=tc["id"],
                        name=tc["name"],
                        arguments=json.dumps(tc["args"]),
                    )
                )

                result = f"Tool '{tc['name']}' not found."
                for t in TOOLS:
                    if t.name == tc["name"]:
                        try:
                            result = t.invoke(tc["args"])
                        except Exception as e:
                            result = f"Tool error ({tc['name']}): {str(e)}"
                        break

                logger.debug("Tool %s → %s", tc["name"], str(result)[:100])

                # Emit the tool result so MLflow traces it as a span
                output_items.append(
                    self.create_function_call_output_item(
                        call_id=tc["id"],
                        output=str(result),
                    )
                )
                lc_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # Max iterations — get final answer without tools
        llm_plain = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=1024)
        final = llm_plain.invoke(lc_messages)
        output_items.append(
            self.create_text_output_item(
                text=final.content or "",
                id=str(uuid.uuid4()),
            )
        )
        return ResponsesAgentResponse(output=output_items)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        result = self.predict(request)
        for item in result.output:
            # MLflow serving calls .get() on item expecting a dict — convert from pydantic
            item_dict = item.model_dump() if hasattr(item, "model_dump") else item
            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item_dict)


# ── Singleton instance ─────────────────────────────────────────────────────────
mlflow.langchain.autolog()
AGENT = HireRightAgent()


def get_input_example():
    return {
        "input": [
            {"role": "user", "content": "Tell me about Sarah Chen (C001) — what are her qualifications?"}
        ]
    }


mlflow.models.set_model(AGENT)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Who are the top 3 candidates for Director of HR?"}]
    )
    result = AGENT.predict(request)
    text = next((item.text for item in reversed(result.output) if hasattr(item, "text")), "")
    print(text)
