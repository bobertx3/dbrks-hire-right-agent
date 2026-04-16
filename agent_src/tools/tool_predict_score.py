"""
Tool: predict_hiring_score
Looks up a candidate's feature scores from the Unity Catalog `candidates` table,
optionally overrides interview_score / skills_match_score / culture_fit with
values supplied at runtime (for new candidates who haven't been fully scored),
then calls the ML serving endpoint for a hire/no-hire prediction + probability.
"""
import os
import json
import requests
from langchain_core.tools import tool
from config_helper import cfg_get


FEATURE_COLS = [
    "education_score",
    "experience_score",
    "leadership_score",
    "certification_score",
    "skills_match_score",
    "industry_relevance_score",
    "interview_score",
    "culture_fit",
]


def _get_config():
    return {
        "endpoint": cfg_get("model_endpoint_name", "MODEL_ENDPOINT_NAME", "hr-predictive-hiring-endpoint"),
        "catalog":  cfg_get("target_catalog",       "TARGET_CATALOG",      "bx4"),
        "schema":   cfg_get("target_schema",        "TARGET_SCHEMA",       "hrd_2030"),
        "host":     os.getenv("DATABRICKS_HOST", "").rstrip("/"),
        "token":    os.getenv("DATABRICKS_TOKEN", ""),
    }


def _fetch_candidate(catalog: str, schema: str, candidate_id: str) -> dict | None:
    """Fetch an active candidate (hired IS NULL) from the Delta table via Spark SQL.
    Only returns candidates still in the hiring pipeline — not historical training data.
    Returns a dict of column→value, or None if not found / already decided."""
    try:
        # Import spark lazily — only available inside a Databricks notebook/job
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            return None
        rows = spark.sql(
            f"SELECT * FROM `{catalog}`.`{schema}`.candidates "
            f"WHERE candidate_id = '{candidate_id.upper()}' AND hired IS NULL LIMIT 1"
        ).collect()
        return rows[0].asDict() if rows else None
    except Exception:
        return None


def _call_endpoint(host: str, token: str, endpoint: str, feature_values: dict) -> tuple[int, float | None]:
    """POST to the model serving endpoint. Returns (prediction, probability)."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload = {"dataframe_records": [feature_values]}
    resp = requests.post(
        f"{host}/serving-endpoints/{endpoint}/invocations",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    raw = body.get("predictions", [None])[0]
    if isinstance(raw, dict):
        return int(raw.get("prediction", raw.get("0", 0))), raw.get("probability")
    return int(raw) if raw is not None else 0, None


@tool
def predict_hiring_score(
    candidate_id: str,
    interview_score: int = None,
    skills_match_score: int = None,
    culture_fit: int = None,
) -> str:
    """Predict whether an ACTIVE pipeline candidate should be hired using the trained ML model.
    Only works for candidates where hired IS NULL (still in the hiring process).
    Historical candidates (C001–C010, already hired or rejected) are not scored here —
    use query_genie to look up their historical data.

    Active candidates: C011–C020 (JR002–JR004) plus C019–C020 (JR001).
    For candidates who have completed their interview, supply interview_score,
    skills_match_score, and culture_fit. For others these will be fetched from the database.

    Args:
        candidate_id: Active candidate ID, e.g. "C011" or "C019"
        interview_score: Interview rating 0–100 (if collected post-interview)
        skills_match_score: Skills match rating 0–100 (if collected post-interview)
        culture_fit: Culture fit rating 0–100 (if collected post-interview)

    Returns a Data Science recommendation, confidence score, and feature breakdown.
    """
    cfg = _get_config()
    cid = candidate_id.upper().strip()

    # ── Fetch from Delta table ──────────────────────────────────────────────────
    row = _fetch_candidate(cfg["catalog"], cfg["schema"], cid)

    if row is None:
        return (
            f"Candidate '{cid}' is not an active hiring candidate (hired IS NULL). "
            f"They may have already been hired or rejected (historical training data), "
            f"or the ID is invalid. Active pipeline candidates are C011–C020 plus C019–C020 for JR001. "
            f"Use query_genie to look up historical data on past candidates."
        )

    name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
    job_id = row.get("job_id", "unknown")

    # ── Apply overrides for new candidates ────────────────────────────────────
    if interview_score    is not None: row["interview_score"]    = interview_score
    if skills_match_score is not None: row["skills_match_score"] = skills_match_score
    if culture_fit        is not None: row["culture_fit"]        = culture_fit

    # ── Check for missing features ─────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if row.get(c) is None]
    if missing:
        return (
            f"Cannot score {name} ({cid}) — missing features: {', '.join(missing)}.\n"
            f"For new candidates, please supply: interview_score, skills_match_score, culture_fit."
        )

    feature_values = {col: int(row[col]) for col in FEATURE_COLS}

    # ── Call ML endpoint ───────────────────────────────────────────────────────
    try:
        prediction, probability = _call_endpoint(
            cfg["host"], cfg["token"], cfg["endpoint"], feature_values
        )
        source = "ML endpoint"
    except Exception as e:
        # Graceful fallback: simple weighted sum threshold
        total = (
            feature_values["education_score"]          * 0.10 +
            feature_values["experience_score"]         * 0.20 +
            feature_values["leadership_score"]         * 0.20 +
            feature_values["certification_score"]      * 0.10 +
            feature_values["skills_match_score"]       * 0.10 +
            feature_values["industry_relevance_score"] * 0.10 +
            feature_values["interview_score"]          * 0.10 +
            feature_values["culture_fit"]              * 0.10
        )
        prediction  = 1 if total >= 75 else 0
        probability = None
        source      = f"fallback (endpoint error: {str(e)[:60]})"

    # ── Format result ──────────────────────────────────────────────────────────
    rec      = "Data Science — Recommend Hire" if prediction == 1 else "Data Science — Not Recommended"
    prob_str = f" (confidence: {float(probability):.0%})" if probability is not None else ""

    hired_label = row.get("hired")
    if hired_label is not None:
        actual = f"Historical training label: {'Hired ✅' if int(hired_label) == 1 else 'Not Hired ❌'} (past outcome used to train the model)"
    else:
        actual = "Historical training label: N/A (new candidate — no prior outcome)"

    fv = feature_values
    total_score = round(
        fv["education_score"]          * 0.10 +
        fv["experience_score"]         * 0.20 +
        fv["leadership_score"]         * 0.20 +
        fv["certification_score"]      * 0.10 +
        fv["skills_match_score"]       * 0.10 +
        fv["industry_relevance_score"] * 0.10 +
        fv["interview_score"]          * 0.10 +
        fv["culture_fit"]              * 0.10,
        1,
    )

    return (
        f"**ML Prediction for {name} ({cid}, {job_id}): {rec}{prob_str}**\n\n"
        f"Composite Score: **{total_score}/100**\n\n"
        f"Score Breakdown:\n"
        f"  • Education:           {fv['education_score']:>3}/100\n"
        f"  • Experience:          {fv['experience_score']:>3}/100\n"
        f"  • Leadership:          {fv['leadership_score']:>3}/100\n"
        f"  • Certifications:      {fv['certification_score']:>3}/100\n"
        f"  • Skills Match:        {fv['skills_match_score']:>3}/100\n"
        f"  • Industry Relevance:  {fv['industry_relevance_score']:>3}/100\n"
        f"  • Interview:           {fv['interview_score']:>3}/100\n"
        f"  • Culture Fit:         {fv['culture_fit']:>3}/100\n\n"
        f"{actual}\n"
        f"Prediction source: {source}"
    )
