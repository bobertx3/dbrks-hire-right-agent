"""
Tool: predict_hiring_score
Looks up a candidate's feature scores from the Unity Catalog `candidates` table,
optionally overrides interview_score / skills_match_score / culture_fit with
values supplied at runtime (for new candidates who haven't been fully scored),
then calls the ML serving endpoint for a hire/no-hire prediction + probability.
"""
import os
import mlflow
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
        "endpoint":     cfg_get("model_endpoint_name", "MODEL_ENDPOINT_NAME", "hr-predictive-hiring-endpoint"),
        "catalog":      cfg_get("target_catalog",       "TARGET_CATALOG",      "bx4"),
        "schema":       cfg_get("target_schema",        "TARGET_SCHEMA",       "hrd_2030"),
        "warehouse_id": cfg_get("warehouse_id",         "DATABRICKS_WAREHOUSE_ID", ""),
    }


def _get_ws():
    """Return a WorkspaceClient — handles credential passthrough in serving endpoints."""
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _fetch_via_sql_api(w, warehouse_id: str, catalog: str, schema: str, candidate_id: str) -> dict | None:
    """Fetch candidate via Statement Execution API using SDK (no token env var needed)."""
    sql = (
        f"SELECT * FROM `{catalog}`.`{schema}`.candidates "
        f"WHERE candidate_id = '{candidate_id}' LIMIT 1"
    )
    try:
        body = w.api_client.do(
            "POST",
            "/api/2.0/sql/statements",
            body={"statement": sql, "warehouse_id": warehouse_id, "wait_timeout": "30s", "on_wait_timeout": "CANCEL"},
        )
        if body.get("status", {}).get("state") != "SUCCEEDED":
            return None
        cols = [c["name"] for c in body["manifest"]["schema"]["columns"]]
        rows = body.get("result", {}).get("data_array", [])
        if not rows:
            return None
        row = dict(zip(cols, rows[0]))
        # Cast numeric score columns from string to int/float
        for col in FEATURE_COLS:
            if col in row and row[col] is not None:
                try:
                    row[col] = int(float(row[col]))
                except (ValueError, TypeError):
                    row[col] = None
        return row
    except Exception:
        return None


def _fetch_candidate(catalog: str, schema: str, candidate_id: str, cfg: dict) -> dict | None:
    """Fetch a candidate by ID. Tries Spark first, falls back to SDK Statement API."""
    # Try Spark (notebook/job context)
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is not None:
            rows = spark.sql(
                f"SELECT * FROM `{catalog}`.`{schema}`.candidates "
                f"WHERE candidate_id = '{candidate_id.upper()}' LIMIT 1"
            ).collect()
            return rows[0].asDict() if rows else None
    except Exception:
        pass

    # Fallback: SDK Statement Execution API (serving endpoint context)
    try:
        w = _get_ws()
        return _fetch_via_sql_api(w, cfg["warehouse_id"], catalog, schema, candidate_id)
    except Exception:
        return None


def _call_endpoint(w, endpoint: str, feature_values: dict) -> tuple[int, float | None]:
    """POST to the ML model serving endpoint via SDK. Returns (prediction, probability)."""
    body = w.api_client.do(
        "POST",
        f"/serving-endpoints/{endpoint}/invocations",
        body={"dataframe_records": [feature_values]},
    )
    raw = body.get("predictions", [None])[0]
    if isinstance(raw, dict):
        return int(raw.get("prediction", raw.get("0", 0))), raw.get("probability")
    return int(raw) if raw is not None else 0, None


@tool
@mlflow.trace(span_type="TOOL")
def predict_hiring_score(
    candidate_id: str,
    education_score: int = None,
    experience_score: int = None,
    leadership_score: int = None,
    certification_score: int = None,
    skills_match_score: int = None,
    industry_relevance_score: int = None,
    interview_score: int = None,
    culture_fit: int = None,
) -> str:
    """Predict whether a candidate should be hired using the trained ML model.
    Works for any candidate. If all 8 feature scores are supplied as arguments the
    database lookup is skipped entirely — pass scores from conversation context when
    available. Otherwise fetches stored scores from the database and applies overrides.

    Args:
        candidate_id: Candidate ID, e.g. "C011" or "C001"
        education_score: Education rating 0–100
        experience_score: Experience rating 0–100
        leadership_score: Leadership rating 0–100
        certification_score: Certification rating 0–100
        skills_match_score: Skills match rating 0–100
        industry_relevance_score: Industry relevance rating 0–100
        interview_score: Interview rating 0–100
        culture_fit: Culture fit rating 0–100

    Returns a Data Science recommendation, confidence score, and feature breakdown.
    """
    cfg = _get_config()
    cid = candidate_id.upper().strip()

    # Collect all provided score overrides
    overrides = {
        "education_score":          education_score,
        "experience_score":         experience_score,
        "leadership_score":         leadership_score,
        "certification_score":      certification_score,
        "skills_match_score":       skills_match_score,
        "industry_relevance_score": industry_relevance_score,
        "interview_score":          interview_score,
        "culture_fit":              culture_fit,
    }
    provided = {k: v for k, v in overrides.items() if v is not None}

    # ── If all 8 scores provided, skip DB fetch ────────────────────────────────
    if len(provided) == len(FEATURE_COLS):
        row = {"candidate_id": cid, "first_name": cid, "last_name": "", "job_id": "unknown", "hired": None}
    else:
        # Try DB fetch to fill missing scores and get candidate display info
        row = _fetch_candidate(cfg["catalog"], cfg["schema"], cid, cfg)
        if row is None:
            # DB fetch failed — use provided scores only
            row = {"candidate_id": cid, "first_name": cid, "last_name": "", "job_id": "unknown", "hired": None}

    name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip() or cid
    job_id = row.get("job_id", "unknown")

    # ── Apply all provided overrides ───────────────────────────────────────────
    for k, v in provided.items():
        row[k] = v

    # ── Check for missing features ─────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if row.get(c) is None]
    if missing:
        return (
            f"Cannot score {name} ({cid}) — missing features: {', '.join(missing)}.\n"
            f"Please supply the missing scores to run the prediction."
        )

    feature_values = {col: int(row[col]) for col in FEATURE_COLS}

    # ── Call ML endpoint ───────────────────────────────────────────────────────
    try:
        w = _get_ws()
        prediction, probability = _call_endpoint(w, cfg["endpoint"], feature_values)
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
        actual = f"Historical outcome: {'Hired ✅' if int(hired_label) == 1 else 'Not Hired ❌'}"
    else:
        actual = "Historical outcome: N/A (no prior decision on record)"

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
