"""
Tool: search_resumes
Uses Databricks Vector Search to find relevant resumes by semantic similarity.
Returns the top-3 matching resume excerpts for the agent to summarize.
"""
import os
import mlflow
from langchain_core.tools import tool
from config_helper import cfg_get


def _get_vs_config():
    return (
        cfg_get("vs_endpoint_name", "VS_ENDPOINT_NAME", "bx3_hrd_vs_endpoint"),
        cfg_get("target_catalog",   "TARGET_CATALOG",   "bx4"),
        cfg_get("target_schema",    "TARGET_SCHEMA",    "hrd_2030"),
        cfg_get("vs_index",         "VS_INDEX",         "hr_resumes_vs_index"),
    )


@tool
@mlflow.trace(span_type="TOOL")
def search_resumes(query: str) -> str:
    """Search candidate resumes using semantic vector search to find candidates matching
    specific qualifications, experience, or skills. Use this to look up a specific candidate's
    background, summarize their resume, or find candidates with particular expertise.
    Examples: 'Sarah Chen background', 'SPHR certified pharma HR director', 'MBA leadership experience'."""
    _vs_endpoint, catalog, schema, vs_index = _get_vs_config()

    try:
        # Query through the Databricks SDK so auth follows the same chain as the other
        # tools: on the endpoint that is hire-right-agent-sp via DATABRICKS_HOST +
        # DATABRICKS_CLIENT_ID/SECRET (OAuth M2M). The no-arg VectorSearchClient()
        # instead asks MLflow for a static token, which is None under OAuth M2M →
        # "Please specify either personal access token or service principal ...".
        from databricks.sdk import WorkspaceClient

        cols = ["candidate_id", "first_name", "last_name", "current_title", "resume_text"]
        resp = WorkspaceClient().vector_search_indexes.query_index(
            index_name=f"{catalog}.{schema}.{vs_index}",
            columns=cols,
            query_text=query,
            num_results=3,
        )

        rows = (resp.result.data_array if resp.result else None) or []
        if not rows:
            return "No matching resumes found for that query."
        if resp.manifest and resp.manifest.columns:
            cols = [c.name for c in resp.manifest.columns]

        output_parts = []
        for row in rows:
            rec = dict(zip(cols, row))
            name = f"{rec.get('first_name', '')} {rec.get('last_name', '')}".strip()
            cid = rec.get("candidate_id", "")
            title = rec.get("current_title", "")
            resume_text = str(rec.get("resume_text", ""))
            output_parts.append(
                f"**{name}** ({cid})\nTitle: {title}\n\nResume:\n{resume_text}"
            )

        return "\n\n---\n\n".join(output_parts)

    except Exception as e:
        return f"Error searching resumes: {str(e)}"
