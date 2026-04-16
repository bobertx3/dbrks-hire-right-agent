"""
Tool: search_resumes
Uses Databricks Vector Search to find relevant resumes by semantic similarity.
Returns the top-3 matching resume excerpts for the agent to summarize.
"""
import os
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
def search_resumes(query: str) -> str:
    """Search candidate resumes using semantic vector search to find candidates matching
    specific qualifications, experience, or skills. Use this to look up a specific candidate's
    background, summarize their resume, or find candidates with particular expertise.
    Examples: 'Sarah Chen background', 'SPHR certified pharma HR director', 'MBA leadership experience'."""
    vs_endpoint, catalog, schema, vs_index = _get_vs_config()

    try:
        from databricks.vector_search.client import VectorSearchClient

        vsc = VectorSearchClient(disable_notice=True)
        index = vsc.get_index(
            endpoint_name=vs_endpoint,
            index_name=f"{catalog}.{schema}.{vs_index}",
        )
        results = index.similarity_search(
            query_text=query,
            columns=["candidate_id", "first_name", "last_name", "current_title", "resume_text"],
            num_results=3,
        )

        if not results or not results.get("result", {}).get("data_array"):
            return "No matching resumes found for that query."

        cols = [c["name"] for c in results["result"]["manifest"]["columns"]]
        output_parts = []
        for row in results["result"]["data_array"]:
            rec = dict(zip(cols, row))
            name = f"{rec.get('first_name', '')} {rec.get('last_name', '')}".strip()
            cid = rec.get("candidate_id", "")
            title = rec.get("current_title", "")
            # Truncate resume text to 800 chars for context efficiency
            resume_snippet = str(rec.get("resume_text", ""))[:800]
            output_parts.append(
                f"**{name}** ({cid})\nTitle: {title}\n\nResume excerpt:\n{resume_snippet}..."
            )

        return "\n\n---\n\n".join(output_parts)

    except Exception as e:
        return f"Error searching resumes: {str(e)}"
