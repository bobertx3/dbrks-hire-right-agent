"""
config_helper.py — Centralised config resolution for agent_src tools.

Reads from environment variables set on the serving endpoint.
In development (notebook context), values come from the .env file loaded via dotenv.

Usage:
    from config_helper import cfg_get
    genie_space_id = cfg_get("genie_space_id", "GENIE_SPACE_ID")
"""
import os


def cfg_get(key: str, env_var: str, default: str = "") -> str:
    """Return config value from environment variable with fallback to default."""
    return os.getenv(env_var, default)
