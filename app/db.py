"""
Lakebase (PostgreSQL) access for the Hire Right app.

Serves the app's UI data (candidates, jobs) from low-latency synced tables and
persists HR annotations to a native transactional table — all in the same
Postgres schema, so annotations join to candidate info on candidate_id.

Auth model:
  - Deployed app: the app service principal connects via a short-lived OAuth
    token minted by `w.database.generate_database_credential`. The Lakebase
    resource binding (CAN_CONNECT_AND_CREATE) grants the SP that ability.
  - Local dev: same credential flow using the configured CLI profile.
The token is injected fresh on every physical connect (SQLAlchemy do_connect),
so it never goes stale for a long-lived pool.
"""
import os
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Optional

from databricks.sdk import WorkspaceClient
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

SCHEMA = os.environ.get("TARGET_SCHEMA", "hrd_2030")

IS_DATABRICKS_APP = bool(os.environ.get("DATABRICKS_APP_NAME"))

LAKEBASE_HOST = os.environ.get("LAKEBASE_HOST", "")
LAKEBASE_DATABASE = os.environ.get("LAKEBASE_DATABASE", "databricks_postgres")
LAKEBASE_INSTANCE_NAME = os.environ.get("LAKEBASE_INSTANCE_NAME", "hire-right-lb")
# Synced tables + annotations land in the Postgres schema matching the UC schema.
PG_SCHEMA = os.environ.get("PG_SCHEMA", SCHEMA)
ANNOTATIONS_TABLE = "candidate_annotations"


def get_workspace_client() -> WorkspaceClient:
    if IS_DATABRICKS_APP:
        try:
            return WorkspaceClient(auth_type="oauth-m2m")
        except Exception:
            return WorkspaceClient()
    profile = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
    return WorkspaceClient(profile=profile)


# ── Engine ───────────────────────────────────────────────────────────────────
_engine: Optional[Engine] = None


def _inject_credential(dialect, conn_rec, cargs, cparams):
    """SQLAlchemy do_connect hook: inject a fresh OAuth token as the password."""
    w = get_workspace_client()
    cred = w.database.generate_database_credential(instance_names=[LAKEBASE_INSTANCE_NAME])
    cparams["password"] = cred.token


def get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine

    w = get_workspace_client()

    host = LAKEBASE_HOST
    if not host:
        instance = w.database.get_database_instance(LAKEBASE_INSTANCE_NAME)
        host = instance.read_write_dns

    # SP client_id when deployed, user email for local dev.
    username = w.config.client_id if w.config.client_id else w.current_user.me().user_name
    port = os.environ.get("PGPORT", "5432")

    logger.info("Connecting to Lakebase host=%s db=%s user=%s", host, LAKEBASE_DATABASE, username)

    url = f"postgresql+psycopg://{username}:@{host}:{port}/{LAKEBASE_DATABASE}"
    engine = create_engine(
        url,
        pool_recycle=45 * 60,
        pool_size=4,
        pool_pre_ping=True,
        connect_args={"sslmode": "require"},
    )
    event.listen(engine, "do_connect", _inject_credential)
    _engine = engine
    return engine


# ── Serialization ────────────────────────────────────────────────────────────
def _jsonable(val):
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    return val


def execute_query(sql: str, params: dict = None) -> list[dict]:
    """Run a read query and return rows as JSON-safe dicts."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(f"SET search_path TO {PG_SCHEMA}, public"))
        result = conn.execute(text(sql), params or {})
        cols = list(result.keys())
        return [{c: _jsonable(row[i]) for i, c in enumerate(cols)} for row in result]


def execute_write(sql: str, params: dict = None) -> list[dict]:
    """Run a write (INSERT/UPDATE/DELETE), commit, and return any RETURNING rows."""
    engine = get_engine()
    with engine.begin() as conn:  # begin() commits on success
        conn.execute(text(f"SET search_path TO {PG_SCHEMA}, public"))
        result = conn.execute(text(sql), params or {})
        if result.returns_rows:
            cols = list(result.keys())
            return [{c: _jsonable(row[i]) for i, c in enumerate(cols)} for row in result]
        return []


def ensure_annotations_table() -> None:
    """Idempotently ensure the annotations table exists (fallback if the setup
    notebook hasn't run yet). Safe no-op when it already exists."""
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {PG_SCHEMA}.{ANNOTATIONS_TABLE} (
            id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            candidate_id TEXT        NOT NULL,
            note         TEXT        NOT NULL,
            author       TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {PG_SCHEMA}, public"))
            conn.execute(text(ddl))
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_{ANNOTATIONS_TABLE}_candidate "
                f"ON {PG_SCHEMA}.{ANNOTATIONS_TABLE} (candidate_id)"
            ))
        logger.info("Annotations table ensured.")
    except Exception as e:
        logger.warning("Could not ensure annotations table (may already exist / perms): %s", e)
