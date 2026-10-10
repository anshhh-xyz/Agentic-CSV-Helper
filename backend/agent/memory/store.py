import json 
import sqlite3
from datetime import datetime, timezone
from agent.config import MEMORY_DB_PATH


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(scope, key)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def upsert(scope: str, kind: str, key: str, value) -> None:
    if not scope:
        raise ValueError("scope is required (use 'global' for cross-dataset facts).")
    if not key or not key.strip():
        raise ValueError("key is required.")


    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(value)
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO memory (scope, kind, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, key) DO UPDATE SET
                kind = excluded.kind,
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (scope, kind, key.strip(), payload, now, now),
        )
        conn.commit()
    finally:
        conn.close()

def get_all(scope: str) -> list[dict]:
    if not scope:
        raise ValueError("scope is required.")

    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT kind, key, value, updated_at FROM memory WHERE scope = ? ORDER BY updated_at DESC",
            (scope,),
        ).fetchall()
    finally:
        conn.close()

    return [
        {"kind": kind, "key": key, "value": json.loads(value), "updated_at": updated_at}
        for kind, key, value, updated_at in rows
    ]


def delete(scope: str, key: str) -> bool:
    if not scope or not key:
        raise ValueError("scope and key are both required.")

    conn = _connect()
    try:
        cursor = conn.execute(
            "DELETE FROM memory WHERE scope = ? AND key = ?",
            (scope, key.strip()),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()