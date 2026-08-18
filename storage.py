"""Local persistence for scored assessments."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "patients.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            risk_label TEXT,
            probability REAL,
            username TEXT
        )
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(assessments)").fetchall()}
    if "username" not in cols:
        conn.execute("ALTER TABLE assessments ADD COLUMN username TEXT")
    return conn


def save_assessment(
    payload: dict[str, Any], risk_label: str, probability: float, username: str | None = None
) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO assessments (created_at, payload, risk_label, probability, username) VALUES (?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(payload),
            risk_label,
            probability,
            username,
        ),
    )
    conn.commit()
    conn.close()


def recent(limit: int = 25, username: str | None = None) -> list[dict[str, Any]]:
    conn = _connect()
    if username:
        rows = conn.execute(
            "SELECT * FROM assessments WHERE username = ? ORDER BY id DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM assessments ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item.update(json.loads(item.get("payload") or "{}"))
        except json.JSONDecodeError:
            pass
        out.append(item)
    return out
