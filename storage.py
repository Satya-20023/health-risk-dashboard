"""Local persistence. Set MONGO_URI to use MongoDB Atlas instead of SQLite."""

from __future__ import annotations

import os
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
            age INTEGER,
            bmi REAL,
            systolic_bp INTEGER,
            cholesterol INTEGER,
            glucose INTEGER,
            smoking INTEGER,
            exercise_hours REAL,
            family_history INTEGER,
            risk_label TEXT,
            probability REAL
        )
        """
    )
    return conn


def save_assessment(payload: dict[str, Any]) -> None:
    uri = os.getenv("MONGO_URI")
    if uri:
        try:
            from pymongo import MongoClient

            client = MongoClient(uri)
            client.get_default_database()["assessments"].insert_one(payload)
            return
        except Exception:
            pass
    conn = _connect()
    cols = [
        "created_at",
        "age",
        "bmi",
        "systolic_bp",
        "cholesterol",
        "glucose",
        "smoking",
        "exercise_hours",
        "family_history",
        "risk_label",
        "probability",
    ]
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    conn.execute(
        f"INSERT INTO assessments ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [payload.get(c) for c in cols],
    )
    conn.commit()
    conn.close()


def recent(limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM assessments ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
