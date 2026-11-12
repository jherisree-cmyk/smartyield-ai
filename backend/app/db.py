"""SQLite persistence. SQLite is a single file, needs no server, and is
enough for one farm or one research group. The schema maps 1:1 to tables in
PostgreSQL or MongoDB collections if the app ever has to scale."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from smartyield import config

DB_PATH = Path(os.environ.get("SMARTYIELD_DB", config.DB_PATH))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    crop TEXT NOT NULL DEFAULT 'rice',
    area_ha REAL,
    location TEXT,
    sowing_date TEXT,
    variety TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    field_id INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    image_name TEXT NOT NULL,
    image_path TEXT NOT NULL,
    image_bytes INTEGER,
    capture_date TEXT NOT NULL,
    season TEXT,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | done | failed
    error TEXT,
    track TEXT,
    predicted_t_ha REAL,
    lo90_t_ha REAL,
    hi90_t_ha REAL,
    veg_cover_pct REAL,
    production_t REAL,
    actual_t_ha REAL,                         -- harvest measurement entered later (evaluation)
    summary_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_analyses_field ON analyses(field_id, capture_date);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    return con


@contextmanager
def session():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with session() as con:
        con.executescript(SCHEMA)


def row(r: sqlite3.Row | None) -> dict | None:
    return dict(r) if r is not None else None
