# app/db.py

from __future__ import annotations

import os
import sqlite3
from typing import Optional, Any, Dict

from app.config import SQLITE_DB_PATH


def _ensure_parent_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    _ensure_parent_dir(SQLITE_DB_PATH)
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return any(str(row[1]) == column for row in rows)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    if not _column_exists(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS participants (
        participant_id TEXT PRIMARY KEY,
        ts_utc TEXT NOT NULL,
        age_group TEXT,
        background TEXT,
        ai_familiarity TEXT,
        finance_familiarity TEXT,
        condition_order TEXT DEFAULT 'AB',
        completed INTEGER NOT NULL DEFAULT 0
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        case_id TEXT,
        event TEXT NOT NULL,
        payload_json TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        case_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        ground_truth INTEGER NOT NULL,
        correct INTEGER NOT NULL,
        time_ms INTEGER,
        ai_followed INTEGER,
        ai_seen INTEGER,
        explanation_opened INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS surveys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        answers_json TEXT NOT NULL
    )
    """)

    _ensure_column(conn, "decisions", "ai_recommendation", "TEXT")
    _ensure_column(conn, "decisions", "ai_confidence", "REAL")
    _ensure_column(conn, "decisions", "ai_prob_approve", "REAL")
    _ensure_column(conn, "participants", "condition_order", "TEXT")

    conn.commit()
    conn.close()


def db_get_participant_count() -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM participants")
    n = int(cur.fetchone()[0])
    conn.close()
    return n


def db_count_rows(table: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    n = int(cur.fetchone()[0])
    conn.close()
    return n


def db_list_participants() -> list[str]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT participant_id FROM participants
        UNION
        SELECT DISTINCT participant_id FROM decisions
        UNION
        SELECT DISTINCT participant_id FROM surveys
        UNION
        SELECT DISTINCT participant_id FROM events
    """)
    rows = [r[0] for r in cur.fetchall()]
    conn.close()

    cleaned = sorted([p for p in rows if p and str(p).strip() and str(p) != "ADMIN"])
    return cleaned


def db_get_participant_stats() -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        WITH participant_ids AS (
            SELECT participant_id FROM participants
            UNION
            SELECT participant_id FROM decisions
            UNION
            SELECT participant_id FROM surveys
            UNION
            SELECT participant_id FROM events
        )
        SELECT
            p.participant_id,
            COALESCE(d.decisions_count, 0) AS decisions_count,
            COALESCE(e.events_count, 0) AS events_count,
            COALESCE(s.surveys_count, 0) AS surveys_count,
            COALESCE(pt.completed, 0) AS completed,
            COALESCE(pt.age_group, '') AS age_group,
            COALESCE(pt.background, '') AS background,
            COALESCE(pt.ai_familiarity, '') AS ai_familiarity,
            COALESCE(pt.finance_familiarity, '') AS finance_familiarity,
            COALESCE(pt.condition_order, 'AB') AS condition_order
        FROM participant_ids p
        LEFT JOIN (
            SELECT participant_id, COUNT(*) AS decisions_count
            FROM decisions
            GROUP BY participant_id
        ) d ON p.participant_id = d.participant_id
        LEFT JOIN (
            SELECT participant_id, COUNT(*) AS events_count
            FROM events
            GROUP BY participant_id
        ) e ON p.participant_id = e.participant_id
        LEFT JOIN (
            SELECT participant_id, COUNT(*) AS surveys_count
            FROM surveys
            GROUP BY participant_id
        ) s ON p.participant_id = s.participant_id
        LEFT JOIN participants pt ON p.participant_id = pt.participant_id
        WHERE p.participant_id IS NOT NULL
          AND TRIM(p.participant_id) != ''
          AND p.participant_id != 'ADMIN'
        ORDER BY p.participant_id
    """)

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "participant_id": row[0],
            "decisions": int(row[1]),
            "events": int(row[2]),
            "surveys": int(row[3]),
            "completed": int(row[4]),
            "age_group": row[5],
            "background": row[6],
            "ai_familiarity": row[7],
            "finance_familiarity": row[8],
            "condition_order": row[9],
        }
        for row in rows
    ]


def db_delete_participant(participant_id: str) -> dict:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM decisions WHERE participant_id = ?", (participant_id,))
    d_before = int(cur.fetchone()[0])
    cur.execute("DELETE FROM decisions WHERE participant_id = ?", (participant_id,))

    cur.execute("SELECT COUNT(*) FROM surveys WHERE participant_id = ?", (participant_id,))
    s_before = int(cur.fetchone()[0])
    cur.execute("DELETE FROM surveys WHERE participant_id = ?", (participant_id,))

    cur.execute("SELECT COUNT(*) FROM events WHERE participant_id = ?", (participant_id,))
    e_before = int(cur.fetchone()[0])
    cur.execute("DELETE FROM events WHERE participant_id = ?", (participant_id,))

    cur.execute("SELECT COUNT(*) FROM participants WHERE participant_id = ?", (participant_id,))
    p_before = int(cur.fetchone()[0])
    cur.execute("DELETE FROM participants WHERE participant_id = ?", (participant_id,))

    conn.commit()
    conn.close()

    return {"decisions": d_before, "surveys": s_before, "events": e_before, "participants": p_before}


def db_clear_table(table: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    before = int(cur.fetchone()[0])
    cur.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    return before


def db_clear_all() -> dict:
    return {
        "participants": db_clear_table("participants"),
        "events": db_clear_table("events"),
        "decisions": db_clear_table("decisions"),
        "surveys": db_clear_table("surveys"),
    }