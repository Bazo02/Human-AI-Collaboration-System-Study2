# app/logger.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.db import get_conn, init_db


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


init_db()


def log_participant(
    participant_id: str,
    age_group: str,
    background: str,
    ai_familiarity: str,
    finance_familiarity: str,
    condition_order: str = "AB",
):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO participants (
            participant_id, ts_utc, age_group, background,
            ai_familiarity, finance_familiarity, condition_order, completed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(participant_id) DO UPDATE SET
            age_group = excluded.age_group,
            background = excluded.background,
            ai_familiarity = excluded.ai_familiarity,
            finance_familiarity = excluded.finance_familiarity,
            condition_order = excluded.condition_order
        """,
        (
            str(participant_id),
            _now_utc_iso(),
            str(age_group),
            str(background),
            str(ai_familiarity),
            str(finance_familiarity),
            str(condition_order),
        ),
    )
    conn.commit()
    conn.close()


def mark_participant_completed(participant_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO participants (
            participant_id, ts_utc, age_group, background, ai_familiarity, finance_familiarity, completed
        )
        VALUES (?, ?, '', '', '', '', 1)
        ON CONFLICT(participant_id) DO UPDATE SET
            completed = 1
        """,
        (
            str(participant_id),
            _now_utc_iso(),
        ),
    )
    conn.commit()
    conn.close()


def log_event(
    participant_id: str,
    condition: str,
    case_id: Optional[Any],
    event: str,
    payload: Dict[str, Any],
):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events (ts_utc, participant_id, condition, case_id, event, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            _now_utc_iso(),
            str(participant_id),
            str(condition),
            None if case_id is None else str(case_id),
            str(event),
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def log_decision(
    participant_id: str,
    condition: str,
    case_id: Any,
    decision: str,
    ground_truth: int,
    correct: int,
    time_ms: Optional[int],
    ai_followed: Optional[int],
    ai_seen: Optional[int],
    explanation_opened: Optional[int],
    ai_recommendation: Optional[str] = None,
    ai_confidence: Optional[float] = None,
    ai_prob_approve: Optional[float] = None,
):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO decisions (
            ts_utc, participant_id, condition, case_id,
            decision, ground_truth, correct, time_ms,
            ai_followed, ai_seen, explanation_opened,
            ai_recommendation, ai_confidence, ai_prob_approve
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now_utc_iso(),
            str(participant_id),
            str(condition),
            str(case_id),
            str(decision),
            int(ground_truth),
            int(correct),
            None if time_ms is None else int(time_ms),
            None if ai_followed is None else int(ai_followed),
            None if ai_seen is None else int(ai_seen),
            None if explanation_opened is None else int(explanation_opened),
            None if ai_recommendation is None else str(ai_recommendation),
            None if ai_confidence is None else float(ai_confidence),
            None if ai_prob_approve is None else float(ai_prob_approve),
        ),
    )
    conn.commit()
    conn.close()


def log_survey(participant_id: str, condition: str, answers: Dict[str, Any]):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO surveys (ts_utc, participant_id, condition, answers_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            _now_utc_iso(),
            str(participant_id),
            str(condition),
            json.dumps(answers, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()