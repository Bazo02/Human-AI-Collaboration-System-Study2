# app/config.py

from __future__ import annotations

import os
import secrets


SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

DATA_PATH = os.path.join(DATA_DIR, "german_credit.csv")
CASES_FOR_STUDY_PATH = os.path.join(DATA_DIR, "cases_for_study.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "app", "model.joblib")
PARTICIPANT_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "participant_summary.csv")


TARGET_COL = "loan_approved"

DROP_COLS_FOR_UI = [
    "Sex",
]

CONDITION_NAMES = {
    "baseline": "Non-assisted (baseline)",
    "ai": "AI-assisted",
}


CASES_PER_PARTICIPANT = 12

TOTAL_CASES_PER_PARTICIPANT = CASES_PER_PARTICIPANT * 2

APPROVAL_THRESHOLD = 0.55


EVENTS_LOG_PATH = os.path.join(OUTPUTS_DIR, "events.csv")
DECISIONS_LOG_PATH = os.path.join(OUTPUTS_DIR, "decisions.csv")
SURVEYS_LOG_PATH = os.path.join(OUTPUTS_DIR, "surveys.csv")
SQLITE_DB_PATH = os.path.join(OUTPUTS_DIR, "study.db")


ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")