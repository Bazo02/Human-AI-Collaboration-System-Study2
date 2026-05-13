# app/main.py

from __future__ import annotations

import os
import time
import uuid
from typing import Dict, Any, List, Optional

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file

from app.ai import get_ai_advice
from app.logger import log_event, log_decision, log_survey, log_participant, mark_participant_completed
from app.analysis import generate_results

from app.config import (
    SECRET_KEY,
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    TARGET_COL,
    DROP_COLS_FOR_UI,
    CASES_PER_PARTICIPANT,
    TOTAL_CASES_PER_PARTICIPANT,
    APPROVAL_THRESHOLD,
    ADMIN_PASSWORD,
    SQLITE_DB_PATH,
    PARTICIPANT_SUMMARY_PATH,
)

from app.db import (
    db_count_rows,
    db_get_participant_stats,
    db_delete_participant,
    db_clear_all,
    db_get_participant_count,
    init_db,
)

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = SECRET_KEY


FIELD_DESCRIPTIONS: Dict[str, str] = {
    "Age": "The applicant's age in years.",
    "Job": "The applicant's employment skill level.",
    "Housing": "The applicant's housing situation (own, rent, or free).",
    "Saving accounts": "The applicant's savings account balance (little, moderate, quite rich, or rich).",
    "Checking account": "The applicant's checking account balance (little, moderate, or rich).",
    "Credit amount": "The total amount of credit requested in euros.",
    "Duration": "The requested loan repayment period in months.",
    "Purpose": "The stated reason for the loan.",
}

FIELD_ORDER = [
    "Age", "Job", "Housing", "Purpose",
    "Credit amount", "Duration",
    "Saving accounts", "Checking account",
]


def _load_cases() -> pd.DataFrame:
    if os.path.exists(CASES_FOR_STUDY_PATH):
        df = pd.read_csv(CASES_FOR_STUDY_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = range(1, len(df) + 1)

    return df


CASES_DF = _load_cases()


def _pick_cases_for_participant() -> Dict[str, List[Dict[str, Any]]]:
    seed = session.get("seed")
    if seed is None:
        seed = int(time.time())
        session["seed"] = seed

    n_needed = TOTAL_CASES_PER_PARTICIPANT
    if len(CASES_DF) < n_needed:
        df_sample = CASES_DF.sample(n=n_needed, random_state=seed, replace=True).reset_index(drop=True)
    else:
        df_sample = CASES_DF.sample(n=n_needed, random_state=seed).reset_index(drop=True)

    cases = df_sample.to_dict(orient="records")

    return {
        "baseline": cases[:CASES_PER_PARTICIPANT],
        "ai": cases[CASES_PER_PARTICIPANT:CASES_PER_PARTICIPANT * 2],
    }


def _ui_case_view(case_row: Dict[str, Any]) -> Dict[str, Any]:
    view = dict(case_row)
    view.pop(TARGET_COL, None)
    view.pop("case_id", None)
    for col in DROP_COLS_FOR_UI:
        view.pop(col, None)

    ordered = {k: view[k] for k in FIELD_ORDER if k in view}
    for k, v in view.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _features_for_model(case_row: Dict[str, Any]) -> Dict[str, Any]:
    feats = dict(case_row)
    feats.pop(TARGET_COL, None)
    feats.pop("case_id", None)
    return feats


def _require_admin():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    return None


def _get_condition_order() -> tuple[str, str]:
    count = db_get_participant_count()
    if count % 2 == 0:
        return "AB", "baseline"
    else:
        return "BA", "ai"


def _get_next_block(current_block: str, condition_order: str) -> Optional[str]:
    if condition_order == "AB":
        return "ai" if current_block == "baseline" else None
    else:
        return "baseline" if current_block == "ai" else None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    participant_id = request.form.get("participant_id", "").strip()
    if not participant_id:
        participant_id = f"p_{uuid.uuid4().hex[:8]}"

    age_group = request.form.get("age_group", "").strip()
    background = request.form.get("background", "").strip()
    ai_familiarity = request.form.get("ai_familiarity", "").strip()
    finance_familiarity = request.form.get("finance_familiarity", "").strip()

    condition_order, first_block = _get_condition_order()

    session.clear()
    session["participant_id"] = participant_id
    session["condition_order"] = condition_order
    session["block"] = first_block
    session["case_index"] = 0
    session["cases_by_block"] = _pick_cases_for_participant()
    session["started_at"] = time.time()
    session["guidelines_ok"] = False
    session["guidelines_shown_logged"] = False

    log_participant(
        participant_id=participant_id,
        age_group=age_group,
        background=background,
        ai_familiarity=ai_familiarity,
        finance_familiarity=finance_familiarity,
        condition_order=condition_order,
    )
    log_event(participant_id, first_block, case_id=None, event="session_start", payload={})
    return redirect(url_for("guidelines"))


@app.route("/guidelines", methods=["GET", "POST"])
def guidelines():
    participant_id = session.get("participant_id")
    if not participant_id:
        return redirect(url_for("index"))

    block = session.get("block", "baseline")

    if request.method == "GET":
        if not session.get("guidelines_shown_logged"):
            log_event(participant_id, block, case_id=None, event="guidelines_shown", payload={})
            session["guidelines_shown_logged"] = True
        return render_template("guidelines.html", approval_threshold=APPROVAL_THRESHOLD)

    session["guidelines_ok"] = True
    log_event(participant_id, block, case_id=None, event="guidelines_accepted", payload={})
    return redirect(url_for("task"))


@app.route("/transition", methods=["GET"])
def transition():
    participant_id = session.get("participant_id")
    if not participant_id:
        return redirect(url_for("index"))

    if not session.get("guidelines_ok"):
        return redirect(url_for("guidelines"))

    if session.get("block") != "ai":
        return redirect(url_for("task"))

    return render_template("transition.html")


@app.route("/task", methods=["GET"])
def task():
    participant_id = session.get("participant_id")
    block = session.get("block", "baseline")
    condition_order = session.get("condition_order", "AB")
    cases_by_block = session.get("cases_by_block", {})
    cases = cases_by_block.get(block, [])
    idx = session.get("case_index", 0)

    if not participant_id or not cases:
        return redirect(url_for("index"))

    if not session.get("guidelines_ok"):
        return redirect(url_for("guidelines"))

    if idx >= len(cases):
        next_block = _get_next_block(block, condition_order)
        if next_block is not None:
            log_event(participant_id, block, case_id=None, event=f"{block}_block_complete", payload={})
            session["block"] = next_block
            session["case_index"] = 0
            log_event(participant_id, next_block, case_id=None, event=f"{next_block}_block_start", payload={})
            if next_block == "ai":
                return redirect(url_for("transition"))
            return redirect(url_for("task"))
        return redirect(url_for("survey"))

    case_row = cases[idx]
    case_id = case_row.get("case_id")
    case_for_ui = _ui_case_view(case_row)

    ai_payload: Optional[Dict[str, Any]] = None
    if block == "ai":
        ai_payload = get_ai_advice(
            features=_features_for_model(case_row),
            approval_threshold=APPROVAL_THRESHOLD
        )

    log_event(participant_id, block, case_id=case_id, event="case_shown", payload={"index": idx, "block": block})

    return render_template(
        "task.html",
        participant_id=participant_id,
        condition=block,
        case_id=case_id,
        case=case_for_ui,
        ai=ai_payload,
        case_number=idx + 1,
        total_cases=len(cases),
        field_descriptions=FIELD_DESCRIPTIONS,
    )


@app.route("/submit_decision", methods=["POST"])
def submit_decision():
    participant_id = session.get("participant_id")
    block = session.get("block", "baseline")
    condition_order = session.get("condition_order", "AB")
    cases_by_block = session.get("cases_by_block", {})
    cases = cases_by_block.get(block, [])
    idx = session.get("case_index", 0)

    if not participant_id or not cases:
        return jsonify({"ok": False, "error": "No active session"}), 400

    if not session.get("guidelines_ok"):
        return jsonify({"ok": False, "error": "Guidelines not accepted"}), 400

    if idx >= len(cases):
        return jsonify({"ok": False, "error": "No more cases"}), 400

    payload = request.get_json(force=True) or {}
    case_id_from_client = payload.get("case_id")
    decision = payload.get("decision")
    time_ms = payload.get("time_ms")

    if decision not in ("Approve", "Reject"):
        return jsonify({"ok": False, "error": "Invalid decision"}), 400

    current_case = cases[idx]
    current_case_id = current_case.get("case_id")
    if str(case_id_from_client) != str(current_case_id):
        return jsonify({"ok": False, "error": "Case mismatch"}), 400

    gt = int(current_case.get(TARGET_COL))
    correct = int((decision == "Approve" and gt == 1) or (decision == "Reject" and gt == 0))

    ai_recommendation = None
    ai_confidence = None
    ai_prob_approve = None
    if block == "ai":
        ai_payload = get_ai_advice(
            features=_features_for_model(current_case),
            approval_threshold=APPROVAL_THRESHOLD
        )
        ai_recommendation = ai_payload.get("recommendation")
        ai_confidence = ai_payload.get("confidence")
        ai_prob_approve = ai_payload.get("prob_approve")

    log_decision(
        participant_id=participant_id,
        condition=block,
        case_id=current_case_id,
        decision=decision,
        ground_truth=gt,
        correct=correct,
        time_ms=time_ms,
        ai_followed=payload.get("ai_followed"),
        ai_seen=payload.get("ai_seen"),
        explanation_opened=payload.get("explanation_opened"),
        ai_recommendation=ai_recommendation,
        ai_confidence=ai_confidence,
        ai_prob_approve=ai_prob_approve,
    )

    session["case_index"] = idx + 1

    if session["case_index"] >= len(cases):
        next_block = _get_next_block(block, condition_order)
        if next_block is not None:
            log_event(participant_id, block, case_id=None, event=f"{block}_block_complete", payload={})
            session["block"] = next_block
            session["case_index"] = 0
            log_event(participant_id, next_block, case_id=None, event=f"{next_block}_block_start", payload={})
            if next_block == "ai":
                return jsonify({"ok": True, "next": "/transition"})
            return jsonify({"ok": True, "next": "/task"})
        log_event(participant_id, block, case_id=None, event=f"{block}_block_complete", payload={})
        return jsonify({"ok": True, "next": "/survey"})

    return jsonify({"ok": True, "next": "/task"})


@app.route("/survey", methods=["GET", "POST"])
def survey():
    participant_id = session.get("participant_id")
    if not participant_id:
        return redirect(url_for("index"))

    if not session.get("guidelines_ok"):
        return redirect(url_for("guidelines"))

    if request.method == "GET":
        return render_template("survey.html")

    log_survey(participant_id=participant_id, condition="ai", answers=dict(request.form.items()))
    mark_participant_completed(participant_id)
    log_event(participant_id, "ai", case_id=None, event="survey_submitted", payload={})
    log_event(participant_id, "ai", case_id=None, event="study_completed", payload={})
    return redirect(url_for("done"))


@app.route("/done", methods=["GET"])
def done():
    return render_template("done.html")


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html")

    pw = request.form.get("password", "")
    if pw == ADMIN_PASSWORD:
        session["is_admin"] = True
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html", error="Wrong password")


@app.route("/admin/logout", methods=["GET"])
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("index"))


@app.route("/admin/dashboard", methods=["GET"])
def admin_dashboard():
    r = _require_admin()
    if r:
        return r

    counts = {
        "participants": db_count_rows("participants"),
        "decisions": db_count_rows("decisions"),
        "events": db_count_rows("events"),
        "surveys": db_count_rows("surveys"),
    }
    participants = db_get_participant_stats()
    return render_template("admin_dashboard.html", counts=counts, participants=participants)


@app.route("/admin/results", methods=["GET"])
def admin_results():
    r = _require_admin()
    if r:
        return r

    results = generate_results(app.static_folder)
    return render_template("results.html", results=results)


@app.route("/admin/download_db", methods=["GET"])
def admin_download_db():
    r = _require_admin()
    if r:
        return r

    if not os.path.exists(SQLITE_DB_PATH):
        init_db()

    return send_file(
        SQLITE_DB_PATH,
        as_attachment=True,
        download_name="study.db",
        mimetype="application/x-sqlite3",
    )


@app.route("/admin/download_participant_summary", methods=["GET"])
def admin_download_participant_summary():
    r = _require_admin()
    if r:
        return r

    generate_results(app.static_folder)

    if not os.path.exists(PARTICIPANT_SUMMARY_PATH):
        return redirect(url_for("admin_results"))

    return send_file(
        PARTICIPANT_SUMMARY_PATH,
        as_attachment=True,
        download_name="participant_summary.csv",
        mimetype="text/csv",
    )


@app.route("/admin/upload_db", methods=["POST"])
def admin_upload_db():
    r = _require_admin()
    if r:
        return r

    uploaded_file = request.files.get("db_file")
    if uploaded_file and uploaded_file.filename:
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        uploaded_file.save(SQLITE_DB_PATH)
        init_db()

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/clear_all", methods=["POST"])
def admin_clear_all_route():
    r = _require_admin()
    if r:
        return r

    db_clear_all()
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete_participant", methods=["POST"])
def admin_delete_participant_route():
    r = _require_admin()
    if r:
        return r

    pid = request.form.get("participant_id", "").strip()
    if pid:
        db_delete_participant(pid)

    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    app.run(debug=True)