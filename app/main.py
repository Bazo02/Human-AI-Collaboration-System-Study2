# app/main.py

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for

from app.ai import get_ai_advice
from app.analysis import generate_results
from app.config import (
    ADMIN_PASSWORD,
    APPROVAL_THRESHOLD,
    CASES_FOR_STUDY_PATH,
    CASES_PER_PARTICIPANT,
    DATA_PATH,
    DROP_COLS_FOR_UI,
    PARTICIPANT_SUMMARY_PATH,
    SECRET_KEY,
    SQLITE_DB_PATH,
    TARGET_COL,
    TOTAL_CASES_PER_PARTICIPANT,
)
from app.db import (
    db_clear_all,
    db_count_rows,
    db_delete_participant,
    db_get_participant_count,
    db_get_participant_stats,
    init_db,
)
from app.logger import (
    log_decision,
    log_event,
    log_participant,
    log_survey,
    mark_participant_completed,
)

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = SECRET_KEY

FIELD_DISPLAY_NAMES: Dict[str, str] = {
    "person_age": "Applicant Age",
    "person_education": "Education Level",
    "person_income": "Annual Income",
    "person_emp_exp": "Years of Work Experience",
    "person_home_ownership": "Housing Situation",
    "loan_amnt": "Requested Loan Amount",
    "loan_intent": "Purpose of the Loan",
    "loan_int_rate": "Loan Interest Rate",
    "credit_score": "Credit Score",
    "previous_loan_defaults_on_file": "Previous Loan Repayment Problems",
}

FIELD_DESCRIPTIONS: Dict[str, str] = {
    "person_age": "The age of the applicant.",
    "person_education": "The applicant's highest completed education level.",
    "person_income": "The applicant's yearly income before taxes.",
    "person_emp_exp": "How many years of work experience the applicant has.",
    "person_home_ownership": "The applicant's current housing situation, such as renting, owning, or having a mortgage.",
    "loan_amnt": "The amount of money the applicant wants to borrow.",
    "loan_intent": "The main purpose of the loan.",
    "loan_int_rate": "The interest rate connected to the loan. A higher interest rate makes repayment more expensive.",
    "credit_score": "A score that represents the applicant's creditworthiness. A higher score usually indicates lower repayment risk.",
    "previous_loan_defaults_on_file": "Shows whether the applicant has previously failed to repay loans.",
}

FIELD_ORDER = [
    "person_age",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_intent",
    "loan_amnt",
    "loan_int_rate",
    "credit_score",
    "previous_loan_defaults_on_file",
]


# Loads the loan cases CSV, falling back to the full dataset if study cases don't exist
def _load_cases() -> pd.DataFrame:
    if os.path.exists(CASES_FOR_STUDY_PATH):
        df = pd.read_csv(CASES_FOR_STUDY_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    if "loan_status" in df.columns and TARGET_COL not in df.columns:
        df[TARGET_COL] = df["loan_status"].astype(int)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = range(1, len(df) + 1)

    return df


CASES_DF = _load_cases()


# Randomly picks cases for one participant and splits them into baseline and AI blocks
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


# Returns a cleaned case dict with sensitive columns removed and fields in display order
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


# Returns a case dict with only the features needed by the model
def _features_for_model(case_row: Dict[str, Any]) -> Dict[str, Any]:
    feats = dict(case_row)
    feats.pop(TARGET_COL, None)
    feats.pop("case_id", None)
    feats.pop("loan_status", None)
    return feats


# Redirects to the admin login page if the user is not authenticated as admin
def _require_admin():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    return None


# Returns the condition order (AB or BA) and the first block for the next participant
def _get_condition_order() -> Tuple[str, str]:
    count = db_get_participant_count()
    if count % 2 == 0:
        return "AB", "baseline"
    return "BA", "ai"


# Returns the next block name given the current block and condition order
def _get_next_block(current_block: str, condition_order: str) -> Optional[str]:
    if condition_order == "AB":
        return "ai" if current_block == "baseline" else None

    return "baseline" if current_block == "ai" else None


# Shows the landing page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Handles form submission from the landing page and sets up the session
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

    log_event(
        participant_id,
        first_block,
        case_id=None,
        event="session_start",
        payload={"condition_order": condition_order},
    )

    return redirect(url_for("guidelines"))


# Shows the guidelines page and records when the participant accepts them
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


# Shows the transition screen between the two study conditions
@app.route("/transition", methods=["GET"])
def transition():
    participant_id = session.get("participant_id")

    if not participant_id:
        return redirect(url_for("index"))

    if not session.get("guidelines_ok"):
        return redirect(url_for("guidelines"))

    block = session.get("block")

    if block not in ("baseline", "ai"):
        return redirect(url_for("task"))

    return render_template("transition.html", next_condition=block)


# Shows one loan case and, in the AI condition, the AI recommendation
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

            return redirect(url_for("transition"))

        return redirect(url_for("survey"))

    case_row = cases[idx]
    case_id = case_row.get("case_id")
    case_for_ui = _ui_case_view(case_row)

    ai_payload: Optional[Dict[str, Any]] = None

    if block == "ai":
        ai_payload = get_ai_advice(
            features=_features_for_model(case_row),
            approval_threshold=APPROVAL_THRESHOLD,
        )

    log_event(
        participant_id,
        block,
        case_id=case_id,
        event="case_shown",
        payload={"index": idx, "block": block},
    )

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
        field_display_names=FIELD_DISPLAY_NAMES,
    )


# Receives a decision from the browser, saves it, and returns the next URL
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
            approval_threshold=APPROVAL_THRESHOLD,
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

            return jsonify({"ok": True, "next": "/transition"})

        log_event(participant_id, block, case_id=None, event=f"{block}_block_complete", payload={})

        return jsonify({"ok": True, "next": "/survey"})

    return jsonify({"ok": True, "next": "/task"})


# Shows the post-task survey and saves answers on submission
@app.route("/survey", methods=["GET", "POST"])
def survey():
    participant_id = session.get("participant_id")

    if not participant_id:
        return redirect(url_for("index"))

    if not session.get("guidelines_ok"):
        return redirect(url_for("guidelines"))

    if request.method == "GET":
        return render_template("survey.html")

    log_survey(participant_id=participant_id, condition="post_task", answers=dict(request.form.items()))
    mark_participant_completed(participant_id)
    log_event(participant_id, "post_task", case_id=None, event="survey_submitted", payload={})
    log_event(participant_id, "post_task", case_id=None, event="study_completed", payload={})

    return redirect(url_for("done"))


# Shows the completion page
@app.route("/done", methods=["GET"])
def done():
    return render_template("done.html")


# Shows the admin login form and checks the password on POST
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html")

    pw = request.form.get("password", "")

    if pw == ADMIN_PASSWORD:
        session["is_admin"] = True
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html", error="Wrong password")


# Logs out the admin by removing the session flag
@app.route("/admin/logout", methods=["GET"])
def admin_logout():
    session.pop("is_admin", None)

    return redirect(url_for("index"))


# Shows the admin dashboard with participant counts and stats
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


# Generates and shows the results page with charts and statistics
@app.route("/admin/results", methods=["GET"])
def admin_results():
    r = _require_admin()

    if r:
        return r

    results = generate_results(app.static_folder)

    return render_template("results.html", results=results)


# Lets the admin download the SQLite database file
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


# Lets the admin download the participant summary as a CSV file
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


# Lets the admin upload and replace the database file
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


# Deletes all data from every table
@app.route("/admin/clear_all", methods=["POST"])
def admin_clear_all_route():
    r = _require_admin()

    if r:
        return r

    db_clear_all()

    return redirect(url_for("admin_dashboard"))


# Deletes all data for a specific participant
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
