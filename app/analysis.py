# app/analysis.py

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from app.db import get_conn
from app.config import PARTICIPANT_SUMMARY_PATH

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


RESULTS_DIRNAME = "results"


# Tries to convert a value to int, returns None if it fails
def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


# Calculates the SUS score from survey answers
def _compute_sus_from_answers(answers: Dict[str, Any]) -> Optional[float]:
    scores: List[int] = []
    for i in range(1, 11):
        key = f"sus_q{i}"
        v = _to_int(answers.get(key))
        if v is None:
            return None
        if i % 2 == 1:
            scores.append(v - 1)
        else:
            scores.append(5 - v)
    return float(sum(scores) * 2.5)


# Calculates the average trust score from the three trust questions
def _compute_trust_from_answers(answers: Dict[str, Any]) -> Optional[float]:
    vals = []
    for k in ["trust_q1", "trust_q2", "trust_q3"]:
        v = _to_int(answers.get(k))
        if v is None:
            return None
        vals.append(v)
    return float(sum(vals) / len(vals))


# Extracts the free-text comment from survey answers if one exists
def _extract_comment(answers: Dict[str, Any]) -> str:
    candidate_keys = [
        "comment", "comments", "feedback", "message",
        "free_text", "additional_feedback", "open_feedback",
        "participant_comment", "notes"
    ]
    for k in candidate_keys:
        if k in answers:
            txt = str(answers.get(k, "")).strip()
            if txt:
                return txt
    for k, v in answers.items():
        if v is None:
            continue
        txt = str(v).strip()
        if not txt:
            continue
        if txt.isdigit():
            continue
        lowk = str(k).lower()
        if lowk.startswith(("sus_", "trust_")):
            continue
        if txt.lower() in ("baseline", "ai"):
            continue
        return txt
    return ""


# Reads a full database table and returns it as a DataFrame
def _read_table_as_df(table: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


# Parses raw survey rows and computes SUS, trust score, and comment per participant
def _parse_surveys_df(raw_surveys: pd.DataFrame) -> pd.DataFrame:
    if raw_surveys.empty:
        return raw_surveys

    rows = []
    for _, r in raw_surveys.iterrows():
        answers_json = r.get("answers_json", "{}")
        try:
            answers = json.loads(answers_json) if isinstance(answers_json, str) else {}
        except Exception:
            answers = {}

        rows.append({
            "participant_id": r.get("participant_id"),
            "condition": r.get("condition"),
            "sus_score": _compute_sus_from_answers(answers),
            "trust_score": _compute_trust_from_answers(answers),
            "comment": _extract_comment(answers),
        })

    return pd.DataFrame(rows)


# Runs a paired t-test between two conditions and returns descriptive stats
def _paired_stats(df: pd.DataFrame, baseline_col: str, ai_col: str) -> Dict[str, Any]:
    if df.empty or baseline_col not in df.columns or ai_col not in df.columns:
        return {}

    pair_df = df[["participant_id", baseline_col, ai_col]].dropna().copy()
    if pair_df.empty:
        return {}

    baseline = pair_df[baseline_col].astype(float).to_numpy()
    ai = pair_df[ai_col].astype(float).to_numpy()
    diff = ai - baseline

    result: Dict[str, Any] = {
        "n": int(len(pair_df)),
        "baseline_mean": float(np.mean(baseline)),
        "baseline_sd": float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0,
        "ai_mean": float(np.mean(ai)),
        "ai_sd": float(np.std(ai, ddof=1)) if len(ai) > 1 else 0.0,
        "mean_difference": float(np.mean(diff)),
        "sd_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
    }

    if len(diff) > 1 and np.std(diff, ddof=1) > 0:
        result["cohens_dz"] = float(np.mean(diff) / np.std(diff, ddof=1))
    else:
        result["cohens_dz"] = 0.0

    if scipy_stats is not None and len(diff) > 1:
        t_stat, p_value = scipy_stats.ttest_rel(ai, baseline, nan_policy="omit")
        sem = scipy_stats.sem(diff, nan_policy="omit")
        ci_low, ci_high = scipy_stats.t.interval(
            confidence=0.95,
            df=len(diff) - 1,
            loc=np.mean(diff),
            scale=sem,
        )
        result["t_statistic"] = float(t_stat)
        result["p_value"] = float(p_value)
        result["ci95_low"] = float(ci_low)
        result["ci95_high"] = float(ci_high)
    else:
        result["t_statistic"] = None
        result["p_value"] = None
        result["ci95_low"] = None
        result["ci95_high"] = None

    return result


# Groups participants by whether accuracy improved, worsened, or stayed the same with AI
def _accuracy_improvement_groups(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}

    needed = ["participant_id", "baseline_accuracy", "ai_accuracy"]
    if not all(c in df.columns for c in needed):
        return {}

    pair_df = df[needed].dropna().copy()
    if pair_df.empty:
        return {}

    pair_df["diff"] = pair_df["ai_accuracy"] - pair_df["baseline_accuracy"]

    improved = pair_df[pair_df["diff"] > 0]
    worsened = pair_df[pair_df["diff"] < 0]
    unchanged = pair_df[pair_df["diff"] == 0]

    result: Dict[str, Any] = {
        "n_improved": int(len(improved)),
        "n_worsened": int(len(worsened)),
        "n_unchanged": int(len(unchanged)),
        "improved_baseline_accuracy_mean": float(improved["baseline_accuracy"].mean()) if not improved.empty else None,
        "worsened_baseline_accuracy_mean": float(worsened["baseline_accuracy"].mean()) if not worsened.empty else None,
    }

    if scipy_stats is not None and len(improved) > 1 and len(worsened) > 1:
        t_stat, p_value = scipy_stats.ttest_ind(
            improved["baseline_accuracy"].to_numpy(),
            worsened["baseline_accuracy"].to_numpy(),
            equal_var=False,
        )
        result["group_ttest_t"] = float(t_stat)
        result["group_ttest_p"] = float(p_value)
    else:
        result["group_ttest_t"] = None
        result["group_ttest_p"] = None

    return result


# Computes Spearman correlation between trust score and AI-followed rate
def _spearman_trust_vs_ai_followed(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}

    needed = ["trust_score", "ai_ai_followed_rate"]
    if not all(c in df.columns for c in needed):
        return {}

    pair_df = df[needed].dropna()
    if len(pair_df) < 3:
        return {}

    if scipy_stats is not None:
        rs, p = scipy_stats.spearmanr(
            pair_df["trust_score"].to_numpy(),
            pair_df["ai_ai_followed_rate"].to_numpy(),
        )
        return {"rs": float(rs), "p_value": float(p), "n": int(len(pair_df))}

    return {}


# Returns descriptive statistics for the distribution of AI-followed rates
def _ai_followed_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "ai_ai_followed_rate" not in df.columns:
        return {}

    vals = df["ai_ai_followed_rate"].dropna().to_numpy()
    if len(vals) == 0:
        return {}

    return {
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
    }


# Splits results by condition order (AB vs BA) to check for order effects
def _counterbalance_subgroup_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "condition_order" not in df.columns:
        return {}

    results: Dict[str, Any] = {}

    for order in ["AB", "BA"]:
        group = df[df["condition_order"] == order]
        if group.empty:
            continue

        entry: Dict[str, Any] = {"n": int(len(group))}

        for col in ["baseline_accuracy", "ai_accuracy", "baseline_avg_time_seconds", "ai_avg_time_seconds"]:
            if col in group.columns:
                vals = group[col].dropna()
                if not vals.empty:
                    entry[f"{col}_mean"] = float(vals.mean())
                    entry[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        results[order] = entry

    if scipy_stats is not None and "AB" in results and "BA" in results:
        ab = df[df["condition_order"] == "AB"]["ai_accuracy"].dropna().to_numpy()
        ba = df[df["condition_order"] == "BA"]["ai_accuracy"].dropna().to_numpy()
        if len(ab) > 1 and len(ba) > 1:
            t_stat, p_value = scipy_stats.ttest_ind(ab, ba, equal_var=False)
            results["order_effect_ai_accuracy_t"] = float(t_stat)
            results["order_effect_ai_accuracy_p"] = float(p_value)

    return results


# Builds one row per participant by merging decisions, surveys, and participant info
def _participant_level_summary(
    participants: pd.DataFrame,
    decisions: pd.DataFrame,
    surveys: pd.DataFrame,
) -> pd.DataFrame:
    if decisions.empty and surveys.empty and participants.empty:
        return pd.DataFrame()

    participant_df = participants.copy()
    if participant_df.empty:
        participant_ids = set()
        if not decisions.empty and "participant_id" in decisions.columns:
            participant_ids.update(str(x) for x in decisions["participant_id"].dropna().tolist())
        if not surveys.empty and "participant_id" in surveys.columns:
            participant_ids.update(str(x) for x in surveys["participant_id"].dropna().tolist())
        participant_df = pd.DataFrame({"participant_id": sorted(participant_ids)})

    if "completed" not in participant_df.columns:
        participant_df["completed"] = 0

    decision_summary = pd.DataFrame()
    if not decisions.empty:
        agg_map: Dict[str, Tuple[str, str]] = {}

        if "correct" in decisions.columns:
            agg_map["accuracy"] = ("correct", "mean")
        if "time_ms" in decisions.columns:
            agg_map["avg_time_seconds"] = ("time_ms", lambda s: float(s.mean() / 1000.0))
        if "ai_followed" in decisions.columns:
            agg_map["ai_followed_rate"] = ("ai_followed", "mean")
        if "ai_seen" in decisions.columns:
            agg_map["ai_seen_rate"] = ("ai_seen", "mean")
        if "ai_confidence" in decisions.columns:
            agg_map["avg_ai_confidence"] = ("ai_confidence", "mean")
        if "ai_prob_approve" in decisions.columns:
            agg_map["avg_ai_prob_approve"] = ("ai_prob_approve", "mean")

        if agg_map:
            grouped = decisions.groupby(["participant_id", "condition"]).agg(**agg_map).reset_index()
            pivoted = grouped.pivot(index="participant_id", columns="condition")
            pivoted.columns = [f"{cond}_{metric}" for metric, cond in pivoted.columns]
            decision_summary = pivoted.reset_index()

    survey_summary = pd.DataFrame()
    if not surveys.empty:
        survey_summary = surveys.groupby("participant_id", as_index=False).agg(
            trust_score=("trust_score", "mean"),
            sus_score=("sus_score", "mean"),
            comment=("comment", lambda s: " | ".join([str(x).strip() for x in s if str(x).strip()])),
        )

    merged = participant_df.copy()

    if not decision_summary.empty:
        merged = merged.merge(decision_summary, on="participant_id", how="left")

    if not survey_summary.empty:
        merged = merged.merge(survey_summary, on="participant_id", how="left")

    return merged.sort_values("participant_id").reset_index(drop=True)


# Saves a simple bar chart to disk
def _make_bar_plot(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# Saves a bar chart with rotated x-axis labels to disk
def _make_count_plot(
    labels: List[str],
    values: List[int],
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# Saves a scatter plot to disk
def _make_scatter_plot(
    x: List[float],
    y: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# Loads all data, computes statistics, generates plots, and returns a summary dict
def generate_results(static_root: str) -> Dict[str, Any]:
    participants = _read_table_as_df("participants")
    decisions = _read_table_as_df("decisions")
    events = _read_table_as_df("events")
    surveys_raw = _read_table_as_df("surveys")
    surveys = _parse_surveys_df(surveys_raw)

    results_dir = os.path.join(static_root, RESULTS_DIRNAME)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(PARTICIPANT_SUMMARY_PATH), exist_ok=True)

    if decisions.empty and surveys_raw.empty and participants.empty:
        return {
            "has_data": False,
            "message": "No results found yet. Complete at least one full run so data is saved in outputs/study.db.",
            "summary": {},
            "plots": {},
        }

    if not decisions.empty:
        for col in [
            "correct", "time_ms", "ai_followed", "ai_seen",
            "ground_truth", "ai_confidence", "ai_prob_approve",
        ]:
            if col in decisions.columns:
                decisions[col] = pd.to_numeric(decisions[col], errors="coerce")

    if not participants.empty and "completed" in participants.columns:
        participants["completed"] = pd.to_numeric(participants["completed"], errors="coerce").fillna(0).astype(int)

    participant_summary = _participant_level_summary(participants, decisions, surveys)

    if not participant_summary.empty:
        participant_summary.to_csv(PARTICIPANT_SUMMARY_PATH, index=False)

    acc_by_cond: Dict[str, float] = {}
    time_by_cond: Dict[str, float] = {}
    follow_by_cond: Dict[str, float] = {}
    trust_by_cond: Dict[str, float] = {}
    sus_by_cond: Dict[str, float] = {}
    ai_confidence_by_cond: Dict[str, float] = {}
    ai_prob_approve_by_cond: Dict[str, float] = {}
    ai_seen_by_cond: Dict[str, float] = {}

    if not participant_summary.empty:
        if "baseline_accuracy" in participant_summary.columns and participant_summary["baseline_accuracy"].notna().any():
            acc_by_cond["baseline"] = float(participant_summary["baseline_accuracy"].mean())
        if "ai_accuracy" in participant_summary.columns and participant_summary["ai_accuracy"].notna().any():
            acc_by_cond["ai"] = float(participant_summary["ai_accuracy"].mean())

        if "baseline_avg_time_seconds" in participant_summary.columns and participant_summary["baseline_avg_time_seconds"].notna().any():
            time_by_cond["baseline"] = float(participant_summary["baseline_avg_time_seconds"].mean())
        if "ai_avg_time_seconds" in participant_summary.columns and participant_summary["ai_avg_time_seconds"].notna().any():
            time_by_cond["ai"] = float(participant_summary["ai_avg_time_seconds"].mean())

        if "ai_ai_followed_rate" in participant_summary.columns and participant_summary["ai_ai_followed_rate"].notna().any():
            follow_by_cond["ai"] = float(participant_summary["ai_ai_followed_rate"].mean())

        if "trust_score" in participant_summary.columns and participant_summary["trust_score"].notna().any():
            trust_by_cond["ai"] = float(participant_summary["trust_score"].mean())

        if "sus_score" in participant_summary.columns and participant_summary["sus_score"].notna().any():
            sus_by_cond["ai"] = float(participant_summary["sus_score"].mean())

        if "ai_avg_ai_confidence" in participant_summary.columns and participant_summary["ai_avg_ai_confidence"].notna().any():
            ai_confidence_by_cond["ai"] = float(participant_summary["ai_avg_ai_confidence"].mean())

        if "ai_avg_ai_prob_approve" in participant_summary.columns and participant_summary["ai_avg_ai_prob_approve"].notna().any():
            ai_prob_approve_by_cond["ai"] = float(participant_summary["ai_avg_ai_prob_approve"].mean())

        if "ai_ai_seen_rate" in participant_summary.columns and participant_summary["ai_ai_seen_rate"].notna().any():
            ai_seen_by_cond["ai"] = float(participant_summary["ai_ai_seen_rate"].mean())
        if "baseline_ai_seen_rate" in participant_summary.columns and participant_summary["baseline_ai_seen_rate"].notna().any():
            ai_seen_by_cond["baseline"] = float(participant_summary["baseline_ai_seen_rate"].mean())

    comments: List[Dict[str, str]] = []
    if isinstance(participant_summary, pd.DataFrame) and (not participant_summary.empty) and "comment" in participant_summary.columns:
        for _, row in participant_summary.iterrows():
            c = str(row.get("comment", "")).strip()
            if c:
                comments.append({
                    "participant_id": str(row.get("participant_id", "")).strip(),
                    "condition": "ai",
                    "comment": c,
                })

    accuracy_groups = _accuracy_improvement_groups(participant_summary)
    spearman_trust_followed = _spearman_trust_vs_ai_followed(participant_summary)
    ai_followed_dist = _ai_followed_distribution(participant_summary)
    counterbalance_stats = _counterbalance_subgroup_stats(participant_summary)

    cond_order = ["baseline", "ai"]

    # Returns labels and values in the correct condition order for plotting
    def ordered_values(d: Dict[str, float]) -> Tuple[List[str], List[float]]:
        labels = [c for c in cond_order if c in d]
        vals = [d[c] for c in labels]
        return labels, vals

    plots: Dict[str, str] = {}

    if acc_by_cond:
        labels, vals = ordered_values(acc_by_cond)
        out = os.path.join(results_dir, "accuracy.png")
        _make_bar_plot(labels, vals, "Accuracy by condition", "Accuracy (0–1)", out)
        plots["accuracy"] = f"/static/{RESULTS_DIRNAME}/accuracy.png"

    if time_by_cond:
        labels, vals = ordered_values(time_by_cond)
        out = os.path.join(results_dir, "time.png")
        _make_bar_plot(labels, vals, "Average decision time by condition", "Seconds", out)
        plots["time"] = f"/static/{RESULTS_DIRNAME}/time.png"

    if trust_by_cond:
        labels, vals = ordered_values(trust_by_cond)
        out = os.path.join(results_dir, "trust.png")
        _make_bar_plot(labels, vals, "Trust score by condition", "Average (1–5)", out)
        plots["trust"] = f"/static/{RESULTS_DIRNAME}/trust.png"

    if sus_by_cond:
        labels, vals = ordered_values(sus_by_cond)
        out = os.path.join(results_dir, "sus.png")
        _make_bar_plot(labels, vals, "SUS score by condition", "SUS (0–100)", out)
        plots["sus"] = f"/static/{RESULTS_DIRNAME}/sus.png"

    if follow_by_cond:
        labels, vals = ordered_values(follow_by_cond)
        out = os.path.join(results_dir, "ai_followed.png")
        _make_bar_plot(labels, vals, "AI-followed rate (AI condition)", "Rate (0–1)", out)
        plots["ai_followed"] = f"/static/{RESULTS_DIRNAME}/ai_followed.png"

    if ai_confidence_by_cond:
        labels, vals = ordered_values(ai_confidence_by_cond)
        out = os.path.join(results_dir, "ai_confidence.png")
        _make_bar_plot(labels, vals, "Average AI confidence", "Confidence (0–1)", out)
        plots["ai_confidence"] = f"/static/{RESULTS_DIRNAME}/ai_confidence.png"

    if ai_prob_approve_by_cond:
        labels, vals = ordered_values(ai_prob_approve_by_cond)
        out = os.path.join(results_dir, "ai_prob_approve.png")
        _make_bar_plot(labels, vals, "Average AI approval probability", "Probability (0–1)", out)
        plots["ai_prob_approve"] = f"/static/{RESULTS_DIRNAME}/ai_prob_approve.png"

    if ai_seen_by_cond:
        labels, vals = ordered_values(ai_seen_by_cond)
        out = os.path.join(results_dir, "ai_seen_rate.png")
        _make_bar_plot(labels, vals, "AI seen rate by condition", "Rate (0–1)", out)
        plots["ai_seen_rate"] = f"/static/{RESULTS_DIRNAME}/ai_seen_rate.png"

    if not participant_summary.empty:
        needed = ["trust_score", "ai_ai_followed_rate"]
        if all(c in participant_summary.columns for c in needed):
            scatter_df = participant_summary[needed].dropna()
            if len(scatter_df) >= 3:
                out = os.path.join(results_dir, "trust_vs_ai_followed.png")
                _make_scatter_plot(
                    scatter_df["trust_score"].tolist(),
                    scatter_df["ai_ai_followed_rate"].tolist(),
                    "Trust score vs AI-followed rate",
                    "Trust score (1–5)",
                    "AI-followed rate (0–1)",
                    out,
                )
                plots["trust_vs_ai_followed"] = f"/static/{RESULTS_DIRNAME}/trust_vs_ai_followed.png"

    if not participant_summary.empty:
        needed = ["baseline_accuracy", "ai_accuracy"]
        if all(c in participant_summary.columns for c in needed):
            pair_df = participant_summary[needed].dropna().copy()
            pair_df["diff"] = pair_df["ai_accuracy"] - pair_df["baseline_accuracy"]
            improved = pair_df[pair_df["diff"] > 0]["baseline_accuracy"].tolist()
            worsened = pair_df[pair_df["diff"] < 0]["baseline_accuracy"].tolist()
            if improved or worsened:
                plt.figure()
                if improved:
                    plt.hist(improved, alpha=0.6, label="Improved", bins=10)
                if worsened:
                    plt.hist(worsened, alpha=0.6, label="Worsened", bins=10)
                plt.title("Baseline accuracy: improved vs worsened")
                plt.xlabel("Baseline accuracy")
                plt.ylabel("Count")
                plt.legend()
                plt.tight_layout()
                out = os.path.join(results_dir, "accuracy_groups.png")
                plt.savefig(out, dpi=160)
                plt.close()
                plots["accuracy_groups"] = f"/static/{RESULTS_DIRNAME}/accuracy_groups.png"

    if counterbalance_stats and len(counterbalance_stats) >= 2:
        orders = [o for o in ["AB", "BA"] if o in counterbalance_stats]
        ai_acc_vals = [counterbalance_stats[o].get("ai_accuracy_mean") for o in orders if counterbalance_stats[o].get("ai_accuracy_mean") is not None]
        if len(orders) == len(ai_acc_vals) and ai_acc_vals:
            out = os.path.join(results_dir, "counterbalance_accuracy.png")
            _make_bar_plot(orders, ai_acc_vals, "AI accuracy by condition order", "Accuracy (0–1)", out)
            plots["counterbalance_accuracy"] = f"/static/{RESULTS_DIRNAME}/counterbalance_accuracy.png"

    if not participants.empty and "age_group" in participants.columns:
        counts = participants["age_group"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "age_group_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "Participant age group distribution", "Count", out)
            plots["age_group_distribution"] = f"/static/{RESULTS_DIRNAME}/age_group_distribution.png"

    if not participants.empty and "background" in participants.columns:
        counts = participants["background"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "background_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "Participant background distribution", "Count", out)
            plots["background_distribution"] = f"/static/{RESULTS_DIRNAME}/background_distribution.png"

    if not participants.empty and "ai_familiarity" in participants.columns:
        counts = participants["ai_familiarity"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "ai_familiarity_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "AI familiarity distribution", "Count", out)
            plots["ai_familiarity_distribution"] = f"/static/{RESULTS_DIRNAME}/ai_familiarity_distribution.png"

    if not participants.empty and "finance_familiarity" in participants.columns:
        counts = participants["finance_familiarity"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "finance_familiarity_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "Finance familiarity distribution", "Count", out)
            plots["finance_familiarity_distribution"] = f"/static/{RESULTS_DIRNAME}/finance_familiarity_distribution.png"

    if not participants.empty and "condition_order" in participants.columns:
        counts = participants["condition_order"].fillna("").astype(str).str.strip()
        counts = counts[counts != ""].value_counts()
        if not counts.empty:
            out = os.path.join(results_dir, "condition_order_distribution.png")
            _make_count_plot(counts.index.tolist(), counts.astype(int).tolist(), "Condition order distribution", "Count", out)
            plots["condition_order_distribution"] = f"/static/{RESULTS_DIRNAME}/condition_order_distribution.png"

    paired_tests = {}
    if not participant_summary.empty:
        paired_tests["accuracy"] = _paired_stats(participant_summary, "baseline_accuracy", "ai_accuracy")
        paired_tests["decision_time_seconds"] = _paired_stats(
            participant_summary,
            "baseline_avg_time_seconds",
            "ai_avg_time_seconds",
        )

    participant_summary_preview: List[Dict[str, Any]] = []
    if not participant_summary.empty:
        preview_df = participant_summary.copy()
        for col in preview_df.columns:
            if pd.api.types.is_float_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].round(3)
        participant_summary_preview = preview_df.to_dict(orient="records")

    summary = {
        "accuracy_by_condition": acc_by_cond,
        "time_seconds_by_condition": time_by_cond,
        "trust_by_condition": trust_by_cond,
        "sus_by_condition": sus_by_cond,
        "ai_followed_rate": follow_by_cond,
        "ai_confidence_by_condition": ai_confidence_by_cond,
        "ai_prob_approve_by_condition": ai_prob_approve_by_cond,
        "ai_seen_rate_by_condition": ai_seen_by_cond,
        "paired_tests": paired_tests,
        "accuracy_improvement_groups": accuracy_groups,
        "spearman_trust_vs_ai_followed": spearman_trust_followed,
        "ai_followed_distribution": ai_followed_dist,
        "counterbalance_subgroup_stats": counterbalance_stats,
        "comments": comments,
        "participant_summary_preview": participant_summary_preview,
        "n_participants": int(len(participants)) if not participants.empty else 0,
        "n_completed_participants": int(participants["completed"].sum()) if (not participants.empty and "completed" in participants.columns) else 0,
        "n_decisions": int(len(decisions)),
        "n_surveys": int(len(surveys_raw)),
        "n_events": int(len(events)),
    }

    return {
        "has_data": True,
        "message": "",
        "summary": summary,
        "plots": plots,
    }
