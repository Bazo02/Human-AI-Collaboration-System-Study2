# app/data_prep.py

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

from app.config import (
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    TARGET_COL,
    DROP_COLS_FOR_UI,
)

STUDY_SET_SIZE = 120
BORDERLINE_SHARE = 0.50  


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    
    if "Risk" in df.columns:
        df[TARGET_COL] = (df["Risk"] == "good").astype(int)
        df = df.drop(columns=["Risk"])

    df = df.drop_duplicates()

    if "case_id" not in df.columns:
        df.insert(0, "case_id", range(1, len(df) + 1))

    for col in df.columns:
        if col in (TARGET_COL, "case_id"):
            continue
        if df[col].dtype.kind in "biufc":
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna("Unknown")

    return df


def _heuristic_risk_score(df: pd.DataFrame) -> pd.Series:
    # German Credit risk score: higher = higher risk (more likely bad/reject)
    def col_or_zero(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([0] * len(df), index=df.index)

    duration = col_or_zero("Duration")
    credit_amount = col_or_zero("Credit amount")
    age = col_or_zero("Age")

    duration_risk = (duration - duration.min()) / (duration.max() - duration.min() + 1e-9)
    amount_risk = (credit_amount - credit_amount.min()) / (credit_amount.max() - credit_amount.min() + 1e-9)
    age_risk = 1 - (age - age.min()) / (age.max() - age.min() + 1e-9)

    score = 0.40 * duration_risk + 0.40 * amount_risk + 0.20 * age_risk
    return score


def _select_study_cases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_score"] = _heuristic_risk_score(df)

    df_approve = df[df[TARGET_COL] == 1].copy()
    df_reject = df[df[TARGET_COL] == 0].copy()

    half = STUDY_SET_SIZE // 2
    n_approve = min(half, len(df_approve))
    n_reject = min(STUDY_SET_SIZE - n_approve, len(df_reject))

    n_borderline_total = int(STUDY_SET_SIZE * BORDERLINE_SHARE)
    n_borderline_each = n_borderline_total // 2

    approve_borderline = df_approve.sort_values("risk_score", ascending=False).head(n_borderline_each)
    approve_easy = df_approve.sort_values("risk_score", ascending=True).head(max(0, n_approve - len(approve_borderline)))

    reject_borderline = df_reject.sort_values("risk_score", ascending=True).head(n_borderline_each)
    reject_easy = df_reject.sort_values("risk_score", ascending=False).head(max(0, n_reject - len(reject_borderline)))

    selected = pd.concat([approve_borderline, approve_easy, reject_borderline, reject_easy], axis=0)

    if len(selected) < STUDY_SET_SIZE:
        missing = STUDY_SET_SIZE - len(selected)
        remaining = df.drop(index=selected.index, errors="ignore")
        if missing > 0 and len(remaining) > 0:
            fill = remaining.sample(n=min(missing, len(remaining)), random_state=42)
            selected = pd.concat([selected, fill], axis=0)

    selected = selected.sample(frac=1.0, random_state=123).reset_index(drop=True)
    selected = selected.drop(columns=["risk_score"], errors="ignore")

    return selected


def _drop_sensitive_and_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DROP_COLS_FOR_UI:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = _basic_clean(df)
    df = _drop_sensitive_and_unused_cols(df)
    study_df = _select_study_cases(df)

    os.makedirs(os.path.dirname(CASES_FOR_STUDY_PATH), exist_ok=True)
    study_df.to_csv(CASES_FOR_STUDY_PATH, index=False)

    print("Done.")
    print(f"Saved study cases to: {CASES_FOR_STUDY_PATH}")
    print(f"Rows: {len(study_df)}")
    print(f"Approve rate: {study_df[TARGET_COL].mean():.3f}")


if __name__ == "__main__":
    main()