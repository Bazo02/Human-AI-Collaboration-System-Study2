# app/data_prep.py

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd

from app.config import (
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    TARGET_COL,
    DROP_COLS_FOR_UI,
)

STUDY_SET_SIZE = 120
BORDERLINE_PROB_LOW = 0.30
BORDERLINE_PROB_HIGH = 0.70
RANDOM_STATE = 42


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "loan_status" in df.columns and TARGET_COL not in df.columns:
        df[TARGET_COL] = df["loan_status"].astype(int)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' could not be created or found.")

    df = df.drop_duplicates().reset_index(drop=True)

    if "case_id" not in df.columns:
        df.insert(0, "case_id", range(1, len(df) + 1))

    for col in df.columns:
        if col in (TARGET_COL, "case_id"):
            continue

        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = set(["case_id", TARGET_COL] + DROP_COLS_FOR_UI)
    feature_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = [c for c in feature_cols if df[c].dtype.kind in "biufc"]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return numeric_cols, categorical_cols


def _select_study_cases(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    numeric_cols, categorical_cols = _split_columns(df)

    X = df[numeric_cols + categorical_cols]
    y = df[TARGET_COL].astype(int)

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf),
    ])

    model.fit(X, y)

    scored = df.copy()
    scored["_prob"] = model.predict_proba(X)[:, 1]

    borderline = scored[
        (scored["_prob"] >= BORDERLINE_PROB_LOW)
        & (scored["_prob"] <= BORDERLINE_PROB_HIGH)
    ].copy()

    approved = borderline[borderline[TARGET_COL] == 1]
    rejected = borderline[borderline[TARGET_COL] == 0]

    half = STUDY_SET_SIZE // 2
    n_approve = min(half, len(approved))
    n_reject = min(half, len(rejected))

    selected_parts = []

    if n_approve > 0:
        selected_parts.append(approved.sample(n=n_approve, random_state=RANDOM_STATE))

    if n_reject > 0:
        selected_parts.append(rejected.sample(n=n_reject, random_state=RANDOM_STATE))

    selected = pd.concat(selected_parts, axis=0) if selected_parts else pd.DataFrame()

    remaining = STUDY_SET_SIZE - len(selected)

    if remaining > 0:
        selected_ids = set(selected["case_id"].tolist()) if not selected.empty else set()
        fallback = scored[~scored["case_id"].isin(selected_ids)].copy()
        fallback["_distance_to_boundary"] = (fallback["_prob"] - 0.5).abs()
        fallback = fallback.sort_values("_distance_to_boundary").head(remaining)
        selected = pd.concat([selected, fallback], axis=0)

    selected = selected.sample(frac=1.0, random_state=123).reset_index(drop=True)

    print(f"Study set: {len(selected)} cases")
    print(f"Approved: {int(selected[TARGET_COL].sum())}, Rejected: {int((selected[TARGET_COL] == 0).sum())}")
    print(f"Mean prob: {selected['_prob'].mean():.3f}")

    selected = selected.drop(columns=["_prob", "_distance_to_boundary"], errors="ignore")

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
    study_df = _select_study_cases(df)
    study_df = _drop_sensitive_and_unused_cols(study_df)

    os.makedirs(os.path.dirname(CASES_FOR_STUDY_PATH), exist_ok=True)
    study_df.to_csv(CASES_FOR_STUDY_PATH, index=False)

    print(f"Saved study cases to: {CASES_FOR_STUDY_PATH}")


if __name__ == "__main__":
    main()