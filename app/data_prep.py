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
N_BORDERLINE = 40
N_CLEAR_APPROVE = 40
N_CLEAR_REJECT = 40

RANDOM_STATE = 42


# Removes duplicates, fills missing values, and adds a case_id column
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


# Splits columns into numeric and categorical lists for the model
def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = set(["case_id", TARGET_COL] + DROP_COLS_FOR_UI)

    feature_cols = [
        c for c in df.columns
        if c not in exclude
    ]

    numeric_cols = [
        c for c in feature_cols
        if df[c].dtype.kind in "biufc"
    ]

    categorical_cols = [
        c for c in feature_cols
        if c not in numeric_cols
    ]

    return numeric_cols, categorical_cols


# Trains a quick model and adds approval probability scores to each case
def _score_cases(df: pd.DataFrame) -> pd.DataFrame:
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
    scored["_distance_to_boundary"] = (scored["_prob"] - 0.5).abs()

    return scored


# Returns n cases from df sorted by sort_col, excluding already selected IDs
def _take_cases(
    df: pd.DataFrame,
    n: int,
    selected_ids: set,
    sort_col: str,
    ascending: bool,
) -> pd.DataFrame:
    available = df[~df["case_id"].isin(selected_ids)].copy()

    if available.empty or n <= 0:
        return pd.DataFrame()

    return available.sort_values(sort_col, ascending=ascending).head(n)


# Selects a balanced set of borderline, clear-approve, and clear-reject cases
def _select_study_cases(df: pd.DataFrame) -> pd.DataFrame:
    scored = _score_cases(df)

    selected_parts = []
    selected_ids = set()

    borderline = scored.sort_values(
        "_distance_to_boundary",
        ascending=True,
    ).head(N_BORDERLINE)

    selected_parts.append(borderline)
    selected_ids.update(borderline["case_id"].tolist())

    clear_approve_pool = scored[
        (scored[TARGET_COL] == 1)
        & (~scored["case_id"].isin(selected_ids))
    ].copy()

    clear_approve = _take_cases(
        df=clear_approve_pool,
        n=N_CLEAR_APPROVE,
        selected_ids=selected_ids,
        sort_col="_prob",
        ascending=False,
    )

    selected_parts.append(clear_approve)
    selected_ids.update(clear_approve["case_id"].tolist())

    clear_reject_pool = scored[
        (scored[TARGET_COL] == 0)
        & (~scored["case_id"].isin(selected_ids))
    ].copy()

    clear_reject = _take_cases(
        df=clear_reject_pool,
        n=N_CLEAR_REJECT,
        selected_ids=selected_ids,
        sort_col="_prob",
        ascending=True,
    )

    selected_parts.append(clear_reject)
    selected_ids.update(clear_reject["case_id"].tolist())

    selected = pd.concat(selected_parts, axis=0)

    if len(selected) < STUDY_SET_SIZE:
        missing = STUDY_SET_SIZE - len(selected)

        remaining = scored[
            ~scored["case_id"].isin(selected_ids)
        ].copy()

        fill = remaining.sort_values(
            "_distance_to_boundary",
            ascending=True,
        ).head(missing)

        selected = pd.concat([selected, fill], axis=0)

    selected = selected.head(STUDY_SET_SIZE)
    selected = selected.sample(frac=1.0, random_state=123).reset_index(drop=True)

    print(f"Study set: {len(selected)} cases")
    print(f"Approved: {int(selected[TARGET_COL].sum())}")
    print(f"Rejected: {int((selected[TARGET_COL] == 0).sum())}")
    print(f"Mean prob: {selected['_prob'].mean():.3f}")
    print(f"Min prob: {selected['_prob'].min():.3f}")
    print(f"Max prob: {selected['_prob'].max():.3f}")

    selected = selected.drop(
        columns=["_prob", "_distance_to_boundary"],
        errors="ignore",
    )

    return selected


# Drops columns that should not be shown in the UI
def _drop_sensitive_and_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DROP_COLS_FOR_UI:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


# Runs the full data preparation pipeline and saves the result to CSV
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
