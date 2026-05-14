from __future__ import annotations

import os
from typing import List, Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app.config import (
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    MODEL_PATH,
    TARGET_COL,
    DROP_COLS_FOR_UI,
)

EXCLUDE_COLS = set([
    "Unnamed: 0",
    "case_id",
    TARGET_COL,
] + DROP_COLS_FOR_UI)

RANDOM_STATE = 42


def _load_training_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    elif os.path.exists(CASES_FOR_STUDY_PATH):
        df = pd.read_csv(CASES_FOR_STUDY_PATH)
    else:
        raise FileNotFoundError("Could not find a dataset to train on.")

    if "loan_status" in df.columns and TARGET_COL not in df.columns:
        df[TARGET_COL] = df["loan_status"].astype(int)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' could not be created or found.")

    return df


def _clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    for col in df.columns:
        if col == TARGET_COL:
            continue

        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    numeric_cols = [c for c in feature_cols if df[c].dtype.kind in "biufc"]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return numeric_cols, categorical_cols


def main() -> None:
    df = _load_training_data()
    df = _clean_training_data(df)

    numeric_cols, categorical_cols = _split_columns(df)

    print("Training columns:")
    print(f"  Numeric: {numeric_cols}")
    print(f"  Categorical: {categorical_cols}")

    X = df[numeric_cols + categorical_cols]
    y = df[TARGET_COL].astype(int)

    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

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

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    try:
        import shap
        print("SHAP is available.")
    except ImportError:
        print("SHAP not available. Install it with: pip install shap")


if __name__ == "__main__":
    main()