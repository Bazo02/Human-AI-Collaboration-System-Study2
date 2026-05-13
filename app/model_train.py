# app/model_train.py

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from app.config import (
    DATA_PATH,
    CASES_FOR_STUDY_PATH,
    MODEL_PATH,
    TARGET_COL,
)

EXCLUDE_COLS = [
    "Unnamed: 0",
    "case_id",
    "Sex",
    "Risk",
]


def _load_training_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Map Risk -> loan_approved if not already done
        if "Risk" in df.columns and TARGET_COL not in df.columns:
            df[TARGET_COL] = (df["Risk"] == "good").astype(int)
        return df
    if os.path.exists(CASES_FOR_STUDY_PATH):
        return pd.read_csv(CASES_FOR_STUDY_PATH)
    raise FileNotFoundError("Could not find a dataset to train on.")


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != TARGET_COL and c not in EXCLUDE_COLS]
    numeric_cols = [c for c in feature_cols if df[c].dtype.kind in "biufc"]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def main() -> None:
    df = _load_training_data().copy()
    df = df.drop_duplicates()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    for col in df.columns:
        if col == TARGET_COL:
            continue
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    for col in EXCLUDE_COLS:
        if col in X.columns:
            X = X.drop(columns=[col])

    numeric_cols, categorical_cols = _split_columns(df)
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    print("Training columns:")
    print("  Numeric:", numeric_cols)
    print("  Categorical:", categorical_cols)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nModel evaluation:")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    try:
        import shap  # noqa: F401
        print("\nSHAP is available.")
    except ImportError:
        print("\nWARNING: shap is not installed. Run: pip install shap")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()