# app/ai.py

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from app.config import MODEL_PATH
from app.explanations import build_reason

_MODEL = None

DISPLAY_NAMES = {
    "person_age": "Applicant age",
    "person_education": "Education level",
    "person_income": "Annual income",
    "person_emp_exp": "Years of work experience",
    "person_home_ownership": "Housing situation",
    "loan_amnt": "Requested loan amount",
    "loan_intent": "Purpose of the loan",
    "loan_int_rate": "Loan interest rate",
    "credit_score": "Credit score",
    "previous_loan_defaults_on_file": "Previous loan repayment problems",
}


def _load_model():
    global _MODEL

    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Run python -m app.model_train first."
            )

        _MODEL = joblib.load(MODEL_PATH)

    return _MODEL


def _get_pipeline_parts(model) -> Tuple[Optional[Any], Optional[Any]]:
    preprocess = None
    clf = None

    if hasattr(model, "named_steps"):
        preprocess = model.named_steps.get("preprocess")
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier")
    else:
        clf = model

    return preprocess, clf


def _get_missing_original_features(X_dict: Dict[str, Any]) -> set:
    missing = set()

    for key, val in X_dict.items():
        if val is None:
            missing.add(key.lower())
            continue

        text = str(val).strip().lower()

        if text in ("unknown", "not available", "nan", ""):
            missing.add(key.lower())

    return missing


def _base_feature_name(transformed_name: str) -> str:
    clean = (
        transformed_name.split("__", 1)[1]
        if "__" in transformed_name
        else transformed_name
    )

    for original in DISPLAY_NAMES.keys():
        if clean == original or clean.startswith(original + "_"):
            return original

    return clean


def _aggregate_shap_by_feature(
    feat_names: List[str],
    values: np.ndarray,
    missing_features: set,
) -> List[Tuple[str, float]]:
    aggregated: Dict[str, float] = defaultdict(float)

    for fname, val in zip(feat_names, values.tolist()):
        val = float(val)

        if abs(val) < 1e-9:
            continue

        if "unknown" in fname.lower():
            continue

        base = _base_feature_name(fname).strip().lower()

        if base in missing_features:
            continue

        aggregated[base] += val

    return sorted(
        aggregated.items(),
        key=lambda fv: abs(fv[1]),
        reverse=True,
    )


def _compute_shap_contributions(
    model,
    X_dict: Dict[str, Any],
) -> List[Tuple[str, float]]:
    try:
        import pandas as pd
        import shap

        preprocess, clf = _get_pipeline_parts(model)

        if clf is None:
            return []

        X_df = pd.DataFrame([X_dict])

        if preprocess is not None:
            X_trans = preprocess.transform(X_df)

            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()

            X_trans = np.asarray(X_trans)

            if hasattr(preprocess, "get_feature_names_out"):
                feat_names = list(preprocess.get_feature_names_out())
            else:
                feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
        else:
            X_trans = X_df.values
            feat_names = list(X_df.columns)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            values = np.asarray(shap_values[1][0])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            values = np.asarray(shap_values[0, :, 1])
        else:
            values = np.asarray(shap_values).reshape(-1)

        missing_features = _get_missing_original_features(X_dict)

        return _aggregate_shap_by_feature(
            feat_names=feat_names,
            values=values,
            missing_features=missing_features,
        )

    except ImportError:
        return []

    except Exception as e:
        print(f"SHAP error: {e}")
        return []


def _weight_label(score: float) -> str:
    score = abs(score)

    if score >= 0.15:
        return "high"

    if score >= 0.06:
        return "medium"

    return "low"


def _confidence_label(confidence: float) -> str:
    if confidence < 0.65:
        return "Low"

    if confidence < 0.80:
        return "Moderate"

    return "High"


def _make_explanation_item(
    feature: str,
    contribution: float,
    recommendation: str,
    features: Dict[str, Any],
) -> Dict[str, str]:
    return {
        "factor": DISPLAY_NAMES.get(
            feature,
            feature.replace("_", " ").title(),
        ),
        "text": build_reason(
            feature=feature,
            value=features.get(feature),
            recommendation=recommendation,
            contribution=contribution,
        ),
        "weight": _weight_label(contribution),
    }


def _build_explanation(
    contribs: List[Tuple[str, float]],
    recommendation: str,
    features: Dict[str, Any],
    max_items: int = 3,
    min_shap_threshold: float = 0.01,
) -> List[Dict[str, str]]:
    if not contribs:
        return []

    contribs = [
        (f, c)
        for (f, c) in contribs
        if abs(c) >= min_shap_threshold
    ]

    if not contribs:
        return []

    selected = sorted(
        contribs,
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:max_items]

    return [
        _make_explanation_item(
            feature=f,
            contribution=c,
            recommendation=recommendation,
            features=features,
        )
        for f, c in selected
    ]


def get_ai_advice(
    features: Dict[str, Any],
    approval_threshold: float = 0.50,
) -> Dict[str, Any]:
    model = _load_model()

    import pandas as pd

    X_df = pd.DataFrame([features])
    probas = model.predict_proba(X_df)[0]

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        _, clf = _get_pipeline_parts(model)
        classes = getattr(clf, "classes_", None)

    if classes is not None and 1 in list(classes):
        idx_approve = int(
            np.where(np.asarray(classes) == 1)[0][0]
        )
        prob_approve = float(probas[idx_approve])
    else:
        prob_approve = float(probas[1])

    recommendation = (
        "Approve"
        if prob_approve >= approval_threshold
        else "Reject"
    )

    confidence = (
        prob_approve
        if recommendation == "Approve"
        else 1.0 - prob_approve
    )

    confidence = float(confidence)
    confidence_label = _confidence_label(confidence)

    contribs = _compute_shap_contributions(
        model=model,
        X_dict=features,
    )

    explanation = _build_explanation(
        contribs=contribs,
        recommendation=recommendation,
        features=features,
        max_items=3,
    )

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 3),
        "confidence_label": confidence_label,
        "prob_approve": round(prob_approve, 3),
        "explanation": explanation,
        "supporting_explanation": explanation,
        "opposing_explanation": [],
    }