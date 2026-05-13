# app/ai.py

from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import joblib

from app.config import MODEL_PATH


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Make sure model.joblib is included in the deployed repo."
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def _prettify_feature_name(raw_name: str) -> str:
    name = raw_name

    if "__" in name:
        name = name.split("__", 1)[1]

    if "_" in name:
        parts = name.split("_", 1)
        col, rest = parts[0], parts[1]
        if "cat__" in raw_name:
            return f"{col.replace('_', ' ')}: {rest.replace('_', ' ')}"

    return name.replace("_", " ")


def _get_pipeline_parts(model) -> Tuple[Optional[Any], Optional[Any]]:
    preprocess = None
    clf = None

    if hasattr(model, "named_steps"):
        preprocess = model.named_steps.get("preprocess")
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier")
    else:
        clf = model

    return preprocess, clf


def _compute_shap_contributions(model, X_dict: Dict[str, Any]) -> List[Tuple[str, float]]:
    try:
        import shap
        import pandas as pd

        preprocess, clf = _get_pipeline_parts(model)
        if clf is None:
            return []

        X_df = pd.DataFrame([X_dict])

        if preprocess is not None:
            X_trans = preprocess.transform(X_df)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            X_trans = np.array(X_trans)

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
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        pairs = list(zip(feat_names, values.tolist()))
        pairs = [(f, v) for (f, v) in pairs if abs(v) > 1e-9]
        pairs.sort(key=lambda fv: abs(fv[1]), reverse=True)
        return pairs

    except ImportError:
        return _compute_importance_contributions(model, X_dict)
    except Exception:
        return []


def _compute_importance_contributions(model, X_dict: Dict[str, Any]) -> List[Tuple[str, float]]:
    try:
        import pandas as pd

        preprocess, clf = _get_pipeline_parts(model)
        if clf is None or not hasattr(clf, "feature_importances_"):
            return []

        X_df = pd.DataFrame([X_dict])

        if preprocess is not None:
            X_trans = preprocess.transform(X_df)
            if hasattr(preprocess, "get_feature_names_out"):
                feat_names = list(preprocess.get_feature_names_out())
            else:
                feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
        else:
            feat_names = list(X_df.columns)

        importances = clf.feature_importances_
        pairs = list(zip(feat_names, importances.tolist()))
        pairs.sort(key=lambda fv: abs(fv[1]), reverse=True)
        return pairs

    except Exception:
        return []


def _build_explanation(
    contribs: List[Tuple[str, float]],
    recommendation: str,
    max_items: int = 3,
) -> List[str]:
    reasons: List[str] = []
    if not contribs:
        return reasons

    if recommendation == "Approve":
        filtered = [(f, c) for (f, c) in contribs if c > 0]
        label = "supported approval"
    else:
        filtered = [(f, c) for (f, c) in contribs if c < 0]
        label = "increased risk"

    if len(filtered) < max_items:
        filtered = contribs

    for f, c in filtered[:max_items]:
        nice = _prettify_feature_name(str(f))
        if recommendation == "Approve":
            reasons.append(f"{nice} {label}" if c >= 0 else f"{nice} slightly reduced approval support")
        else:
            reasons.append(f"{nice} {label}" if c <= 0 else f"{nice} slightly reduced risk")

    return reasons[:max_items]


def get_ai_advice(features: Dict[str, Any], approval_threshold: float = 0.65) -> Dict[str, Any]:
    model = _load_model()

    import pandas as pd
    X_df = pd.DataFrame([features])

    probas = model.predict_proba(X_df)[0]
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier")
        classes = getattr(clf, "classes_", None)

    if classes is not None:
        idx_approve = int(np.where(np.array(classes) == 1)[0][0])
        prob_approve = float(probas[idx_approve])
    else:
        prob_approve = float(probas[1])

    recommendation = "Approve" if prob_approve >= approval_threshold else "Reject"
    confidence = prob_approve if recommendation == "Approve" else (1.0 - prob_approve)

    contribs = _compute_shap_contributions(model, features)
    explanation = _build_explanation(contribs, recommendation=recommendation, max_items=3)

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 3),
        "prob_approve": round(prob_approve, 3),
        "explanation": explanation,
    }