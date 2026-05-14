# app/ai.py

from __future__ import annotations

import os
from collections import defaultdict
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
    """Returns the set of original feature names where the value is missing/unknown."""
    missing = set()
    for key, val in X_dict.items():
        if val is None or str(val).strip().lower() in ("unknown", "not available", "nan", ""):
            missing.add(key.lower())
    return missing


def _aggregate_shap_by_feature(
    feat_names: List[str],
    values: np.ndarray,
    missing_features: set,
) -> List[Tuple[str, float]]:
    aggregated: Dict[str, float] = defaultdict(float)

    for fname, val in zip(feat_names, values.tolist()):
        if abs(val) < 1e-9:
            continue

        
        if "unknown" in fname.lower():
            continue

        
        clean = fname.split("__", 1)[1] if "__" in fname else fname

        
        base = clean.rsplit("_", 1)[0].strip().lower() if "_" in clean else clean.strip().lower()

        
        if base in missing_features:
            continue

        aggregated[clean] += val

    return sorted(aggregated.items(), key=lambda fv: abs(fv[1]), reverse=True)


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
            feat_names = list(preprocess.get_feature_names_out()) if hasattr(preprocess, "get_feature_names_out") else [f"f{i}" for i in range(X_trans.shape[1])]
        else:
            X_trans = X_df.values
            feat_names = list(X_df.columns)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            values = shap_values[0, :, 1]
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            values = shap_values[1][0]
        else:
            values = np.array(shap_values).ravel()

        missing_features = _get_missing_original_features(X_dict)
        return _aggregate_shap_by_feature(feat_names, values, missing_features)

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
            feat_names = list(preprocess.get_feature_names_out()) if hasattr(preprocess, "get_feature_names_out") else [f"f{i}" for i in range(X_trans.shape[1])]
        else:
            feat_names = list(X_df.columns)

        pairs = list(zip(feat_names, clf.feature_importances_.tolist()))
        pairs.sort(key=lambda fv: abs(fv[1]), reverse=True)
        return pairs

    except Exception:
        return []


def _prettify_name(name: str) -> str:
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and not parts[1].isdigit():
        return parts[0].strip()
    return name.strip()


def _build_explanation(
    contribs: List[Tuple[str, float]],
    recommendation: str,
    max_items: int = 3,
    min_shap_threshold: float = 0.01,
) -> List[str]:
    reasons: List[str] = []
    if not contribs:
        return reasons

    # Filter out very weak contributions
    contribs = [(f, c) for (f, c) in contribs if abs(c) >= min_shap_threshold]
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
        nice = _prettify_name(f)
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