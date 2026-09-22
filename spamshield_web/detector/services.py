"""Inference and explainability service layer for SpamShield."""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Any
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

_CACHED_PIPELINE = None
_CACHED_METADATA = None
_CACHED_FEATURE_NAMES = None


def get_model_pipeline():
    """Load and cache the trained Scikit-Learn pipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run training first.")
        _CACHED_PIPELINE = joblib.load(MODEL_PATH)
    return _CACHED_PIPELINE


def get_model_metadata() -> dict[str, Any]:
    """Load metadata including test metrics and training statistics."""
    global _CACHED_METADATA
    if _CACHED_METADATA is None:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _CACHED_METADATA = json.load(f)
        else:
            _CACHED_METADATA = {}
    return _CACHED_METADATA


def get_feature_names() -> list[str]:
    """Retrieve cached feature names corresponding to pipeline columns."""
    global _CACHED_FEATURE_NAMES
    if _CACHED_FEATURE_NAMES is None:
        pipeline = get_model_pipeline()
        from src.feature_engineering import get_all_feature_names
        feature_pipe = pipeline.named_steps["feature_engineering"]
        _CACHED_FEATURE_NAMES = get_all_feature_names(feature_pipe)
    return _CACHED_FEATURE_NAMES


def explain_prediction(text: str, top_n: int = 8) -> dict[str, Any]:
    """Perform real-time prediction and feature-attribution explanation.
    
    Uses exact Logistic Regression additive log-odds decomposition:
    logit(p) = w_0 + sum(w_i * x_i)
    """
    start_time = time.perf_counter()
    pipeline = get_model_pipeline()
    metadata = get_model_metadata()
    optimal_threshold = metadata.get("optimal_threshold", 0.50)

    # 1. Transform raw text through feature engineering
    feature_pipe = pipeline.named_steps["feature_engineering"]
    classifier = pipeline.named_steps["classifier"]

    # Transform single sample
    X_trans = feature_pipe.transform([text])
    feature_names = get_feature_names()

    # 2. Probability and Prediction
    probabilities = classifier.predict_proba(X_trans)[0]
    p_ham = float(probabilities[0])
    p_spam = float(probabilities[1])

    is_spam = bool(p_spam >= optimal_threshold)
    label = "SPAM" if is_spam else "HAM"

    # Risk level categorization
    if p_spam >= 0.70:
        risk_level = "CRITICAL"
        risk_color = "danger"
    elif p_spam >= 0.35:
        risk_level = "SUSPICIOUS"
        risk_color = "warning"
    else:
        risk_level = "SAFE"
        risk_color = "success"

    # 3. Log-Odds Feature Decomposition
    # Active non-zero features for this specific input
    weights = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])

    # Convert sparse row to coordinate format
    coo = X_trans.tocoo()
    contributions = []

    for col_idx, value in zip(coo.col, coo.data):
        feat_name = feature_names[col_idx] if col_idx < len(feature_names) else f"feature_{col_idx}"
        w = float(weights[col_idx])
        contrib = float(w * value)
        contributions.append({
            "feature": feat_name,
            "value": float(value),
            "weight": w,
            "contribution": contrib,
            "pushes_toward": "SPAM" if contrib > 0 else "HAM",
        })

    # Sort by absolute log-odds magnitude
    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    spam_drivers = [c for c in contributions if c["contribution"] > 0][:top_n]
    ham_drivers = [c for c in contributions if c["contribution"] < 0][:top_n]

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "text": text,
        "label": label,
        "is_spam": is_spam,
        "spam_probability": p_spam,
        "ham_probability": p_ham,
        "spam_percentage": round(p_spam * 100, 2),
        "ham_percentage": round(p_ham * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "optimal_threshold": optimal_threshold,
        "intercept": intercept,
        "spam_drivers": spam_drivers,
        "ham_drivers": ham_drivers,
        "all_contributions": contributions[:top_n * 2],
        "inference_time_ms": round(elapsed_ms, 2),
    }


def batch_predict(texts: list[str]) -> list[dict[str, Any]]:
    """Efficient batch classification for multiple text messages."""
    pipeline = get_model_pipeline()
    metadata = get_model_metadata()
    threshold = metadata.get("optimal_threshold", 0.50)

    start_time = time.perf_counter()
    probas = pipeline.predict_proba(texts)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    results = []
    for text, prob_pair in zip(texts, probas):
        p_ham, p_spam = float(prob_pair[0]), float(prob_pair[1])
        is_spam = bool(p_spam >= threshold)
        results.append({
            "text": text,
            "label": "SPAM" if is_spam else "HAM",
            "is_spam": is_spam,
            "spam_probability": round(p_spam, 4),
            "spam_percentage": round(p_spam * 100, 1),
            "risk_level": "CRITICAL" if p_spam >= 0.70 else ("SUSPICIOUS" if p_spam >= 0.35 else "SAFE"),
        })

    return results
