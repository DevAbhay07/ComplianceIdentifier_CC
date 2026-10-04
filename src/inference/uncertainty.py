"""
uncertainty.py
==============
Computes model uncertainty metrics (maximum probability, margin of confidence,
normalized Shannon entropy) and flags predictions requiring human verification.
"""

import numpy as np
from typing import Dict, Any, List, Optional

DEFAULT_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_MARGIN_THRESHOLD = 0.15


def compute_prediction_uncertainty(
    probabilities: Dict[str, float],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD
) -> Dict[str, Any]:
    """
    Computes uncertainty metrics from a class probability distribution.

    Returns:
        Dict[str, Any]:
        {
            "confidence": float,
            "margin": float,
            "entropy": float,
            "decision": "PREDICT" | "UNCERTAIN",
            "uncertainty_reason": str or None
        }
    """
    if not probabilities:
        return {
            "confidence": 1.0,
            "margin": 1.0,
            "entropy": 0.0,
            "decision": "PREDICT",
            "uncertainty_reason": None
        }

    probs = list(probabilities.values())
    if len(probs) == 1:
        return {
            "confidence": 1.0,
            "margin": 1.0,
            "entropy": 0.0,
            "decision": "PREDICT",
            "uncertainty_reason": None
        }

    sorted_probs = sorted(probs, reverse=True)
    p_top = sorted_probs[0]
    p_second = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = round(float(p_top - p_second), 4)

    # Shannon entropy
    k = len(probs)
    eps = 1e-12
    raw_entropy = -sum(p * np.log2(p + eps) for p in probs if p > 0)
    max_entropy = np.log2(k) if k > 1 else 1.0
    normalized_entropy = round(float(raw_entropy / max_entropy), 4)

    # Decision logic
    reasons = []
    if p_top < confidence_threshold:
        reasons.append(f"Low confidence ({p_top:.2f} < {confidence_threshold:.2f})")
    if margin < margin_threshold:
        reasons.append(f"Narrow margin ({margin:.2f} < {margin_threshold:.2f})")

    decision = "UNCERTAIN" if reasons else "PREDICT"
    reason_str = "; ".join(reasons) if reasons else None

    return {
        "confidence": round(float(p_top), 4),
        "margin": margin,
        "entropy": normalized_entropy,
        "decision": decision,
        "uncertainty_reason": reason_str
    }
