"""
product_compliance_model.py
===========================
Supervised Product-Level Compliance Model wrapper.

Capabilities:
1. Supervised probability prediction (COMPLIANT, NON_COMPLIANT)
2. Evidence-aware and confidence-calibrated abstention to UNCERTAIN
3. Distinction between RAW_ML_PREDICTION and FINAL_ABSTAINED_PREDICTION
4. Serialization and inspection metadata
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict


@dataclass
class ProductCompliancePrediction:
    raw_prediction: str         # Model argmax label (COMPLIANT / NON_COMPLIANT)
    final_prediction: str       # Prediction after abstention (COMPLIANT / NON_COMPLIANT / UNCERTAIN)
    confidence: float           # 0.0 to 1.0
    probabilities: Dict[str, float]
    probability_margin: float   # Difference between top 2 class probabilities
    abstention_applied: bool
    abstention_reason: Optional[str]
    feature_set: str
    model_name: str
    model_version: str
    source_contribution: Optional[str] = None


class ProductComplianceModel:
    """
    Product-Level Compliance Model wrapper.
    Evaluates compliance from common declaration and physical packaging feature vectors.
    """

    def __init__(
        self,
        base_estimator: Any,
        feature_set: str = "SET_E",
        feature_names: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        abstention_threshold: float = 0.65,
        margin_threshold: float = 0.20,
        model_name: str = "LogisticRegression",
        model_version: str = "v2.0-product-compliance"
    ):
        self.base_estimator = base_estimator
        self.feature_set = feature_set
        self.feature_names = feature_names or []
        self.classes = classes or ["COMPLIANT", "NON_COMPLIANT"]
        self.abstention_threshold = abstention_threshold
        self.margin_threshold = margin_threshold
        self.model_name = model_name
        self.model_version = model_version

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities (N, num_classes)."""
        if hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X)
        elif hasattr(self.base_estimator, "decision_function"):
            df = self.base_estimator.decision_function(X)
            if df.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-df))
                return np.column_stack([1.0 - p1, p1])
            else:
                exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            n = len(X)
            k = len(self.classes)
            return np.ones((n, k)) / k

    def predict_single(
        self,
        features: np.ndarray,
        completeness_ratio: float = 1.0,
        has_conflicts: bool = False,
        ocr_confidence: float = 1.0
    ) -> ProductCompliancePrediction:
        """
        Predicts compliance for a single feature vector, clearly separating raw from abstained results.
        """
        X = features.reshape(1, -1) if features.ndim == 1 else features
        probs = self.predict_proba(X)[0]

        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(self.classes) if i < len(probs)}
        max_idx = int(np.argmax(probs))
        raw_label = self.classes[max_idx]
        confidence = float(probs[max_idx])

        sorted_probs = sorted(probs, reverse=True)
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

        abstain = False
        reason = None

        if has_conflicts:
            abstain = True
            reason = "Conflicting declaration evidence observed across packaging views."
        elif completeness_ratio == 0.0 and (ocr_confidence < 0.60 or len(features) == 0):
            abstain = True
            reason = "No readable packaging text or declaration evidence detected on package (Abstained to UNCERTAIN)."
        elif completeness_ratio < 0.35 and ocr_confidence < 0.60:
            abstain = True
            reason = "Insufficient declaration and optical clarity to verify compliance."
        elif confidence < self.abstention_threshold:
            abstain = True
            reason = f"Prediction confidence ({confidence:.1%}) below calibrated threshold ({self.abstention_threshold:.1%})."
        elif margin < self.margin_threshold:
            abstain = True
            reason = f"Probability margin ({margin:.2f}) below minimum margin ({self.margin_threshold:.2f})."

        final_label = "UNCERTAIN" if abstain else raw_label

        return ProductCompliancePrediction(
            raw_prediction=raw_label,
            final_prediction=final_label,
            confidence=confidence,
            probabilities=prob_dict,
            probability_margin=margin,
            abstention_applied=abstain,
            abstention_reason=reason,
            feature_set=self.feature_set,
            model_name=self.model_name,
            model_version=self.model_version
        )

    def predict_batch(
        self,
        X: np.ndarray,
        apply_abstention: bool = True
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Batch prediction returning: (raw_predictions, final_predictions, probabilities).
        """
        probs = self.predict_proba(X)
        raw_preds = []
        final_preds = []

        for i in range(len(X)):
            max_idx = int(np.argmax(probs[i]))
            raw_label = self.classes[max_idx]
            conf = float(probs[i, max_idx])

            sorted_p = sorted(probs[i], reverse=True)
            margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

            raw_preds.append(raw_label)
            if apply_abstention and (conf < self.abstention_threshold or margin < self.margin_threshold):
                final_preds.append("UNCERTAIN")
            else:
                final_preds.append(raw_label)

        return raw_preds, final_preds, probs

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ProductComplianceModel":
        return joblib.load(filepath)
