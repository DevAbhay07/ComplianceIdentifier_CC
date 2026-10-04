"""
declaration_product_model.py
=============================
Wraps product-level ML classifiers (Logistic Regression, Random Forest, XGBoost)
trained on declaration-centric feature sets (Sets A-E).

Supports:
1. Supervised prediction (COMPLIANT, NON_COMPLIANT)
2. Calibrated / probability outputs
3. Evidence- and confidence-aware abstention to UNCERTAIN
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
class ProductPredictionResult:
    predicted_label: str  # COMPLIANT, NON_COMPLIANT, UNCERTAIN
    raw_label: str        # Model argmax label
    confidence: float     # 0.0 to 1.0
    probabilities: Dict[str, float]
    abstention_applied: bool
    abstention_reason: Optional[str]
    feature_set: str
    model_name: str
    model_version: str


class DeclarationProductModel:
    """
    Product-Level Compliance Model wrapper.
    Predicts product compliance from structured declaration features.
    """

    def __init__(
        self,
        base_estimator: Any,
        feature_set: str = "SET_E",
        feature_names: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        abstention_threshold: float = 0.65,
        margin_threshold: float = 0.20,
        model_name: str = "XGBoost",
        model_version: str = "v1.0-corrective"
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
                # Binary sigmoid
                p1 = 1.0 / (1.0 + np.exp(-df))
                return np.column_stack([1.0 - p1, p1])
            else:
                exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            # Fallback uniform
            n = len(X)
            k = len(self.classes)
            return np.ones((n, k)) / k

    def predict_single(
        self,
        features: np.ndarray,
        evidence_completeness: float = 1.0,
        has_conflicts: bool = False,
        ocr_confidence: float = 1.0
    ) -> ProductPredictionResult:
        """
        Predicts compliance for a single sample with confidence-based and evidence-aware abstention.
        """
        X = features.reshape(1, -1) if features.ndim == 1 else features
        probs = self.predict_proba(X)[0]
        
        # Map class probabilities
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(self.classes) if i < len(probs)}
        
        # Determine raw argmax
        max_idx = int(np.argmax(probs))
        raw_label = self.classes[max_idx]
        confidence = float(probs[max_idx])
        
        # Compute probability margin (difference between top 2 probs)
        sorted_probs = sorted(probs, reverse=True)
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        
        # Abstention logic:
        abstain = False
        abstention_reason = None
        
        if has_conflicts:
            abstain = True
            abstention_reason = "Conflicting declaration evidence observed across packaging views."
        elif evidence_completeness < 0.20 and ocr_confidence < 0.40:
            abstain = True
            abstention_reason = "Insufficient declaration and OCR evidence to determine compliance."
        elif confidence < self.abstention_threshold:
            abstain = True
            abstention_reason = f"Model prediction confidence ({confidence:.2%}) below abstention threshold ({self.abstention_threshold:.2%})."
        elif margin < self.margin_threshold:
            abstain = True
            abstention_reason = f"Model probability margin ({margin:.2f}) below margin threshold ({self.margin_threshold:.2f})."
            
        final_label = "UNCERTAIN" if abstain else raw_label
        
        return ProductPredictionResult(
            predicted_label=final_label,
            raw_label=raw_label,
            confidence=confidence,
            probabilities=prob_dict,
            abstention_applied=abstain,
            abstention_reason=abstention_reason,
            feature_set=self.feature_set,
            model_name=self.model_name,
            model_version=self.model_version
        )

    def predict_batch(
        self,
        X: np.ndarray,
        apply_abstention: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """
        Batch prediction for evaluation and testing.
        """
        probs = self.predict_proba(X)
        predictions = []
        for i in range(len(X)):
            max_idx = int(np.argmax(probs[i]))
            raw_label = self.classes[max_idx]
            conf = float(probs[i, max_idx])
            
            sorted_p = sorted(probs[i], reverse=True)
            margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
            
            if apply_abstention and (conf < self.abstention_threshold or margin < self.margin_threshold):
                predictions.append("UNCERTAIN")
            else:
                predictions.append(raw_label)
                
        return predictions, probs

    def save(self, filepath: str):
        """Persist model and metadata."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DeclarationProductModel":
        """Load model from filepath."""
        return joblib.load(filepath)
