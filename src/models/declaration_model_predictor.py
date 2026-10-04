"""
declaration_model_predictor.py
==============================
High-level prediction interface that maps a structured ProductEvidenceRecord
into declaration features, evaluates the trained DeclarationProductModel,
and returns full prediction diagnostics, confidence, and decision traces.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.declaration_schema import ProductEvidenceRecord, DeclarationStatus
from src.features.declaration_feature_extractor import DeclarationFeatureExtractor
from src.models.declaration_product_model import DeclarationProductModel, ProductPredictionResult


class DeclarationModelPredictor:
    """
    High-level predictor executing:
    ProductEvidenceRecord -> Declaration Features -> DeclarationProductModel -> Compliance Result
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_set: Optional[str] = None
    ):
        if model_path is None:
            default_path = PROJECT_ROOT / "models/declaration_centric/best_declaration_model.joblib"
            model_path = str(default_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Declaration model not found at: {model_path}")

        self.model: DeclarationProductModel = DeclarationProductModel.load(model_path)
        self.feature_set = feature_set or self.model.feature_set
        self.feature_extractor = DeclarationFeatureExtractor(default_feature_set=self.feature_set)

    def predict(
        self,
        record: ProductEvidenceRecord
    ) -> Dict[str, Any]:
        """
        Executes end-to-end declaration-centric prediction on a ProductEvidenceRecord.
        """
        start_time = time.time()

        # 1. Feature extraction
        features = self.feature_extractor.transform([record], feature_set=self.feature_set)
        feature_names = self.feature_extractor.get_feature_names(self.feature_set)
        feat_vector = features[0]
        feature_dict = {f_name: float(feat_vector[i]) for i, f_name in enumerate(feature_names)}

        # 2. Extract quality / conflict indicators
        completeness = record.declaration_completeness_ratio
        has_conflicts = len(record.conflicting_fields) > 0
        mean_ocr_conf = record.overall_ocr_confidence

        # 3. Model prediction with abstention
        res: ProductPredictionResult = self.model.predict_single(
            features=feat_vector,
            evidence_completeness=completeness,
            has_conflicts=has_conflicts,
            ocr_confidence=mean_ocr_conf
        )

        latency_ms = (time.time() - start_time) * 1000.0

        # 4. Formulate human-readable decision basis
        if res.predicted_label == "UNCERTAIN":
            basis = f"Prediction abstained to UNCERTAIN. Reason: {res.abstention_reason}"
        elif res.predicted_label == "NON_COMPLIANT":
            missing = [did for did, d in record.declarations.items() if d.status == DeclarationStatus.ABSENT]
            if missing:
                basis = f"Product predicted NON_COMPLIANT due to missing statutory declarations ({', '.join(missing)}) with confidence {res.confidence:.1%}."
            else:
                basis = f"Product predicted NON_COMPLIANT based on declaration feature profile with confidence {res.confidence:.1%}."
        else:
            basis = f"All mandatory declarations verified present and valid. Product predicted COMPLIANT with confidence {res.confidence:.1%}."

        return {
            "predicted_label": res.predicted_label,
            "raw_label": res.raw_label,
            "confidence": round(res.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in res.probabilities.items()},
            "abstention_applied": res.abstention_applied,
            "abstention_reason": res.abstention_reason,
            "decision_basis": basis,
            "feature_set": self.feature_set,
            "model_name": self.model.model_name,
            "model_version": self.model.model_version,
            "feature_count": len(feature_names),
            "feature_trace": feature_dict,
            "latency_ms": round(latency_ms, 2)
        }
