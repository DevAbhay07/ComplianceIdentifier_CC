"""
predict_product_compliance.py
=============================
Authoritative inference module for predicting product-level compliance.
Loads trained ProductComplianceModel, computes declaration-centric features (Sets A-E),
and outputs COMPLIANT, NON_COMPLIANT, or UNCERTAIN.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.models.product_compliance_model import ProductComplianceModel
from src.decision.product_compliance_decision import UncertaintyStatus

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/product_compliance/best_product_compliance_model.joblib"


class ProductCompliancePredictor:
    """
    Inference predictor wrapper that maps product evidence records or structured
    declaration dictionaries to product compliance predictions.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, abstention_threshold: Optional[float] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"Product compliance model not found at {self.model_path}. Train model first.")

        logger.info(f"Loading ProductComplianceModel from {self.model_path}...")
        self.model = ProductComplianceModel.load(str(self.model_path))
        if abstention_threshold is not None:
            self.model.abstention_threshold = abstention_threshold

        self.feature_extractor = ProductComplianceFeatureExtractor()
        logger.info(f"Initialized ProductCompliancePredictor (Model: {self.model.model_name}, Feature Set: {self.model.feature_set})")

    def predict_record(self, record: ProductComplianceTrainingRecord, apply_abstention: bool = True) -> Dict[str, Any]:
        """
        Runs compliance prediction on a single ProductComplianceTrainingRecord.
        """
        X = self.feature_extractor.transform([record], feature_set=self.model.feature_set)
        
        # Compute dynamic completeness ratio and conflict status from record
        decls = record.declarations or {}
        present_count = sum(1 for d in decls.values() if isinstance(d, dict) and d.get("status") in ["PRESENT", "DeclarationStatus.PRESENT"])
        absent_count = sum(1 for d in decls.values() if isinstance(d, dict) and d.get("status") in ["ABSENT", "DeclarationStatus.ABSENT"])
        cannot_det_count = sum(1 for d in decls.values() if isinstance(d, dict) and d.get("status") in ["CANNOT_DETERMINE", "PARTIAL", "UNKNOWN"])
        tot_decls = len(decls) if len(decls) > 0 else 7
        comp_ratio = present_count / tot_decls
        
        meta = record.metadata or {}
        has_conflicts = (
            bool(meta.get("conflicting_fields")) or
            bool(meta.get("has_conflicts", False)) or
            any(
                isinstance(d, dict) and (
                    d.get("status") == "CONFLICTING_EVIDENCE" or
                    len(d.get("conflicting_values", [])) > 0
                )
                for d in decls.values()
            )
        )

        pred_obj = self.model.predict_single(
            features=X[0],
            completeness_ratio=comp_ratio,
            has_conflicts=has_conflicts,
            ocr_confidence=float(record.ocr_quality)
        )
        
        raw_pred = pred_obj.raw_prediction
        abs_pred = pred_obj.final_prediction if apply_abstention else raw_pred

        # Build explainability details
        f_names = self.feature_extractor.get_feature_names(self.model.feature_set)
        feat_dict = {name: float(X[0, i]) for i, name in enumerate(f_names)}

        # Determine statutory compliance reasons
        reasons = []
        decls = record.declarations or {}
        if decls.get("D01", {}).get("status") == "ABSENT":
            reasons.append("Missing mandatory Manufacturer / Packer name and address (D01)")
        if decls.get("D02", {}).get("status") == "ABSENT":
            reasons.append("Missing mandatory Generic Commodity Name (D02)")
        if decls.get("D03", {}).get("status") == "ABSENT":
            reasons.append("Missing mandatory Net Quantity declaration (D03)")
        if decls.get("D04", {}).get("status") == "ABSENT":
            reasons.append("Missing mandatory Date of Manufacture / Packing (D04)")
        if decls.get("D05", {}).get("status") == "ABSENT":
            reasons.append("Missing mandatory Maximum Retail Price / Unit Sale Price (D05)")
        if decls.get("D06", {}).get("status") == "ABSENT":
            reasons.append("Missing mandatory Consumer Care Contact Details (D06)")
        if decls.get("D07", {}).get("status") == "ABSENT":
            reasons.append("Missing mandatory Country of Origin (D07)")

        if not reasons and abs_pred == "COMPLIANT":
            reasons.append("All mandatory statutory declarations verified on packaging")
        elif not reasons and abs_pred == "UNCERTAIN":
            reasons.append(f"Model prediction uncertainty: {pred_obj.abstention_reason}")

        # ═══════════════════════════════════════════════════════
        # Evidence-Quality-Adjusted Confidence Calibration
        # Model probability alone ≠ legal certainty.
        # Calibrated confidence = model_probability × evidence_quality_factor
        # ═══════════════════════════════════════════════════════
        raw_model_confidence = float(pred_obj.confidence)
        
        # Evidence quality factors
        ocr_quality_factor = min(1.0, max(0.3, float(record.ocr_quality)))
        completeness_factor = min(1.0, max(0.2, comp_ratio))
        conflict_penalty = 0.7 if has_conflicts else 1.0
        
        # Combined evidence quality ceiling
        evidence_quality = ocr_quality_factor * completeness_factor * conflict_penalty
        
        # Calibrated confidence: model prob × evidence quality, but floor at 0.1
        calibrated_confidence = round(max(0.1, raw_model_confidence * evidence_quality), 4)
        
        # Build canonical uncertainty container
        unc_score = round(float(1.0 - calibrated_confidence), 4)
        if pred_obj.abstention_applied or abs_pred == "UNCERTAIN" or unc_score >= 0.45:
            unc_status = "HIGH"
        elif unc_score >= 0.25:
            unc_status = "MEDIUM"
        else:
            unc_status = "LOW"

        unc_reason = pred_obj.abstention_reason or (
            "High prediction certainty." if unc_status == "LOW" else "Moderate prediction certainty."
        )

        unc_obj = UncertaintyStatus(
            is_uncertain=(abs_pred == "UNCERTAIN" or pred_obj.abstention_applied or unc_status == "HIGH"),
            status=unc_status,
            score=unc_score,
            reason=unc_reason,
            abstained=pred_obj.abstention_applied if apply_abstention else False
        )

        return {
            "overall_prediction": abs_pred,
            "raw_prediction": raw_pred,
            "confidence": calibrated_confidence,
            "raw_model_confidence": raw_model_confidence,
            "evidence_quality": round(evidence_quality, 4),
            "margin": round(float(pred_obj.probability_margin), 4),
            "probabilities": pred_obj.probabilities,
            "abstention_applied": pred_obj.abstention_applied if apply_abstention else False,
            "abstention_reason": pred_obj.abstention_reason if apply_abstention else None,
            "uncertainty": unc_obj,
            "decision_basis": "; ".join(reasons) if reasons else "Based on multi-declaration feature analysis",
            "features": feat_dict,
            "feature_set": self.model.feature_set,
            "model_version": self.model.model_version
        }

    def predict_batch(self, records: List[ProductComplianceTrainingRecord], apply_abstention: bool = True) -> List[Dict[str, Any]]:
        """
        Runs batch compliance prediction over a list of records.
        """
        return [self.predict_record(r, apply_abstention=apply_abstention) for r in records]
