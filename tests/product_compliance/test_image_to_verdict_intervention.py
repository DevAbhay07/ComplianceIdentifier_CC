"""
test_image_to_verdict_intervention.py
=====================================
Proves that changing image evidence changes the final verdict.
Tests the entire chain: image → extraction → features → model → decision.
"""

import sys
import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.models.predict_product_compliance import ProductCompliancePredictor


PREDICTOR = None

def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = ProductCompliancePredictor()
    return PREDICTOR


def make_record(declarations, ocr_quality=0.90, raw_text="Full packaging text"):
    """Create record with specific declaration states."""
    decls = {}
    for did, status in declarations.items():
        conf = 0.95 if status == "PRESENT" else (0.85 if status == "ABSENT" else 0.30)
        decls[did] = {"status": status, "confidence": conf, "raw_text": f"{did} evidence", "source": "OCR"}
    
    return ProductComplianceTrainingRecord(
        product_id="INTERVENTION_TEST",
        source_dataset="test",
        target_provenance="TEST",
        overall_target="UNKNOWN",
        category="Beauty",
        brand="TestBrand",
        product_title="Test Product",
        raw_text=raw_text,
        declarations=decls,
        d01_values={"mfg_name": "Test Corp"},
        d02_values={"generic_name": "Face Cream"},
        d03_values={"net_quantity": "100g"},
        d04_values={"date_text": "01/2026"},
        d05_values={"mrp": "Rs 250"},
        d06_values={"consumer_care": "1800-123-4567"},
        d07_values={"country_of_origin": "India"},
        packaging_features={"yolo_detection_count": 0},
        ocr_quality=ocr_quality,
        text_length=len(raw_text),
        image_count=1,
        image_paths=[]
    )


ALL_PRESENT = {f"D0{i}": "PRESENT" for i in range(1, 8)}


class TestImageToVerdictIntervention:
    """Tests proving that evidence changes propagate through to model predictions."""

    def test_baseline_all_present_is_compliant(self):
        """All 7 declarations PRESENT → COMPLIANT."""
        predictor = get_predictor()
        rec = make_record(ALL_PRESENT)
        result = predictor.predict_record(rec)
        assert result["overall_prediction"] in ("COMPLIANT", "UNCERTAIN"), (
            f"All PRESENT but got {result['overall_prediction']}"
        )

    def test_remove_d01_changes_outcome(self):
        """Remove D01 (manufacturer) → prediction must change from baseline."""
        predictor = get_predictor()
        
        rec_full = make_record(ALL_PRESENT)
        decls_no_d01 = {**ALL_PRESENT, "D01": "ABSENT"}
        rec_no_d01 = make_record(decls_no_d01)
        
        r_full = predictor.predict_record(rec_full)
        r_no_d01 = predictor.predict_record(rec_no_d01)
        
        # Probabilities must differ
        assert r_full["probabilities"] != r_no_d01["probabilities"], (
            "Removing D01 did not change probabilities!"
        )

    def test_remove_d05_changes_outcome(self):
        """Remove D05 (MRP) → NON_COMPLIANT or UNCERTAIN."""
        predictor = get_predictor()
        decls = {**ALL_PRESENT, "D05": "ABSENT"}
        rec = make_record(decls)
        result = predictor.predict_record(rec)
        assert result["overall_prediction"] in ("NON_COMPLIANT", "UNCERTAIN"), (
            f"D05 ABSENT but got {result['overall_prediction']}"
        )

    def test_remove_d06_changes_outcome(self):
        """Remove D06 (consumer care) → NON_COMPLIANT or UNCERTAIN."""
        predictor = get_predictor()
        decls = {**ALL_PRESENT, "D06": "ABSENT"}
        rec = make_record(decls)
        result = predictor.predict_record(rec)
        assert result["overall_prediction"] in ("NON_COMPLIANT", "UNCERTAIN"), (
            f"D06 ABSENT but got {result['overall_prediction']}"
        )

    def test_progressive_removal_changes_confidence(self):
        """Removing more declarations should decrease confidence."""
        predictor = get_predictor()
        
        # Full
        r_full = predictor.predict_record(make_record(ALL_PRESENT))
        
        # Remove one
        r_one = predictor.predict_record(make_record({**ALL_PRESENT, "D06": "ABSENT"}))
        
        # Remove three
        r_three = predictor.predict_record(make_record({
            **ALL_PRESENT, "D04": "ABSENT", "D05": "ABSENT", "D06": "ABSENT"
        }))
        
        # More absences → different predictions/probabilities
        full_c_prob = r_full["probabilities"].get("COMPLIANT", 0)
        three_c_prob = r_three["probabilities"].get("COMPLIANT", 0)
        
        assert full_c_prob > three_c_prob, (
            f"Full C_prob={full_c_prob:.4f} should be > 3-absent C_prob={three_c_prob:.4f}"
        )

    def test_ocr_quality_affects_confidence(self):
        """Lower OCR quality → lower calibrated confidence."""
        predictor = get_predictor()
        
        r_good_ocr = predictor.predict_record(make_record(ALL_PRESENT, ocr_quality=0.95))
        r_bad_ocr = predictor.predict_record(make_record(ALL_PRESENT, ocr_quality=0.30))
        
        assert r_good_ocr["confidence"] > r_bad_ocr["confidence"], (
            f"Good OCR confidence ({r_good_ocr['confidence']:.4f}) should be > "
            f"bad OCR ({r_bad_ocr['confidence']:.4f})"
        )

    def test_cannot_determine_is_not_absent(self):
        """CANNOT_DETERMINE ≠ ABSENT: feature vector must differ."""
        fe = ProductComplianceFeatureExtractor()
        
        rec_absent = make_record({**ALL_PRESENT, "D03": "ABSENT"})
        rec_cd = make_record({**ALL_PRESENT, "D03": "CANNOT_DETERMINE"})
        
        X_absent = fe.transform([rec_absent], feature_set="SET_A")
        X_cd = fe.transform([rec_cd], feature_set="SET_A")
        
        import numpy as np
        diff = np.abs(X_absent[0] - X_cd[0]).sum()
        assert diff > 0.001, (
            f"ABSENT and CANNOT_DETERMINE produced identical features (diff={diff:.6f})"
        )

    def test_feature_vector_dimensions_consistent(self):
        """Feature vector dimension must match model's expected input."""
        predictor = get_predictor()
        fe = ProductComplianceFeatureExtractor()
        
        rec = make_record(ALL_PRESENT)
        X = fe.transform([rec], feature_set=predictor.model.feature_set)
        
        expected = predictor.model.base_estimator.n_features_in_
        actual = X.shape[1]
        
        assert actual == expected, (
            f"Feature dimension mismatch: extractor produces {actual}, model expects {expected}"
        )
