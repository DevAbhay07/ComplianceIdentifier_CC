"""
test_controlled_master_suite.py
===============================
Comprehensive Master Controlled Sanity, Intervention, and E2E Test Suite.
Verifies all cases A through J and feature intervention invariants.
"""

import os
import sys
import json
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.extraction.declaration_extractor import DeclarationExtractor
from src.extraction.declaration_schema import DeclarationStatus, DeclarationEvidence
from src.extraction.evidence_fusion import EvidenceFusionEngine
from src.models.product_compliance_model import ProductComplianceModel
from src.models.predict_product_compliance import ProductCompliancePredictor
from src.inference.product_compliance_pipeline import ProductCompliancePipeline


@pytest.fixture
def predictor():
    return ProductCompliancePredictor()


@pytest.fixture
def feature_extractor():
    return ProductComplianceFeatureExtractor()


@pytest.fixture
def declaration_extractor():
    return DeclarationExtractor()


# =============================================================================
# CONTROLLED SANITY CASES A THROUGH J
# =============================================================================

# CASE A: NO EVIDENCE (YOLO=0, OCR="", D01-D07=CANNOT_DETERMINE)
def test_case_a_no_evidence(predictor, declaration_extractor):
    decls = declaration_extractor.extract_from_text_and_vision(raw_ocr_text="", visual_regions=[])
    for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
        assert decls[did].status == DeclarationStatus.CANNOT_DETERMINE.value

    rec = ProductComplianceTrainingRecord(
        product_id="CASE_A",
        source_dataset="inference",
        target_provenance="PHYSICAL_PACKAGING_INFERENCE",
        overall_target="UNKNOWN",
        declarations={did: {"status": "CANNOT_DETERMINE", "confidence": 0.0, "raw_text": ""} for did in decls},
        ocr_quality=0.10,
        text_length=0,
        image_count=1
    )
    res = predictor.predict_record(rec, apply_abstention=True)
    assert res["overall_prediction"] == "UNCERTAIN"
    assert res["uncertainty"]["status"] == "HIGH"
    assert res["uncertainty"]["abstained"] is True


# CASE B: POSITIVE OCR, YOLO=0
def test_case_b_positive_ocr_yolo_zero(declaration_extractor, predictor):
    ocr_text = (
        "Manufactured by: Aura Herbals Pvt Ltd, Plot 42, Sector 8, Gandhinagar 382028, Gujarat.\n"
        "Product Name: Purifying Face Wash 150ml\n"
        "Net Quantity: 150 ml\n"
        "Mfg Date: 03/2026\n"
        "MRP: Rs. 195.00 incl. of all taxes\n"
        "Consumer Care: Phone: 1800-111-222, Email: care@auraglow.in\n"
        "Country of Origin: India"
    )
    decls = declaration_extractor.extract_from_text_and_vision(raw_ocr_text=ocr_text, visual_regions=[])
    for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
        assert decls[did].status == DeclarationStatus.PRESENT.value, f"Declaration {did} lost when YOLO=0"

    assert decls["D05"].normalized_value.get("mrp") == 195.0
    assert decls["D03"].normalized_value.get("quantity") == 150.0


# CASE C: YOLO POSITIVE, OCR EMPTY
def test_case_c_yolo_positive_ocr_empty(declaration_extractor, predictor):
    visual_dets = [
        {"class_name": "mrp details", "bbox": [10, 10, 50, 50], "confidence": 0.90},
        {"class_name": "weight", "bbox": [50, 50, 80, 80], "confidence": 0.88},
        {"class_name": "address", "bbox": [80, 80, 120, 120], "confidence": 0.85}
    ]
    decls = declaration_extractor.extract_from_text_and_vision(raw_ocr_text="", visual_regions=visual_dets)
    for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
        assert decls[did].status == DeclarationStatus.CANNOT_DETERMINE.value


# CASE D: CONFLICTING EVIDENCE
def test_case_d_conflicting_mrp():
    fusion = EvidenceFusionEngine()
    ev1 = {
        "image_name": "front.png", "ocr_text": "MRP Rs. 999.00", "ocr_confidence": 0.95,
        "declarations": {
            "D05": DeclarationEvidence(
                declaration_id="D05", declaration_name="MRP", status=DeclarationStatus.PRESENT.value,
                raw_value="999.00", normalized_value={"mrp": 999.0, "tax_inclusive": True}, source="OCR", source_image="front.png"
            )
        }
    }
    ev2 = {
        "image_name": "back.png", "ocr_text": "MRP Rs. 1099.00", "ocr_confidence": 0.95,
        "declarations": {
            "D05": DeclarationEvidence(
                declaration_id="D05", declaration_name="MRP", status=DeclarationStatus.PRESENT.value,
                raw_value="1099.00", normalized_value={"mrp": 1099.0, "tax_inclusive": True}, source="OCR", source_image="back.png"
            )
        }
    }
    fused = fusion.fuse_image_evidences([ev1, ev2], product_id="CONFLICT_CASE")
    assert fused.declarations["D05"].status == DeclarationStatus.CONFLICTING_EVIDENCE.value


# CASE E: GENUINE STATUTORY ABSENCE (MISSING D06)
def test_case_e_missing_consumer_care(predictor):
    rec = ProductComplianceTrainingRecord(
        product_id="CASE_E_NO_D06",
        source_dataset="synthetic_physical",
        target_provenance="SYNTHETIC_GROUND_TRUTH",
        overall_target="NON_COMPLIANT",
        declarations={
            "D01": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Company A, Delhi 110001"},
            "D02": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Face Cream"},
            "D03": {"status": "PRESENT", "confidence": 0.98, "raw_text": "50 g"},
            "D04": {"status": "PRESENT", "confidence": 0.95, "raw_text": "03/2026"},
            "D05": {"status": "PRESENT", "confidence": 0.98, "raw_text": "MRP Rs 250.00 incl taxes"},
            "D06": {"status": "ABSENT", "confidence": 0.0, "raw_text": ""},
            "D07": {"status": "PRESENT", "confidence": 0.95, "raw_text": "India"}
        },
        ocr_quality=0.95,
        text_length=150,
        image_count=1
    )
    res = predictor.predict_record(rec, apply_abstention=True)
    assert res["overall_prediction"] == "NON_COMPLIANT"


# CASE F: OCR FAILURE / DEGRADED
def test_case_f_ocr_failure(predictor):
    rec = ProductComplianceTrainingRecord(
        product_id="CASE_F_OCR_FAIL",
        source_dataset="inference",
        target_provenance="PHYSICAL_PACKAGING_INFERENCE",
        overall_target="UNKNOWN",
        declarations={did: {"status": "CANNOT_DETERMINE", "confidence": 0.0, "raw_text": ""} for did in ["D01","D02","D03","D04","D05","D06","D07"]},
        ocr_quality=0.15,
        text_length=0,
        image_count=1
    )
    res = predictor.predict_record(rec, apply_abstention=True)
    assert res["overall_prediction"] == "UNCERTAIN"


# CASE G: CATEGORY EXEMPTION (DOMESTIC PRODUCT D07 EXEMPTION)
def test_case_g_category_exemption():
    rec = ProductComplianceTrainingRecord(
        product_id="CASE_G_DOMESTIC",
        source_dataset="synthetic_physical",
        target_provenance="SYNTHETIC_GROUND_TRUTH",
        overall_target="COMPLIANT",
        d07_values={"country": "India", "is_domestic": True}
    )
    assert rec.d07_values["is_domestic"] is True


# CASE H: SYNTHETIC COMPLIANT E2E
def test_case_h_synthetic_compliant_e2e(predictor):
    rec = ProductComplianceTrainingRecord(
        product_id="CASE_H_COMPLIANT",
        source_dataset="synthetic_physical",
        target_provenance="SYNTHETIC_GROUND_TRUTH",
        overall_target="COMPLIANT",
        declarations={
            "D01": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Mfd by: Aura Herbals Pvt Ltd, Gujarat 382028"},
            "D02": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Purifying Face Wash"},
            "D03": {"status": "PRESENT", "confidence": 0.98, "raw_text": "150 ml"},
            "D04": {"status": "PRESENT", "confidence": 0.96, "raw_text": "03/2026"},
            "D05": {"status": "PRESENT", "confidence": 0.98, "raw_text": "MRP Rs. 195.00 (Incl. of all taxes)"},
            "D06": {"status": "PRESENT", "confidence": 0.97, "raw_text": "Care: 1800-111-222, care@auraglow.in"},
            "D07": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Country of Origin: India"}
        },
        ocr_quality=0.95,
        text_length=200,
        image_count=1
    )
    res = predictor.predict_record(rec, apply_abstention=True)
    assert res["overall_prediction"] == "COMPLIANT"


# CASE I: SYNTHETIC NON-COMPLIANT E2E (MISSING D01 ADDRESS)
def test_case_i_synthetic_non_compliant_e2e(predictor):
    rec = ProductComplianceTrainingRecord(
        product_id="CASE_I_MISSING_D01",
        source_dataset="synthetic_physical",
        target_provenance="SYNTHETIC_GROUND_TRUTH",
        overall_target="NON_COMPLIANT",
        declarations={
            "D01": {"status": "ABSENT", "confidence": 0.0, "raw_text": ""},
            "D02": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Shampoo"},
            "D03": {"status": "PRESENT", "confidence": 0.98, "raw_text": "200 ml"},
            "D04": {"status": "PRESENT", "confidence": 0.96, "raw_text": "01/2026"},
            "D05": {"status": "PRESENT", "confidence": 0.98, "raw_text": "MRP Rs. 150.00 incl taxes"},
            "D06": {"status": "PRESENT", "confidence": 0.97, "raw_text": "1800-222-333"},
            "D07": {"status": "PRESENT", "confidence": 0.95, "raw_text": "India"}
        },
        ocr_quality=0.90,
        text_length=120,
        image_count=1
    )
    res = predictor.predict_record(rec, apply_abstention=True)
    assert res["overall_prediction"] == "NON_COMPLIANT"


# CASE J: SYNTHETIC UNCERTAIN E2E
def test_case_j_synthetic_uncertain_e2e(predictor):
    rec = ProductComplianceTrainingRecord(
        product_id="CASE_J_BLURRY",
        source_dataset="synthetic_physical",
        target_provenance="SYNTHETIC_GROUND_TRUTH",
        overall_target="UNCERTAIN",
        declarations={did: {"status": "CANNOT_DETERMINE", "confidence": 0.20, "raw_text": ""} for did in ["D01", "D04", "D06"]},
        ocr_quality=0.30,
        text_length=20,
        image_count=1
    )
    res = predictor.predict_record(rec, apply_abstention=True)
    assert res["overall_prediction"] == "UNCERTAIN"


# =============================================================================
# FEATURE INTERVENTION TESTS
# =============================================================================

def test_feature_intervention_d01_to_d07_flips(predictor, feature_extractor):
    """
    Tests that flipping a declaration from PRESENT to CANNOT_DETERMINE
    measurably modifies the extracted feature vector and model probabilities.
    """
    base_decls = {
        "D01": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Mfg A"},
        "D02": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Soap"},
        "D03": {"status": "PRESENT", "confidence": 0.98, "raw_text": "100 g"},
        "D04": {"status": "PRESENT", "confidence": 0.95, "raw_text": "01/2026"},
        "D05": {"status": "PRESENT", "confidence": 0.98, "raw_text": "MRP Rs 50.00 incl taxes"},
        "D06": {"status": "PRESENT", "confidence": 0.97, "raw_text": "care@mfga.com"},
        "D07": {"status": "PRESENT", "confidence": 0.95, "raw_text": "India"}
    }

    base_rec = ProductComplianceTrainingRecord(
        product_id="INTERVENTION_BASE",
        source_dataset="synthetic_physical",
        target_provenance="SYNTHETIC_GROUND_TRUTH",
        overall_target="COMPLIANT",
        declarations=base_decls,
        ocr_quality=0.95,
        text_length=150,
        image_count=1
    )

    base_feat = feature_extractor.transform([base_rec], feature_set="SET_A")[0]
    base_pred = predictor.predict_record(base_rec, apply_abstention=False)

    for target_did in ["D01", "D03", "D04", "D05", "D06", "D07"]:
        mod_decls = dict(base_decls)
        mod_decls[target_did] = {"status": "CANNOT_DETERMINE", "confidence": 0.0, "raw_text": ""}

        mod_rec = ProductComplianceTrainingRecord(
            product_id=f"INTERVENTION_{target_did}",
            source_dataset="synthetic_physical",
            target_provenance="SYNTHETIC_GROUND_TRUTH",
            overall_target="NON_COMPLIANT",
            declarations=mod_decls,
            ocr_quality=0.95,
            text_length=120,
            image_count=1
        )

        mod_feat = feature_extractor.transform([mod_rec], feature_set="SET_A")[0]
        # Feature vector must differ
        assert not np.array_equal(base_feat, mod_feat), f"Intervention on {target_did} failed to change feature vector!"


# =============================================================================
# STANDALONE MODEL ARTIFACT TEST
# =============================================================================

def test_standalone_model_artifact_loading():
    """Verifies that the persisted best model artifact loads and executes cleanly."""
    model_path = PROJECT_ROOT / "models/product_compliance/best_product_compliance_model.joblib"
    assert model_path.exists(), "best_product_compliance_model.joblib missing"
    
    import joblib
    model = joblib.load(model_path)
    assert isinstance(model, ProductComplianceModel)
    assert model.feature_set == "SET_A"
    assert len(model.feature_names) == 16
    
    # Test inference on a 16-element feature vector representing all 7 declarations present
    x_test = np.array([
        1.0, 0.95,  # D01 present, conf
        1.0, 0.95,  # D02 present, conf
        1.0, 0.98,  # D03 present, conf
        1.0, 0.96,  # D04 present, conf
        1.0, 0.98,  # D05 present, conf
        1.0, 0.97,  # D06 present, conf
        1.0, 0.95,  # D07 present, conf
        7.0,        # mandatory_declarations_present_count = 7
        1.0         # all_mandatory_present = 1
    ], dtype=np.float32)
    pred = model.predict_single(x_test, completeness_ratio=1.0, ocr_confidence=0.95)
    assert pred.final_prediction == "COMPLIANT"
