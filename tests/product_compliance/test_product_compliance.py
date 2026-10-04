"""
test_product_compliance.py
==========================
Comprehensive test suite verifying all 17 requirements and invariants for
the Product-Level Compliance ML system.
"""

import os
import sys
import json
import pytest
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_training_representation import (
    ProductComplianceTrainingRecord, TargetProvenance, DatasetSource
)
from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.models.product_compliance_model import ProductComplianceModel, ProductCompliancePrediction
from src.models.predict_product_compliance import ProductCompliancePredictor
from src.inference.product_compliance_pipeline import ProductCompliancePipeline
from app.utils.inference_bridge import InferencePipelineBridge
from src.extraction.declaration_schema import ProductEvidenceRecord, DeclarationEvidence, DeclarationStatus


@pytest.fixture
def mock_record():
    return ProductComplianceTrainingRecord(
        product_id="TEST_001",
        source_dataset="yolo_physical_packaging",
        target_provenance="HUMAN_VERIFIED",
        overall_target="COMPLIANT",
        category="Beauty",
        brand="Himalaya",
        product_title="Himalaya Neem Face Wash 150ml",
        raw_text="Himalaya Neem Face Wash Net Vol 150ml MRP Rs 195.00 incl of all taxes Mfd 11/2025 Care 1800-208-1930 India",
        declarations={
            "D01": {"status": "PRESENT", "confidence": 0.95, "raw_text": "The Himalaya Drug Company, Makali, Bengaluru"},
            "D02": {"status": "PRESENT", "confidence": 0.95, "raw_text": "Neem Face Wash"},
            "D03": {"status": "PRESENT", "confidence": 0.98, "raw_text": "150 ml"},
            "D04": {"status": "PRESENT", "confidence": 0.95, "raw_text": "11/2025"},
            "D05": {"status": "PRESENT", "confidence": 0.98, "raw_text": "MRP Rs 195.00 incl taxes"},
            "D06": {"status": "PRESENT", "confidence": 0.97, "raw_text": "care@himalayawellness.com, 1800-208-1930"},
            "D07": {"status": "PRESENT", "confidence": 0.95, "raw_text": "India"}
        },
        d01_values={"entity_name": "The Himalaya Drug Company", "address_markers": ["Makali", "Bengaluru"], "pincode": "562162"},
        d02_values={"common_name": "Neem Face Wash"},
        d03_values={"quantity": 150.0, "unit": "ml", "unit_category": "VOLUME"},
        d04_values={"month": 11, "year": 2025, "raw_date": "11/2025"},
        d05_values={"mrp": 195.0, "tax_inclusive": True, "currency": "INR"},
        d06_values={"email": "care@himalayawellness.com", "phone": "1800-208-1930", "has_toll_free": True},
        d07_values={"country": "India", "is_domestic": True},
        packaging_features={"yolo_region_count": 5, "yolo_class_counts": {"mrp details": 1, "weight": 1, "address": 1}, "yolo_total_bbox_area": 0.08},
        ocr_quality=0.95,
        text_length=120,
        image_count=1
    )


# 1. Feature Extractor Sets A-E
def test_feature_extractor_sets_a_to_e(mock_record):
    fe = ProductComplianceFeatureExtractor()
    expected_counts = {"SET_A": 16, "SET_B": 40, "SET_C": 46, "SET_D": 56, "SET_E": 61}
    for fset, exp_cnt in expected_counts.items():
        names = fe.get_feature_names(fset)
        assert len(names) == exp_cnt, f"Feature count mismatch for {fset}: expected {exp_cnt}, got {len(names)}"
        X = fe.transform([mock_record], feature_set=fset)
        assert X.shape == (1, exp_cnt), f"Matrix shape mismatch for {fset}: {X.shape}"


# 2. Feature Extractor Leakage Protection
def test_feature_extractor_leakage():
    fe = ProductComplianceFeatureExtractor()
    for fset in ["SET_A", "SET_B", "SET_C", "SET_D", "SET_E"]:
        names = fe.get_feature_names(fset)
        for name in names:
            n_low = name.lower()
            assert "target" not in n_low, f"Target leakage in {fset}: {name}"
            assert "label" not in n_low, f"Label leakage in {fset}: {name}"
            assert "ground_truth" not in n_low, f"Ground truth leakage in {fset}: {name}"


# 3. Serialization and Deserialization
def test_common_product_representation_serialization(mock_record):
    d = mock_record.to_dict()
    assert isinstance(d, dict)
    reconstructed = ProductComplianceTrainingRecord.from_dict(d)
    assert reconstructed.product_id == mock_record.product_id
    assert reconstructed.overall_target == mock_record.overall_target
    assert reconstructed.d05_values["mrp"] == 195.0


# 4. Target Provenance Integrity
def test_dataset_provenance_integrity():
    valid_provenance = {
        TargetProvenance.HUMAN_VERIFIED.value,
        TargetProvenance.WEAKLY_DERIVED_SILVER.value,
        TargetProvenance.YOLO_REGION_ANNOTATION.value,
        TargetProvenance.UNKNOWN.value,
        "PHYSICAL_PACKAGING_INFERENCE"
    }
    training_file = PROJECT_ROOT / "data/processed/product_compliance_training.jsonl"
    if training_file.exists():
        with open(training_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    prov = r.get("target_provenance")
                    assert prov in valid_provenance, f"Invalid provenance {prov} in {r.get('product_id')}"


# 5. Immutable Benchmarks Integrity
def test_immutable_benchmarks_unmodified():
    p75 = PROJECT_ROOT / "data/processed/pilot_ground_truth_75_v1.jsonl"
    assert p75.exists(), "pilot_ground_truth_75_v1.jsonl missing"
    h = hashlib.sha256()
    with open(p75, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    assert h.hexdigest() == "37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520", "SHA-256 mismatch on 75 benchmark!"


# 6. Optical Text Extraction Invariance (Tata 1mg regression)
def test_yolo_detection_invariance_tata_1mg():
    from src.extraction.declaration_extractor import DeclarationExtractor
    extractor = DeclarationExtractor()
    ocr_text = "Mfd by: Tata 1mg Healthcare Solutions Pvt Ltd, Gurgaon. Generic Name: Paracetamol Tablets 500mg. Net Qty: 10 tablets. Date: 05/2026. MRP Rs. 35.00 incl taxes. Care: contact@1mg.com. Made in India."
    decls = extractor.extract_from_text_and_vision(
        raw_ocr_text=ocr_text,
        visual_regions=[]  # YOLO = 0 (Detector miss)
    )
    # Optical declarations must survive
    for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
        assert decls[did].status == DeclarationStatus.PRESENT.value, f"Declaration {did} failed to survive YOLO=0"


# 7. Multi-Image Evidence Fusion
def test_multi_image_evidence_fusion():
    from src.extraction.evidence_fusion import EvidenceFusionEngine
    fusion = EvidenceFusionEngine()
    
    # Image 1 has D01, D02, D05
    rec1 = ProductEvidenceRecord(
        product_id="FUSION_TEST",
        image_count=1,
        source_images=["front.png"],
        declarations={
            "D01": DeclarationEvidence(declaration_id="D01", declaration_name="Mfg", status=DeclarationStatus.PRESENT.value, evidence_text="Mfg Company A"),
            "D02": DeclarationEvidence(declaration_id="D02", declaration_name="Commodity", status=DeclarationStatus.PRESENT.value, evidence_text="Toothpaste"),
            "D05": DeclarationEvidence(declaration_id="D05", declaration_name="MRP", status=DeclarationStatus.PRESENT.value, evidence_text="MRP Rs 99.00")
        }
    )
    # Image 2 has D03, D04, D06, D07
    rec2 = ProductEvidenceRecord(
        product_id="FUSION_TEST",
        image_count=1,
        source_images=["back.png"],
        declarations={
            "D03": DeclarationEvidence(declaration_id="D03", declaration_name="Net Qty", status=DeclarationStatus.PRESENT.value, evidence_text="100g"),
            "D04": DeclarationEvidence(declaration_id="D04", declaration_name="Date", status=DeclarationStatus.PRESENT.value, evidence_text="01/2026"),
            "D06": DeclarationEvidence(declaration_id="D06", declaration_name="Care", status=DeclarationStatus.PRESENT.value, evidence_text="care@company.com"),
            "D07": DeclarationEvidence(declaration_id="D07", declaration_name="Origin", status=DeclarationStatus.PRESENT.value, evidence_text="India")
        }
    )

    fused = fusion.fuse_product_evidence([rec1, rec2], product_id="FUSION_TEST")
    assert fused.image_count == 2
    present_decls = [did for did, d in fused.declarations.items() if d.status == DeclarationStatus.PRESENT.value]
    assert len(present_decls) >= 7, f"Multi-image fusion missed declarations: {present_decls}"


# 8. Value Preservation
def test_value_preservation_mrp_quantity_dates(mock_record):
    assert mock_record.d05_values["mrp"] == 195.0
    assert mock_record.d05_values["tax_inclusive"] is True
    assert mock_record.d03_values["quantity"] == 150.0
    assert mock_record.d03_values["unit"] == "ml"
    assert mock_record.d04_values["year"] == 2025


# 9. Abstention and Confidence Calibration
def test_abstention_and_confidence_calibration():
    fe = ProductComplianceFeatureExtractor()
    model = ProductComplianceModel(
        base_estimator=None,
        feature_set="SET_A",
        feature_names=fe.get_feature_names("SET_A"),
        classes=["COMPLIANT", "NON_COMPLIANT"],
        abstention_threshold=0.65,
        margin_threshold=0.20
    )
    # Ambiguous features
    features = np.zeros(16)
    pred = model.predict_single(features, completeness_ratio=0.10, ocr_confidence=0.30)
    assert pred.final_prediction == "UNCERTAIN"
    assert pred.abstention_applied is True


# 10. Three-Class Output Guarantee
def test_three_class_outputs(mock_record):
    predictor = ProductCompliancePredictor()
    res = predictor.predict_record(mock_record, apply_abstention=True)
    assert res["overall_prediction"] in ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"]
    assert 0.0 <= res["confidence"] <= 1.0


# 11. Model Registry Persistence
def test_model_registry_and_persistence():
    reg_path = PROJECT_ROOT / "models/product_compliance/model_registry.json"
    assert reg_path.exists()
    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    assert "best_model_id" in reg
    assert "best_model_path" in reg
    assert Path(reg["best_model_path"]).exists()


# 12. Streamlit Inference Bridge
def test_streamlit_inference_bridge():
    bridge = InferencePipelineBridge.get_instance()
    assert bridge is not None
    assert hasattr(bridge, "run_inference")


# 13. Developer Mode Trace Integrity
def test_developer_mode_trace(mock_record):
    predictor = ProductCompliancePredictor()
    res = predictor.predict_record(mock_record, apply_abstention=True)
    assert "features" in res
    assert "decision_basis" in res
    assert len(res["features"]) > 0


# 14. Zero Label Fabrication
def test_zero_label_fabrication():
    training_file = PROJECT_ROOT / "data/processed/product_compliance_training.jsonl"
    with open(training_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["source_dataset"] == "flipkart":
                    assert r["target_provenance"] == "WEAKLY_DERIVED_SILVER"
                elif r["source_dataset"] == "yolo_physical_packaging":
                    assert r["target_provenance"] in ["WEAKLY_DERIVED_SILVER", "YOLO_REGION_ANNOTATION", "UNKNOWN"]


# 15. Real Packaging Non-Compliant Detection
def test_false_negative_rate_safety():
    eval_json = PROJECT_ROOT / "reports/product_compliance/evaluation_summary.json"
    if eval_json.exists():
        with open(eval_json, "r", encoding="utf-8") as f:
            ev = json.load(f)
        p3 = ev.get("partitions", {}).get("real_packaging_abstained", {})
        if p3:
            assert p3.get("non_compliant_fnr", 0.0) == 0.0, f"False Negative Rate is non-zero: {p3.get('non_compliant_fnr')}"


# 16. Canonical Uncertainty Schema Regression Test
def test_uncertainty_canonical_schema(mock_record):
    predictor = ProductCompliancePredictor()
    res = predictor.predict_record(mock_record, apply_abstention=True)
    unc = res.get("uncertainty")
    assert isinstance(unc, dict), f"uncertainty must be a dict, got {type(unc)}"
    assert "status" in unc, "uncertainty missing 'status'"
    assert unc["status"] in ["LOW", "MEDIUM", "HIGH"], f"Invalid status: {unc['status']}"
    assert "score" in unc, "uncertainty missing 'score'"
    assert isinstance(unc["score"], (float, int)), f"score must be float, got {type(unc['score'])}"
    assert "reason" in unc, "uncertainty missing 'reason'"
    assert isinstance(unc["reason"], str)
    assert "abstained" in unc, "uncertainty missing 'abstained'"
    assert isinstance(unc["abstained"], bool)


# 17. CASE A — NO EVIDENCE (YOLO=0, OCR="", D01-D07=CANNOT_DETERMINE)
def test_controlled_case_a_no_evidence():
    from src.extraction.declaration_extractor import DeclarationExtractor
    extractor = DeclarationExtractor()
    decls = extractor.extract_from_text_and_vision(
        raw_ocr_text="",
        visual_regions=[]
    )
    for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
        assert decls[did].status == DeclarationStatus.CANNOT_DETERMINE.value, f"{did} should be CANNOT_DETERMINE"

    empty_record = ProductComplianceTrainingRecord(
        product_id="CASE_A_NO_EV",
        source_dataset="inference",
        target_provenance="PHYSICAL_PACKAGING_INFERENCE",
        overall_target="UNKNOWN",
        declarations={did: {"status": "CANNOT_DETERMINE", "confidence": 0.0, "raw_text": ""} for did in decls},
        ocr_quality=0.10,
        text_length=0,
        image_count=1
    )
    predictor = ProductCompliancePredictor()
    res = predictor.predict_record(empty_record, apply_abstention=True)
    # Must enter uncertainty/abstention pathway
    assert res["overall_prediction"] == "UNCERTAIN"
    assert res["uncertainty"]["status"] == "HIGH"
    assert res["uncertainty"]["abstained"] is True


# 18. CASE B — POSITIVE OCR, YOLO=0
def test_controlled_case_b_positive_ocr_yolo_zero():
    from src.extraction.declaration_extractor import DeclarationExtractor
    extractor = DeclarationExtractor()
    ocr_text = (
        "Manufactured by: ABC Healthcare Ltd, 123 Industrial Area, Mumbai 400001, Maharashtra, India.\n"
        "Product Name: Almond Oil 250ml\n"
        "Net Quantity: 250 g\n"
        "Mfg Date: 02/2026\n"
        "MRP: Rs. 1063.00 incl. of all taxes\n"
        "Consumer Care: Phone: 1800-222-333, Email: care@abchealth.com\n"
        "Country of Origin: India"
    )
    decls = extractor.extract_from_text_and_vision(
        raw_ocr_text=ocr_text,
        visual_regions=[]  # YOLO = 0
    )
    # Declarations must survive
    for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
        assert decls[did].status == DeclarationStatus.PRESENT.value, f"Declaration {did} lost when YOLO=0"

    # Actual values must be preserved
    assert decls["D05"].normalized_value.get("mrp") == 1063.0
    assert decls["D03"].normalized_value.get("quantity") == 250.0
    assert decls["D04"].normalized_value.get("month") == 2
    assert decls["D04"].normalized_value.get("year") == 2026


# 19. CASE C — YOLO POSITIVE, OCR EMPTY
def test_controlled_case_c_yolo_positive_ocr_empty():
    from src.extraction.declaration_extractor import DeclarationExtractor
    extractor = DeclarationExtractor()
    visual_dets = [
        {"class_name": "mrp details", "bbox": [10, 10, 50, 50], "confidence": 0.90},
        {"class_name": "weight", "bbox": [50, 50, 80, 80], "confidence": 0.88},
        {"class_name": "address", "bbox": [80, 80, 120, 120], "confidence": 0.85}
    ]
    decls = extractor.extract_from_text_and_vision(
        raw_ocr_text="",
        visual_regions=visual_dets
    )
    # YOLO detections alone are spatial/localization evidence, NOT affirmative declaration content
    for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
        assert decls[did].status == DeclarationStatus.CANNOT_DETERMINE.value, f"{did} should remain CANNOT_DETERMINE without text"


# 20. CASE D — CONFLICT (Image 1 MRP ₹999, Image 2 MRP ₹1099)
def test_controlled_case_d_conflict():
    from src.extraction.evidence_fusion import EvidenceFusionEngine
    fusion = EvidenceFusionEngine()

    ev1 = {
        "image_name": "img_01.png",
        "ocr_text": "MRP Rs. 999.00 incl taxes",
        "ocr_confidence": 0.95,
        "declarations": {
            "D05": DeclarationEvidence(
                declaration_id="D05",
                declaration_name="MRP",
                status=DeclarationStatus.PRESENT.value,
                evidence_text="MRP Rs. 999.00",
                raw_value="999.00",
                normalized_value={"mrp": 999.0, "tax_inclusive": True},
                source="OCR",
                source_image="img_01.png"
            )
        }
    }

    ev2 = {
        "image_name": "img_02.png",
        "ocr_text": "MRP Rs. 1099.00 incl taxes",
        "ocr_confidence": 0.95,
        "declarations": {
            "D05": DeclarationEvidence(
                declaration_id="D05",
                declaration_name="MRP",
                status=DeclarationStatus.PRESENT.value,
                evidence_text="MRP Rs. 1099.00",
                raw_value="1099.00",
                normalized_value={"mrp": 1099.0, "tax_inclusive": True},
                source="OCR",
                source_image="img_02.png"
            )
        }
    }

    fused = fusion.fuse_image_evidences([ev1, ev2], product_id="CONFLICT_TEST")
    d05 = fused.declarations["D05"]
    assert d05.status == DeclarationStatus.CONFLICTING_EVIDENCE.value, f"D05 status should be CONFLICTING_EVIDENCE, got {d05.status}"
    assert "conflicting_values" in d05.normalized_value or d05.conflicting_values
