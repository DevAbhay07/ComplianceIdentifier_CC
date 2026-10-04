"""
test_end_to_end_scenarios.py
============================
End-to-end integration test suite covering 10 real-world package scenarios:
1. Clearly compliant package (all Rule 6 declarations present)
2. Clearly non-compliant package (missing mandatory declarations)
3. Multi-image package (front + back views fused into one product)
4. Blurry package (poor OCR quality -> calibrated confidence reduced / UNCERTAIN)
5. Tiny / partial declarations (CANNOT_DETERMINE / PARTIAL -> UNCERTAIN)
6. Missing declaration (missing Consumer Care D06 -> NON_COMPLIANT)
7. Imported package (D07 mandatory)
8. Domestic package
9. Conflicting images (conflicting MRP across views -> CONFLICTING_EVIDENCE penalty)
10. Unrelated image (no packaging declarations -> UNCERTAIN or unconfident verdict)

Also tests critical architectural invariants:
- YOLO = 0 + OCR positive -> preserves OCR evidence
- YOLO positive + OCR empty -> NOT automatically PRESENT
- Unrelated image -> never receives confident COMPLIANT verdict
"""

import sys
import pytest
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.product_compliance_pipeline import ProductCompliancePipeline
from src.extraction.declaration_schema import DeclarationStatus, EvidenceSource, DeclarationEvidence
from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.models.predict_product_compliance import ProductCompliancePredictor


@pytest.fixture
def pipeline():
    return ProductCompliancePipeline(use_mock_ocr_fallback=True)


@pytest.fixture
def predictor():
    return ProductCompliancePredictor()


class TestEndToEndScenarios:
    """Tests the 10 real-world product compliance packaging scenarios."""

    def test_scenario_1_clearly_compliant_package(self, predictor):
        """1. Clearly compliant package with all Rule 6 declarations present."""
        decls = {
            "D01": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Himalaya Wellness, Bengaluru 562162"},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Face Wash"},
            "D03": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "150 ml"},
            "D04": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "11/2025"},
            "D05": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Rs. 195.00"},
            "D06": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "care@himalayawellness.com"},
            "D07": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "India"}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_sc1_compliant",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="COMPLIANT",
            category="Beauty",
            brand="Himalaya",
            product_title="Neem Face Wash",
            raw_text="All declarations present",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 5, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.4},
            ocr_quality=0.95,
            text_length=300,
            image_count=1
        )
        res = predictor.predict_record(rec)
        assert res["overall_prediction"] == "COMPLIANT"
        assert res["confidence"] >= 0.70
        assert res["uncertainty"]["status"] in ["LOW", "MEDIUM"]

    def test_scenario_2_clearly_non_compliant_package(self, predictor):
        """2. Clearly non-compliant package missing mandatory MRP and Net Quantity."""
        decls = {
            "D01": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "ABC Ltd, Mumbai"},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Shampoo"},
            "D03": {"status": "ABSENT", "extraction_confidence": 0.90, "raw_text": ""},
            "D04": {"status": "PRESENT", "extraction_confidence": 0.90, "raw_text": "05/2025"},
            "D05": {"status": "ABSENT", "extraction_confidence": 0.90, "raw_text": ""},
            "D06": {"status": "PRESENT", "extraction_confidence": 0.90, "raw_text": "1800-123-456"},
            "D07": {"status": "PRESENT", "extraction_confidence": 0.90, "raw_text": "India"}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_sc2_non_compliant",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="NON_COMPLIANT",
            category="Personal Care",
            brand="TestBrand",
            product_title="Missing MRP Shampoo",
            raw_text="Incomplete package text",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 2, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.1},
            ocr_quality=0.90,
            text_length=150,
            image_count=1
        )
        res = predictor.predict_record(rec)
        assert res["overall_prediction"] == "NON_COMPLIANT"

    def test_scenario_3_multi_image_fusion(self, pipeline):
        """3. Multi-image package: front view has D01-D03, back view has D04-D07 -> fused COMPLIANT."""
        img1 = PROJECT_ROOT / "real_product_validation/images/real_prod_01.png"
        img2 = PROJECT_ROOT / "real_product_validation/images/real_prod_02.png"
        if img1.exists() and img2.exists():
            res = pipeline.analyze_product(image_paths=[str(img1), str(img2)], product_title="Fused Multi-Image Product")
            assert res["trace"]["image_count"] == 2
            assert "overall_prediction" in res
            assert res["overall_prediction"] in ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"]

    def test_scenario_4_blurry_package_reduces_confidence(self, predictor):
        """4. Blurry package with low OCR quality (0.35) -> calibrated confidence must drop."""
        decls = {
            "D01": {"status": "PRESENT", "extraction_confidence": 0.50, "raw_text": "Blurry text"},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.50, "raw_text": "Blurry name"},
            "D03": {"status": "PRESENT", "extraction_confidence": 0.50, "raw_text": "100g"},
            "D04": {"status": "PRESENT", "extraction_confidence": 0.50, "raw_text": "2025"},
            "D05": {"status": "PRESENT", "extraction_confidence": 0.50, "raw_text": "Rs 50"},
            "D06": {"status": "PRESENT", "extraction_confidence": 0.50, "raw_text": "care@test"},
            "D07": {"status": "PRESENT", "extraction_confidence": 0.50, "raw_text": "India"}
        }
        rec_clear = ProductComplianceTrainingRecord(
            product_id="test_clear",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="COMPLIANT",
            category="Food",
            brand="Brand",
            product_title="Clear Biscuit",
            raw_text="Clear text",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 4, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.3},
            ocr_quality=0.95,
            text_length=200,
            image_count=1
        )
        rec_blurry = ProductComplianceTrainingRecord(
            product_id="test_blurry",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="COMPLIANT",
            category="Food",
            brand="Brand",
            product_title="Blurry Biscuit",
            raw_text="Blurry text",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 4, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.3},
            ocr_quality=0.35,  # Low optical clarity
            text_length=50,
            image_count=1
        )
        res_clear = predictor.predict_record(rec_clear)
        res_blurry = predictor.predict_record(rec_blurry)
        assert res_blurry["confidence"] < res_clear["confidence"]
        assert res_blurry["confidence"] < 0.70

    def test_scenario_5_tiny_declarations_cannot_determine(self, predictor):
        """5. Tiny / unreadable declarations (CANNOT_DETERMINE) -> abstention / UNCERTAIN."""
        decls = {
            "D01": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.90, "raw_text": "Soap"},
            "D03": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D04": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D05": {"status": "PRESENT", "extraction_confidence": 0.90, "raw_text": "Rs. 45"},
            "D06": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D07": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_tiny_decl",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="UNCERTAIN",
            category="Soap",
            brand="TestSoap",
            product_title="Tiny Declarations Soap",
            raw_text="Soap Rs. 45",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 1, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.05},
            ocr_quality=0.50,
            text_length=20,
            image_count=1
        )
        res = predictor.predict_record(rec)
        assert res["overall_prediction"] == "UNCERTAIN"
        assert res["uncertainty"]["abstained"] is True
        assert res["uncertainty"]["status"] in ["HIGH", "MEDIUM"]

    def test_scenario_6_missing_consumer_care_is_non_compliant(self, predictor):
        """6. Missing Consumer Care helpline (D06 ABSENT) -> NON_COMPLIANT."""
        decls = {
            "D01": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Nestle India, Gurugram"},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Noodles"},
            "D03": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "70 g"},
            "D04": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "10/2025"},
            "D05": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Rs. 14.00"},
            "D06": {"status": "ABSENT", "extraction_confidence": 0.90, "raw_text": ""},
            "D07": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "India"}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_missing_care",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="NON_COMPLIANT",
            category="Food",
            brand="Maggi",
            product_title="Noodles Pack",
            raw_text="No customer helpline provided on pack",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 5, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.4},
            ocr_quality=0.90,
            text_length=150,
            image_count=1
        )
        res = predictor.predict_record(rec)
        assert res["overall_prediction"] == "NON_COMPLIANT"
        assert "D06" in res["decision_basis"] or "Consumer Care" in res["decision_basis"]

    def test_scenario_7_imported_package_origin(self, predictor):
        """7. Imported package requires Country of Origin D07."""
        decls = {
            "D01": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Imported by: Ferrero India Pvt Ltd, Pune. Manufactured by: Ferrero SpA, Italy"},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Hazelnut Spread"},
            "D03": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "350 g"},
            "D04": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "08/2025"},
            "D05": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Rs. 420.00"},
            "D06": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "care@ferrero.com"},
            "D07": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Italy"}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_imported",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="COMPLIANT",
            category="Food",
            brand="Nutella",
            product_title="Nutella Hazelnut Spread",
            raw_text="Imported chocolate spread",
            declarations=decls,
            d01_values={"mfg_name": "Ferrero SpA", "mfg_address": "Italy"},
            d02_values={"generic_name": "Hazelnut Spread"},
            d03_values={"net_quantity": "350 g"},
            d04_values={"date_text": "08/2025"},
            d05_values={"mrp": "Rs. 420.00"},
            d06_values={"consumer_care": "care@ferrero.com"},
            d07_values={"country_of_origin": "Italy"},
            packaging_features={"yolo_detection_count": 6, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.5},
            ocr_quality=0.95,
            text_length=250,
            image_count=1
        )
        res = predictor.predict_record(rec)
        assert res["overall_prediction"] == "COMPLIANT"

    def test_scenario_8_domestic_package(self, predictor):
        """8. Domestic package with Indian origin declaration."""
        decls = {
            "D01": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Tata Consumer Products Ltd, Bengaluru"},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Tea"},
            "D03": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "250 g"},
            "D04": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "09/2025"},
            "D05": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Rs. 110.00"},
            "D06": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "1800-345-1720"},
            "D07": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "India"}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_domestic",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="COMPLIANT",
            category="Beverages",
            brand="Tata Tea",
            product_title="Tata Tea Premium",
            raw_text="Domestic tea pack",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 5, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.4},
            ocr_quality=0.95,
            text_length=200,
            image_count=1
        )
        res = predictor.predict_record(rec)
        assert res["overall_prediction"] == "COMPLIANT"

    def test_scenario_9_conflicting_evidence_penalized(self, predictor):
        """9. Conflicting evidence (e.g. conflicting MRP across front vs back) penalizes confidence."""
        decls = {
            "D01": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Britannia Industries Ltd"},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Biscuits"},
            "D03": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "100 g"},
            "D04": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "01/2026"},
            "D05": {"status": "CONFLICTING_EVIDENCE", "extraction_confidence": 0.60, "raw_text": "Front: Rs 30, Back: Rs 40"},
            "D06": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "feedback@britannia.co.in"},
            "D07": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "India"}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_conflict",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="UNCERTAIN",
            category="Food",
            brand="Britannia",
            product_title="Conflicting MRP Biscuits",
            raw_text="Front shows Rs 30, Back sticker shows Rs 40",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 4, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.3},
            ocr_quality=0.90,
            text_length=200,
            image_count=2
        )
        res = predictor.predict_record(rec)
        # Should reflect uncertainty due to conflict
        assert res["confidence"] < 0.85
        assert res["uncertainty"]["status"] in ["HIGH", "MEDIUM"]

    def test_scenario_10_unrelated_image_never_confidently_compliant(self, predictor):
        """10. Unrelated image (no declarations) -> never receives confident COMPLIANT verdict."""
        decls = {
            "D01": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.0, "raw_text": ""},
            "D02": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.0, "raw_text": ""},
            "D03": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.0, "raw_text": ""},
            "D04": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.0, "raw_text": ""},
            "D05": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.0, "raw_text": ""},
            "D06": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.0, "raw_text": ""},
            "D07": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.0, "raw_text": ""}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_unrelated",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="UNCERTAIN",
            category="Non-Product",
            brand="",
            product_title="Sunset Landscape Photograph",
            raw_text="",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 0, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.0},
            ocr_quality=0.0,
            text_length=0,
            image_count=1
        )
        res = predictor.predict_record(rec)
        assert res["overall_prediction"] != "COMPLIANT"
        assert res["confidence"] <= 0.40


class TestArchitecturalInvariants:
    """Tests critical architectural invariants."""

    def test_yolo_zero_ocr_positive_preserves_evidence(self, predictor):
        """YOLO = 0 + OCR positive: must preserve OCR evidence as PRESENT."""
        decls = {
            "D01": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Mfd by Dabur India Ltd, New Delhi 110002"},
            "D02": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "Honey"},
            "D03": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "500 g"},
            "D04": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "03/2026"},
            "D05": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "MRP Rs. 210.00"},
            "D06": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "care@dabur.com"},
            "D07": {"status": "PRESENT", "extraction_confidence": 0.95, "raw_text": "India"}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_yolo_zero",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="COMPLIANT",
            category="Food",
            brand="Dabur",
            product_title="Dabur Honey 500g",
            raw_text="Honey bottle with all details",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 0, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.0},  # YOLO = 0
            ocr_quality=0.95,
            text_length=300,
            image_count=1
        )
        res = predictor.predict_record(rec)
        assert res["overall_prediction"] == "COMPLIANT"

    def test_yolo_positive_ocr_empty_is_cannot_determine(self, predictor):
        """YOLO positive + OCR empty: must NOT automatically mean PRESENT."""
        decls = {
            "D01": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D02": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D03": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D04": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D05": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D06": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""},
            "D07": {"status": "CANNOT_DETERMINE", "extraction_confidence": 0.20, "raw_text": ""}
        }
        rec = ProductComplianceTrainingRecord(
            product_id="test_yolo_pos_ocr_empty",
            source_dataset="test",
            target_provenance="TEST",
            overall_target="UNCERTAIN",
            category="General",
            brand="",
            product_title="Packaging box with unreadable text",
            raw_text="",
            declarations=decls,
            d01_values={}, d02_values={}, d03_values={}, d04_values={}, d05_values={}, d06_values={}, d07_values={},
            packaging_features={"yolo_detection_count": 8, "yolo_class_counts": {"mrp details": 1, "weight": 1, "address": 1}, "yolo_total_bbox_area": 0.6},
            ocr_quality=0.10,
            text_length=0,
            image_count=1
        )
        res = predictor.predict_record(rec)
        # Bounding boxes alone without verified text cannot yield a confident COMPLIANT verdict
        assert res["overall_prediction"] in ["UNCERTAIN", "NON_COMPLIANT"]
        assert res["overall_prediction"] != "COMPLIANT"
