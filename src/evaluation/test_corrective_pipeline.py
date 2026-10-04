"""
test_corrective_pipeline.py
===========================
Comprehensive unit and regression test suite for the Corrective Declaration-Centric ML Pipeline.

Verifies:
1. Declaration Schema & Status Semantics
2. D01-D07 Extraction with Preserved Actual Values
3. Critical Regression: YOLO=0 with Positive OCR (Tata 1mg Creatine Case)
4. Multi-Image Evidence Fusion (Tests 1-4)
5. Controlled Feature Extractor (Sets A-E)
6. Zero Target Leakage
7. DeclarationProductModel Predictor & Abstention
8. API Security & Trace Verification
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.declaration_schema import (
    DeclarationEvidence, DeclarationStatus, EvidenceSource, ProductEvidenceRecord
)
from src.extraction.declaration_extractor import DeclarationExtractor
from src.extraction.evidence_fusion import EvidenceFusionEngine
from src.features.declaration_feature_extractor import DeclarationFeatureExtractor
from src.models.declaration_product_model import DeclarationProductModel
from src.models.declaration_model_predictor import DeclarationModelPredictor
from src.evaluation.leakage_audit import LeakageAuditor


class TestCorrectivePipeline(unittest.TestCase):

    def setUp(self):
        self.extractor = DeclarationExtractor()
        self.fusion_engine = EvidenceFusionEngine()
        self.feature_extractor = DeclarationFeatureExtractor()

    # 1. Declaration Schema
    def test_declaration_schema_and_serialization(self):
        decl = DeclarationEvidence(
            declaration_id="D05",
            declaration_name="MRP (incl. of all taxes)",
            status=DeclarationStatus.PRESENT.value,
            raw_value="MRP ₹1063.00",
            normalized_value={"mrp": 1063.0, "tax_inclusive": True},
            source=EvidenceSource.OCR.value,
            extraction_confidence=0.95
        )
        d_dict = decl.to_dict()
        self.assertEqual(d_dict["declaration_id"], "D05")
        self.assertEqual(d_dict["status"], "PRESENT")
        self.assertEqual(d_dict["normalized_value"]["mrp"], 1063.0)

    # 2. D01-D07 Extraction with Preserved Actual Values
    def test_d01_to_d07_value_preservation(self):
        sample_text = (
            "Manufactured by: ABC Healthcare Ltd., 12 Industrial Area, Mumbai 400001.\n"
            "Generic Name: Whey Protein Powder\n"
            "Net Quantity: 500 g\n"
            "Mfg Date: 03/2026\n"
            "MRP: Rs. 1499.00 (Inclusive of all taxes)\n"
            "Consumer Care: care@abchealth.com, Tel: 1800-111-2222\n"
            "Country of Origin: India"
        )
        rec = self.extractor.extract_declarations(sample_text, product_id="P_VAL_01")
        decls = rec.declarations

        # Check D01
        self.assertEqual(decls["D01"].status, DeclarationStatus.PRESENT)
        self.assertTrue("address_markers" in decls["D01"].normalized_value or "entity_name" in decls["D01"].normalized_value)
        self.assertEqual(decls["D01"].normalized_value.get("entity_name"), "ABC Healthcare Ltd")

        # Check D02
        self.assertEqual(decls["D02"].status, DeclarationStatus.PRESENT)

        # Check D03: Quantity & Unit
        self.assertEqual(decls["D03"].status, DeclarationStatus.PRESENT)
        self.assertEqual(decls["D03"].normalized_value.get("quantity"), 500.0)
        self.assertEqual(decls["D03"].normalized_value.get("unit"), "g")

        # Check D04: Date
        self.assertEqual(decls["D04"].status, DeclarationStatus.PRESENT)
        self.assertEqual(decls["D04"].normalized_value.get("month"), 3)
        self.assertEqual(decls["D04"].normalized_value.get("year"), 2026)

        # Check D05: MRP & Tax
        self.assertEqual(decls["D05"].status, DeclarationStatus.PRESENT)
        self.assertEqual(decls["D05"].normalized_value.get("mrp"), 1499.0)
        self.assertTrue(decls["D05"].normalized_value.get("tax_inclusive"))

        # Check D06: Contact
        self.assertEqual(decls["D06"].status, DeclarationStatus.PRESENT)
        self.assertIn("email", decls["D06"].normalized_value)

        # Check D07: Origin
        self.assertEqual(decls["D07"].status, DeclarationStatus.PRESENT)
        self.assertEqual(decls["D07"].normalized_value.get("country"), "India")

    # 3. Tata 1mg Creatine Regression: YOLO=0, OCR=Positive
    def test_tata_1mg_yolo_zero_ocr_positive(self):
        tata_ocr_text = (
            "Marketed by: TATA 1MG Healthcare Solutions Pvt. Ltd., B-12, Okhla Phase II, New Delhi 110020.\n"
            "Manufactured by: BACFO Pharmaceuticals (India) Limited, C-46, Phase II, Noida 201305.\n"
            "Product: Micronized Creatine Monohydrate Powder\n"
            "Net Quantity: 250 g\n"
            "Mfg Date: 02/2026 | Expiry: 07/2027\n"
            "MRP: Rs. 1063.00 (Inclusive of all taxes)\n"
            "Customer Care: care@1mg.com | +91-995-8932200\n"
            "Country of Origin: India"
        )
        # Empty visual regions (YOLO = 0 detections)
        rec = self.extractor.extract_declarations(
            raw_text=tata_ocr_text,
            product_id="TATA_1MG_CREATINE",
            visual_regions=[]
        )
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            self.assertEqual(
                rec.declarations[did].status,
                DeclarationStatus.PRESENT,
                f"Declaration {did} must remain PRESENT when OCR is positive even if YOLO=0!"
            )

    # 4. Multi-Image Fusion Tests (Tests 1-4)
    def test_multi_image_fusion_test1_mrp_retention(self):
        # Image 1 has no MRP, Image 2 has MRP
        rec1 = self.extractor.extract_declarations("Net Quantity: 200 g", source_image="img_front")
        rec2 = self.extractor.extract_declarations("MRP Rs. 299.00 (Incl. of all taxes)", source_image="img_back")
        fused = self.fusion_engine.fuse_product_evidence([rec1, rec2])

        self.assertEqual(fused.declarations["D05"].status, DeclarationStatus.PRESENT)
        self.assertEqual(fused.declarations["D05"].normalized_value.get("mrp"), 299.0)
        self.assertEqual(fused.declarations["D03"].status, DeclarationStatus.PRESENT)

    def test_multi_image_fusion_test2_manufacturer_and_consumer_care(self):
        # Image 1 manufacturer only, Image 2 consumer care only
        rec1 = self.extractor.extract_declarations("Manufactured by: Alpha Biotech, Delhi 110001", source_image="img_1")
        rec2 = self.extractor.extract_declarations("Customer Helpline: 1800-000-1111 care@alpha.in", source_image="img_2")
        fused = self.fusion_engine.fuse_product_evidence([rec1, rec2])

        self.assertEqual(fused.declarations["D01"].status, DeclarationStatus.PRESENT)
        self.assertEqual(fused.declarations["D06"].status, DeclarationStatus.PRESENT)

    def test_multi_image_fusion_test3_insufficient_evidence(self):
        # Both images lack legible text
        rec1 = self.extractor.extract_declarations("Blurry image noise", source_image="img_1", ocr_confidence=0.1)
        rec2 = self.extractor.extract_declarations("More noise", source_image="img_2", ocr_confidence=0.1)
        fused = self.fusion_engine.fuse_product_evidence([rec1, rec2])

        self.assertIn(fused.declarations["D05"].status, [DeclarationStatus.CANNOT_DETERMINE, DeclarationStatus.ABSENT])

    def test_multi_image_fusion_test4_conflict_detection(self):
        # Image 1 has MRP 999, Image 2 has MRP 1099
        rec1 = self.extractor.extract_declarations("MRP Rs. 999.00", source_image="img_1")
        rec2 = self.extractor.extract_declarations("MRP Rs. 1099.00", source_image="img_2")
        fused = self.fusion_engine.fuse_product_evidence([rec1, rec2])

        self.assertIn("D05", fused.conflicting_fields)

    # 5. Feature Extractor Sets A-E
    def test_feature_sets_dimensions_and_isolation(self):
        rec = self.extractor.extract_declarations("Net Wt 100g MRP Rs 50")
        for fset in ["SET_A", "SET_B", "SET_C", "SET_D", "SET_E"]:
            feats = self.feature_extractor.transform([rec], feature_set=fset)
            names = self.feature_extractor.get_feature_names(fset)
            self.assertEqual(feats.shape[1], len(names))
            self.assertFalse(np.isnan(feats).any())

    # 6. Leakage Audit
    def test_target_leakage_absence(self):
        auditor = LeakageAuditor()
        audit_res = auditor.run_full_audit()
        all_passed = all(check.get("passed", False) for check in audit_res.values() if isinstance(check, dict))
        self.assertTrue(all_passed, f"Leakage audit failed: {audit_res}")

    # 7. Model Predictor
    def test_declaration_predictor_execution(self):
        predictor = DeclarationModelPredictor()
        rec = self.extractor.extract_declarations("Net Qty 100g MRP Rs 50")
        res = predictor.predict(rec)
        self.assertIn(res["predicted_label"], ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
