"""
test_inference_pipeline.py
==========================
Comprehensive 18-point unit test suite for the Phase 9 standalone inference pipeline.
Tests image validation, mock OCR, evidence extraction, feature conversion, schema validation,
model inference, uncertainty estimation, offline mode, and error handling without external API calls.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import unittest
import tempfile
import json
import pandas as pd
from PIL import Image

from src.inference.image_validator import validate_product_image
from src.inference.mock_vision_ocr import MockVisionOCRProvider
from src.inference.evidence_parser import parse_product_evidence
from src.inference.inference_features import build_inference_feature_vectors
from src.inference.schema_validator import validate_inference_feature_schema
from src.inference.model_loader import load_frozen_models
from src.inference.declaration_predictor import predict_all_declarations
from src.inference.uncertainty import compute_prediction_uncertainty
from src.inference.real_product_test import check_product_novelty
from src.inference.trace import InferenceTracer
from src.inference.predict import run_standalone_prediction


class TestInferencePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary valid image for testing
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.valid_img_path = os.path.join(cls.temp_dir.name, "test_img.png")
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        img.save(cls.valid_img_path)

        # Corrupt file
        cls.corrupt_img_path = os.path.join(cls.temp_dir.name, "corrupt.png")
        with open(cls.corrupt_img_path, "wb") as f:
            f.write(b"NOT_A_REAL_IMAGE_BYTES")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_01_image_validator_valid_image(self):
        """Test valid image passes validation with correct metadata."""
        res = validate_product_image(self.valid_img_path)
        self.assertTrue(res["valid"])
        self.assertEqual(res["format"], "PNG")
        self.assertEqual(res["width"], 200)
        self.assertEqual(res["height"], 200)
        self.assertIsNone(res["error"])

    def test_02_image_validator_nonexistent_file(self):
        """Test non-existent image path rejection."""
        res = validate_product_image("non_existent_file_path_123.jpg")
        self.assertFalse(res["valid"])
        self.assertIn("FILE_NOT_FOUND", res["error"])

    def test_03_image_validator_corrupt_file(self):
        """Test corrupt image rejection."""
        res = validate_product_image(self.corrupt_img_path)
        self.assertFalse(res["valid"])
        self.assertIn("CORRUPT_OR_UNREADABLE", res["error"])

    def test_04_mock_vision_ocr_success(self):
        """Test mock OCR provider returns text and status."""
        provider = MockVisionOCRProvider(mock_text="Sample OCR Text 100g")
        res = provider.extract_from_image(self.valid_img_path)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["raw_text"], "Sample OCR Text 100g")
        self.assertIsNone(res["error"])

    def test_05_mock_vision_ocr_error(self):
        """Test mock OCR provider simulates errors."""
        provider = MockVisionOCRProvider(mock_status="TIMEOUT", mock_error="Mock timeout error")
        res = provider.extract_from_image(self.valid_img_path)
        self.assertEqual(res["status"], "TIMEOUT")
        self.assertIn("timeout", res["error"].lower())

    def test_06_evidence_parser_declarations(self):
        """Test evidence parser extracts structured declaration spans."""
        raw_text = "Mfd by Colgate India Ltd. Net Qty: 120 g. MRP Rs. 145.00. Care: 1800-225-599. Origin: India."
        res = parse_product_evidence(raw_ocr_text=raw_text)
        decls = res["declarations"]
        self.assertTrue(decls["D01"]["evidence_detected"])
        self.assertTrue(decls["D03"]["evidence_detected"])
        self.assertTrue(decls["D05"]["evidence_detected"])
        self.assertTrue(decls["D06"]["evidence_detected"])
        self.assertTrue(decls["D07"]["evidence_detected"])

    def test_07_evidence_parser_diagnostics(self):
        """Test evidence parser diagnostics."""
        raw_text = "Net Qty: 500 ml MRP Rs. 99"
        res = parse_product_evidence(raw_ocr_text=raw_text)
        diag = res["diagnostics"]
        self.assertIn("extracted_text_length", diag)
        self.assertIn("detected_declarations_count", diag)
        self.assertIn("completeness_score", diag)
        self.assertGreater(diag["detected_declarations_count"], 0)

    def test_08_inference_features_dimensions(self):
        """Test inference feature adapter generates 219 (Set A) and 719 (Set B) columns."""
        X_a, X_b, raw_feats = build_inference_feature_vectors(
            title="Sample Toothpaste",
            description="Good toothpaste 100g",
            raw_ocr_text="MRP Rs. 99",
            retail_price=99.0
        )
        self.assertEqual(X_a.shape, (1, 219))
        self.assertEqual(X_b.shape, (1, 719))

    def test_09_schema_validator_success(self):
        """Test schema validator on valid feature vectors."""
        X_a, X_b, _ = build_inference_feature_vectors(title="Test Product")
        val = validate_inference_feature_schema(X_a, X_b)
        self.assertTrue(val["valid"])
        self.assertEqual(len(val["errors"]), 0)

    def test_10_schema_validator_failure(self):
        """Test schema validator catches column dimension mismatch."""
        bad_df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        val = validate_inference_feature_schema(bad_df, bad_df)
        self.assertFalse(val["valid"])
        self.assertGreater(len(val["errors"]), 0)

    def test_11_model_loader_structure(self):
        """Test model loader retrieves all 7 declaration models."""
        models = load_frozen_models()
        self.assertEqual(len(models), 7)
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            self.assertIn(did, models)

    def test_12_declaration_predictor_all_decls(self):
        """Test declaration predictor outputs valid predictions."""
        models = load_frozen_models()
        X_a, X_b, _ = build_inference_feature_vectors(title="Test Product")
        preds = predict_all_declarations(models, X_a, X_b)
        self.assertEqual(len(preds), 7)
        for did, p in preds.items():
            self.assertIn("prediction", p)
            self.assertIn("confidence", p)
            self.assertIn("decision", p)

    def test_13_uncertainty_metrics(self):
        """Test uncertainty metric calculation."""
        probs = {"PRESENT": 0.85, "ABSENT": 0.10, "PARTIAL": 0.05}
        unc = compute_prediction_uncertainty(probs)
        self.assertEqual(unc["confidence"], 0.85)
        self.assertEqual(unc["margin"], 0.75)
        self.assertEqual(unc["decision"], "PREDICT")
        self.assertLessEqual(unc["entropy"], 1.0)

    def test_14_standalone_prediction_offline_text(self):
        """Test offline prediction on text input."""
        res = run_standalone_prediction(
            title="Sample Biscuit 100g",
            description="Mfd by Britannia India Ltd. MRP Rs 30.00",
            offline_mode=True
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["vision_ocr"]["status"], "NOT_ATTEMPTED")
        self.assertEqual(len(res["declarations"]), 7)

    def test_15_standalone_prediction_offline_image_only_error(self):
        """Test offline mode on image-only input returns VISION_REQUIRED."""
        res = run_standalone_prediction(
            image_path=self.valid_img_path,
            title="",
            description="",
            offline_mode=True
        )
        self.assertEqual(res["status"], "VISION_REQUIRED")

    def test_16_standalone_prediction_mock_ocr_e2e(self):
        """Test end-to-end prediction with mock OCR."""
        res = run_standalone_prediction(
            image_path=self.valid_img_path,
            title="Sample Toothpaste",
            provider_name="mock"
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["vision_ocr"]["status"], "SUCCESS")
        self.assertIn("D01", res["declarations"])
        self.assertIn("trace_id", res)

    def test_17_trace_logging(self):
        """Test trace generation and logging."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracer = InferenceTracer(log_dir=tmp_dir, enabled=True)
            tracer.mark_start("test_stage")
            tracer.mark_end("test_stage")
            rec = tracer.finalize({"title": "Test"}, {}, {"D01": {}}, {})
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, f"{rec['trace_id']}.json")))

    def test_18_real_product_novelty(self):
        """Test novelty verification logic."""
        res = check_product_novelty(title="Brand New Unique Gizmo 999xyz")
        self.assertEqual(res["status"], "NEW_PRODUCT")


if __name__ == "__main__":
    unittest.main()
