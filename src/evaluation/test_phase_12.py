"""
Comprehensive Unit & Integration Test Suite for Phase 12 Streamlit ML Application.
Covers:
1. App module importability
2. Image validation (valid, invalid, corrupt, dimensions)
3. YOLO model loading & caching
4. YOLO packaging detection execution
5. YOLO bounding box drawing
6. ROI cropping
7. Vision/OCR extraction & offline mode
8. Evidence parsing (Rule 6 declarations)
9. YOLO feature extraction (29 features)
10. Multimodal feature extraction (Set D, D=741)
11. Direct ML model inference & probabilities
12. Evidence-aware statutory decision engine
13. Three-class output validity (COMPLIANT, NON_COMPLIANT, UNCERTAIN)
14. D01-D07 supporting evidence integrity
15. Uncertainty quantification
16. Developer Mode trace & latency metrics
17. JSON export schema
18. API key security & isolation
19. Zero-persistence user privacy
20. End-to-end inference execution on real packaging
"""

import os
import sys
import io
import json
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

WORKSPACE_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
sys.path.insert(0, str(WORKSPACE_ROOT))

from app.utils.validation import validate_uploaded_image
from app.utils.visualizer import draw_bounding_boxes
from app.utils.inference_bridge import InferencePipelineBridge
from src.features.yolo_feature_extractor import YOLOFeatureExtractor
from src.features.phase_11_feature_extractor import Phase11FeatureExtractor
from src.decision.product_compliance_decision import EvidenceAwareComplianceDecisionEngine


class TestPhase12StreamlitSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bridge = InferencePipelineBridge.get_instance()
        cls.real_prod_dir = WORKSPACE_ROOT / "real_product_validation/images"
        cls.sample_img_path = cls.real_prod_dir / "real_prod_01.png"

    # 1. App Module Imports
    def test_01_app_modules_import(self):
        import app.streamlit_app
        import app.utils.validation
        import app.utils.visualizer
        import app.utils.inference_bridge
        import app.components.result_card
        import app.components.why_explanation
        import app.components.evidence_panel
        import app.components.yolo_viewer
        import app.components.roi_viewer
        import app.components.ocr_viewer
        import app.components.technical_trace
        import app.components.export
        self.assertTrue(True)

    # 2. Image Validation - Valid Images
    def test_02_image_validation_valid(self):
        synth = Image.new("RGB", (300, 300), color=(200, 200, 200))
        buf = io.BytesIO()
        synth.save(buf, format="PNG")
        val_res = validate_uploaded_image(buf.getvalue(), "sample.png")
        self.assertTrue(val_res["valid"])
        self.assertIsNone(val_res["error"])
        self.assertEqual(val_res["metadata"]["width"], 300)

    # 3. Image Validation - Corrupt and Invalid Formats
    def test_03_image_validation_corrupt_and_invalid(self):
        # Empty
        res_empty = validate_uploaded_image(b"", "empty.png")
        self.assertFalse(res_empty["valid"])
        
        # Invalid extension
        res_ext = validate_uploaded_image(b"fake data", "file.pdf")
        self.assertFalse(res_ext["valid"])
        self.assertIn("Unsupported format", res_ext["error"])
        
        # Corrupt bytes
        res_corrupt = validate_uploaded_image(b"not an image byte stream", "corrupt.png")
        self.assertFalse(res_corrupt["valid"])

    # 4. YOLO Model Loading
    def test_04_yolo_model_loading(self):
        self.assertIsNotNone(self.bridge.yolo_detector)
        self.assertTrue(hasattr(self.bridge.yolo_detector, "detect"))

    # 5. YOLO Detection Execution
    def test_05_yolo_detection_execution(self):
        synth = Image.new("RGB", (640, 640), color=(255, 255, 255))
        res = self.bridge.yolo_detector.detect(synth)
        self.assertIn("detections", res)
        self.assertIn("latency_ms", res)
        self.assertIsInstance(res["detections"], list)

    # 6. YOLO Bounding Box Drawing
    def test_06_yolo_bounding_box_drawing(self):
        synth = Image.new("RGB", (500, 500), color=(100, 100, 100))
        dets = [{"class_name": "brand", "confidence": 0.95, "bbox": [50, 50, 200, 150]}]
        annotated = draw_bounding_boxes(synth, dets)
        self.assertEqual(annotated.size, synth.size)
        self.assertIsInstance(annotated, Image.Image)

    # 7. ROI Cropping
    def test_07_roi_cropping(self):
        synth = Image.new("RGB", (500, 500), color=(150, 150, 150))
        dets = [{"class_name": "mrp details", "confidence": 0.90, "bbox": [20, 20, 100, 80]}]
        crops = self.bridge.yolo_detector.extract_rois(synth, dets)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["class_name"], "mrp details")
        self.assertIsInstance(crops[0]["roi_image"], Image.Image)

    # 8. OCR Extraction in Offline Mode
    def test_08_ocr_offline_mode(self):
        synth = Image.new("RGB", (200, 200), color=(255, 255, 255))
        res = self.bridge.run_inference(synth, offline_mode=True)
        self.assertEqual(res["ocr"]["status"], "OFFLINE_MODE")
        self.assertGreater(len(res["ocr"]["raw_text"]), 0)

    # 9. Evidence Parsing Integration
    def test_09_evidence_parsing(self):
        synth = Image.new("RGB", (200, 200), color=(255, 255, 255))
        res = self.bridge.run_inference(synth, offline_mode=True)
        self.assertIn("declarations", res)
        for d in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            self.assertIn(d, res["declarations"])

    # 10. YOLO Feature Extractor (29 Features)
    def test_10_yolo_feature_extractor_29_features(self):
        yolo_extractor = YOLOFeatureExtractor()
        self.assertEqual(len(yolo_extractor.feature_names), 29)
        feats = yolo_extractor.extract_from_detections([])
        self.assertEqual(len(feats), 29)

    # 11. Multimodal Feature Pipeline (Set D)
    def test_11_multimodal_feature_pipeline(self):
        extractor = self.bridge.feature_extractor
        df = pd.DataFrame([{
            "product_name": "Sample Soap 100g",
            "description": "MRP Rs 50 Net Wt 100g",
            "product_specifications": "",
            "category": "Beauty and Personal Care",
            "retail_price": 50.0,
            "discounted_price": 45.0,
            "ocr_text": "Sample Soap Net Wt 100g MRP Rs 50",
            "visual_regions": [{"class": "brand", "confidence": 0.9, "bbox": [10, 10, 50, 50]}]
        }])
        vec_d = extractor.transform(df, feature_set="SET_D")
        self.assertEqual(vec_d.shape[1], 741)

    # 12. Direct ML Model Inference & Probabilities
    def test_12_direct_ml_model_inference(self):
        synth = Image.new("RGB", (200, 200), color=(255, 255, 255))
        res = self.bridge.run_inference(synth, offline_mode=True)
        self.assertIn("direct_ml", res)
        self.assertIn("probabilities", res["direct_ml"])
        self.assertIn("NON_COMPLIANT", res["direct_ml"]["probabilities"])

    # 13. Three-Class Verdict Validity
    def test_13_three_class_verdict(self):
        synth = Image.new("RGB", (200, 200), color=(255, 255, 255))
        res = self.bridge.run_inference(synth, offline_mode=True)
        self.assertIn(res["overall_prediction"], ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"])

    # 14. D01-D07 Supporting Evidence Integrity
    def test_14_d01_d07_supporting_evidence(self):
        synth = Image.new("RGB", (200, 200), color=(255, 255, 255))
        res = self.bridge.run_inference(synth, offline_mode=True)
        decs = res["declarations"]
        self.assertEqual(len(decs), 7)
        for code in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            self.assertIn("prediction", decs[code])
            self.assertIn("confidence", decs[code])

    # 15. Uncertainty Quantification
    def test_15_uncertainty_quantification(self):
        synth = Image.new("RGB", (200, 200), color=(255, 255, 255))
        res = self.bridge.run_inference(synth, offline_mode=True)
        self.assertIn("uncertainty", res)
        unc = res["uncertainty"]
        if isinstance(unc, dict):
            self.assertIn("status", unc)
            self.assertIn("score", unc)

    # 16. Developer Trace & Latency Breakdown
    def test_16_developer_trace_latency(self):
        synth = Image.new("RGB", (200, 200), color=(255, 255, 255))
        res = self.bridge.run_inference(synth, offline_mode=True)
        lat = res["latency_ms"]
        self.assertIn("total_latency_ms", lat)
        self.assertIn("yolo_localization_ms", lat)
        self.assertIn("decision_engine_ms", lat)
        self.assertGreater(lat["total_latency_ms"], 0.0)

    # 17. JSON Export Schema
    def test_17_json_export_schema(self):
        synth = Image.new("RGB", (200, 200), color=(255, 255, 255))
        res = self.bridge.run_inference(synth, offline_mode=True)
        for key in ["overall_prediction", "confidence", "uncertainty", "primary_reason", "declarations", "yolo", "ocr", "direct_ml", "model_metadata"]:
            self.assertIn(key, res)

    # 18. API Key Security (No Hardcoded Credentials)
    def test_18_api_key_security(self):
        import glob
        app_files = glob.glob(str(WORKSPACE_ROOT / "app/**/*.py"), recursive=True)
        for fpath in app_files:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("AIzaSy", content, f"Possible API key found in {fpath}")

    # 19. Zero-Persistence Privacy Check
    def test_19_zero_persistence_privacy(self):
        # Verify no temp image files left in project root
        png_in_root = list(WORKSPACE_ROOT.glob("tmp_*.png"))
        self.assertEqual(len(png_in_root), 0)

    # 20. End-to-End Pipeline Execution on Real Packaging
    def test_20_end_to_end_real_packaging_execution(self):
        if self.sample_img_path.exists():
            img = Image.open(str(self.sample_img_path)).convert("RGB")
            res = self.bridge.run_inference(img, offline_mode=True)
            self.assertIn(res["overall_prediction"], ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"])
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
