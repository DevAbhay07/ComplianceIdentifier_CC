"""
test_model_compatibility.py
============================
Verifies that all frozen Phase 8 models and baselines for D01 through D07
load without error and successfully execute inference on synthetic schema-valid vectors.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import unittest
import numpy as np
import pandas as pd
import joblib

from src.inference.model_loader import load_frozen_models
from src.inference.declaration_predictor import predict_all_declarations

DEFAULT_PIPELINE_PATH = "models/feature_pipeline_v1.joblib"


class TestModelCompatibility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = load_frozen_models()
        cls.pipeline = joblib.load(DEFAULT_PIPELINE_PATH)

    def test_all_declarations_present(self):
        """Verify all 7 declarations D01 through D07 are loaded."""
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            self.assertIn(did, self.models)
            self.assertIn("model_type", self.models[did])

    def test_synthetic_vector_inference(self):
        """Verify inference runs on synthetic schema-valid feature matrices."""
        names_a = self.pipeline.feature_names_a_
        names_b = self.pipeline.feature_names_b_

        # Create dummy zeros vector
        X_a = pd.DataFrame(np.zeros((1, len(names_a))), columns=names_a)
        X_b = pd.DataFrame(np.zeros((1, len(names_b))), columns=names_b)

        preds = predict_all_declarations(self.models, X_a, X_b)
        self.assertEqual(len(preds), 7)

        for did, p in preds.items():
            self.assertIn("prediction", p)
            self.assertIn("confidence", p)
            self.assertIn("probabilities", p)
            self.assertIn("model_type", p)
            self.assertGreaterEqual(p["confidence"], 0.0)
            self.assertLessEqual(p["confidence"], 1.0)
            self.assertAlmostEqual(sum(p["probabilities"].values()), 1.0, places=3)

    def test_baseline_declarations(self):
        """Verify baselines report MAJORITY_BASELINE and constant values."""
        self.assertTrue(self.models["D02"]["is_baseline"])
        self.assertEqual(self.models["D02"]["model_type"], "MAJORITY_BASELINE")
        self.assertEqual(self.models["D02"]["model_object"], "PRESENT")

        self.assertTrue(self.models["D04"]["is_baseline"])
        self.assertEqual(self.models["D04"]["model_object"], "CANNOT_DETERMINE")

        self.assertTrue(self.models["D05"]["is_baseline"])
        self.assertEqual(self.models["D05"]["model_object"], "PRESENT")

        self.assertTrue(self.models["D07"]["is_baseline"])
        self.assertEqual(self.models["D07"]["model_object"], "CANNOT_DETERMINE")

    def test_learned_ml_declarations(self):
        """Verify learned models are RandomForestClassifier."""
        for did in ["D01", "D03", "D06"]:
            m = self.models[did]
            self.assertFalse(m["is_baseline"])
            self.assertEqual(m["model_family"], "RandomForest")
            self.assertTrue(hasattr(m["model_object"], "predict"))


if __name__ == "__main__":
    unittest.main()
