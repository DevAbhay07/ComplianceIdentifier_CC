"""
test_phase_10.py
================
Comprehensive unit test suite for Phase 10:
- Leakage audit verification
- Cryptographic hash verification of human reference data
- Decision engine policy logic (COMPLIANT, NON_COMPLIANT, UNCERTAIN)
- Direct ML model artifact existence and execution
- Inference CLI and API schema compatibility
- 15 publication figures existence and size
- 26 Quality Gates execution
"""

import os
import sys
import json
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.product_level_leakage_audit import run_full_leakage_audit
from src.decision.product_compliance_decision import EvidenceAwareComplianceDecisionEngine
from src.inference.predict import run_standalone_prediction
from src.evaluation.phase_10_quality_gate import run_all_quality_gates, compute_sha256, EXPECTED_HUMAN_GT_SHA256, HUMAN_GT_PATH


class TestPhase10ProductLevelCompliance(unittest.TestCase):

    def test_human_ground_truth_hash_immutability(self):
        """Verify human ground truth dataset is 100% immutable."""
        self.assertTrue(os.path.exists(HUMAN_GT_PATH), "Human ground truth file missing.")
        actual_hash = compute_sha256(HUMAN_GT_PATH)
        self.assertEqual(
            actual_hash,
            EXPECTED_HUMAN_GT_SHA256,
            f"Human ground truth hash modified! Got {actual_hash}"
        )

    def test_leakage_audit_passes(self):
        """Verify zero data or target leakage."""
        res = run_full_leakage_audit()
        self.assertTrue(res["passed"], f"Leakage audit failed: {res['errors']}")

    def test_decision_engine_compliant(self):
        """Verify decision engine returns COMPLIANT when all mandatory declarations are PRESENT."""
        eng = EvidenceAwareComplianceDecisionEngine()
        test_decls = {
            "D01": {"prediction": "PRESENT", "confidence": 0.90},
            "D02": {"prediction": "PRESENT", "confidence": 0.95},
            "D03": {"prediction": "PRESENT", "confidence": 0.90},
            "D04": {"prediction": "PRESENT", "confidence": 0.85},
            "D05": {"prediction": "PRESENT", "confidence": 0.99},
            "D06": {"prediction": "PRESENT", "confidence": 0.90},
            "D07": {"prediction": "PRESENT", "confidence": 0.90}
        }
        res = eng.evaluate_product(test_decls)
        self.assertEqual(res["overall_prediction"], "COMPLIANT")
        self.assertFalse(res["uncertainty"])
        self.assertGreater(res["confidence"], 0.70)

    def test_decision_engine_non_compliant(self):
        """Verify decision engine returns NON_COMPLIANT when a mandatory declaration is ABSENT."""
        eng = EvidenceAwareComplianceDecisionEngine()
        test_decls = {
            "D01": {"prediction": "PRESENT", "confidence": 0.90},
            "D02": {"prediction": "PRESENT", "confidence": 0.95},
            "D03": {"prediction": "PRESENT", "confidence": 0.90},
            "D04": {"prediction": "PRESENT", "confidence": 0.85},
            "D05": {"prediction": "PRESENT", "confidence": 0.99},
            "D06": {"prediction": "ABSENT", "confidence": 0.88},
            "D07": {"prediction": "PRESENT", "confidence": 0.90}
        }
        res = eng.evaluate_product(test_decls)
        self.assertEqual(res["overall_prediction"], "NON_COMPLIANT")
        self.assertFalse(res["uncertainty"])
        self.assertIn("D06", res["primary_reason"])

    def test_decision_engine_uncertain(self):
        """Verify decision engine returns UNCERTAIN when a mandatory declaration is CANNOT_DETERMINE."""
        eng = EvidenceAwareComplianceDecisionEngine()
        test_decls = {
            "D01": {"prediction": "CANNOT_DETERMINE", "confidence": 0.50},
            "D02": {"prediction": "PRESENT", "confidence": 0.95},
            "D03": {"prediction": "PRESENT", "confidence": 0.90},
            "D04": {"prediction": "PRESENT", "confidence": 0.85},
            "D05": {"prediction": "PRESENT", "confidence": 0.99},
            "D06": {"prediction": "PRESENT", "confidence": 0.90},
            "D07": {"prediction": "PRESENT", "confidence": 0.90}
        }
        res = eng.evaluate_product(test_decls)
        self.assertEqual(res["overall_prediction"], "UNCERTAIN")
        self.assertTrue(res["uncertainty"])

    def test_direct_ml_artifacts_exist(self):
        """Verify Phase 10 direct ML model artifacts and registry exist."""
        p10_dir = os.path.join(PROJECT_ROOT, "models", "phase_10")
        mod_file = os.path.join(p10_dir, "product_compliance_model.joblib")
        reg_file = os.path.join(p10_dir, "model_registry.json")
        self.assertTrue(os.path.exists(mod_file), "product_compliance_model.joblib missing")
        self.assertTrue(os.path.exists(reg_file), "model_registry.json missing")

    def test_all_15_figures_exist_and_non_empty(self):
        """Verify all 15 publication figures are generated and non-empty."""
        fig_dir = os.path.join(PROJECT_ROOT, "reports", "phase_10", "figures")
        self.assertTrue(os.path.exists(fig_dir), "figures directory missing")
        expected_figs = [
            "01_product_level_model_comparison.png",
            "02_product_level_confusion_matrix.png",
            "03_product_level_class_distribution.png",
            "04_non_compliant_precision_recall.png",
            "05_non_compliant_recall_comparison.png",
            "06_validation_vs_test.png",
            "07_test_vs_human_reference.png",
            "08_test_vs_real_product.png",
            "09_confidence_distribution.png",
            "10_uncertainty_coverage.png",
            "11_calibration_curve.png",
            "12_false_negative_breakdown.png",
            "13_false_positive_breakdown.png",
            "14_failure_taxonomy.png",
            "15_old_vs_new_architecture.png"
        ]
        for f in expected_figs:
            p = os.path.join(fig_dir, f)
            self.assertTrue(os.path.exists(p), f"Missing figure: {f}")
            self.assertGreater(os.path.getsize(p), 5000, f"Figure too small: {f}")

    def test_predict_cli_output_schema(self):
        """Verify standalone prediction includes overall product compliance fields."""
        res = run_standalone_prediction(title="Dettol Handwash 200ml MRP Rs 99 Mfd by Reckitt", offline_mode=True)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("overall_prediction", res)
        self.assertIn(res["overall_prediction"], ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"])
        self.assertIn("confidence", res)
        self.assertIn("primary_reason", res)
        self.assertIn("decision_basis", res)
        self.assertIn("declarations", res)

    def test_all_26_quality_gates_pass(self):
        """Verify all 26 Quality Gates pass."""
        gate_res = run_all_quality_gates()
        self.assertTrue(gate_res["all_passed"], f"Quality gates failed: {gate_res}")
        self.assertEqual(gate_res["passed_gates"], 26)


if __name__ == "__main__":
    unittest.main()
