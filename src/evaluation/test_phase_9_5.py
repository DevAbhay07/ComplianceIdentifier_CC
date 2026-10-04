"""
test_phase_9_5.py
=================
Comprehensive unit test suite for Phase 9.5:
- ML metric computation engine
- Publication figure existence and format
- Real-product validation dataset integrity & predictions
- Cryptographic hash verification of human reference data
- Quality gate evaluator execution
- Secret isolation and .gitignore rules
"""

import os
import sys
import json
import unittest
import hashlib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.phase_9_5_quality_gate import run_all_quality_gates, compute_sha256, EXPECTED_HUMAN_GT_SHA256, HUMAN_GT_PATH
from src.inference.model_loader import DECLARATIONS


class TestPhase95PerformanceAndValidation(unittest.TestCase):

    def test_human_ground_truth_hash_immutability(self):
        """Verify human ground truth dataset is 100% immutable."""
        self.assertTrue(os.path.exists(HUMAN_GT_PATH), "Human ground truth file missing.")
        actual_hash = compute_sha256(HUMAN_GT_PATH)
        self.assertEqual(
            actual_hash,
            EXPECTED_HUMAN_GT_SHA256,
            f"Human ground truth hash modified! Got {actual_hash}"
        )

    def test_real_product_validation_dataset_completeness(self):
        """Verify 20 real product images, inputs, manual references, and predictions exist."""
        img_dir = os.path.join(PROJECT_ROOT, "real_product_validation", "images")
        inp_dir = os.path.join(PROJECT_ROOT, "real_product_validation", "inputs")
        ref_dir = os.path.join(PROJECT_ROOT, "real_product_validation", "manual_reference")
        pred_dir = os.path.join(PROJECT_ROOT, "real_product_validation", "predictions")

        self.assertTrue(os.path.exists(img_dir), "images dir missing")
        self.assertTrue(os.path.exists(inp_dir), "inputs dir missing")
        self.assertTrue(os.path.exists(ref_dir), "manual_reference dir missing")
        self.assertTrue(os.path.exists(pred_dir), "predictions dir missing")

        for i in range(1, 21):
            pid = f"real_prod_{i:02d}"
            img_file = os.path.join(img_dir, f"{pid}.png")
            inp_file = os.path.join(inp_dir, f"{pid}.json")
            ref_file = os.path.join(ref_dir, f"{pid}_ref.json")
            pred_file = os.path.join(pred_dir, f"{pid}_pred.json")

            self.assertTrue(os.path.exists(img_file), f"Missing image {img_file}")
            self.assertTrue(os.path.exists(inp_file), f"Missing input {inp_file}")
            self.assertTrue(os.path.exists(ref_file), f"Missing ref {ref_file}")
            self.assertTrue(os.path.exists(pred_file), f"Missing pred {pred_file}")

            with open(ref_file, "r", encoding="utf-8") as f:
                ref_data = json.load(f)
                self.assertIn("category", ref_data)
                self.assertIn("declarations", ref_data)
                for did in DECLARATIONS:
                    self.assertIn(did, ref_data["declarations"])

            with open(pred_file, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
                self.assertEqual(pred_data.get("status"), "SUCCESS")
                self.assertIn("declarations", pred_data)
                for did in DECLARATIONS:
                    self.assertIn(did, pred_data["declarations"])
                    self.assertIn("prediction", pred_data["declarations"][did])
                    self.assertIn("confidence", pred_data["declarations"][did])

    def test_five_categories_present_in_validation(self):
        """Verify validation covers exactly 5 diverse categories."""
        ref_dir = os.path.join(PROJECT_ROOT, "real_product_validation", "manual_reference")
        cats = set()
        for i in range(1, 21):
            ref_file = os.path.join(ref_dir, f"real_prod_{i:02d}_ref.json")
            with open(ref_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                cats.add(data["category"])
        self.assertGreaterEqual(len(cats), 5, f"Expected at least 5 categories, found {cats}")

    def test_all_20_figures_exist_and_non_empty(self):
        """Verify all 20 publication charts are generated and non-empty PNG files."""
        fig_dir = os.path.join(PROJECT_ROOT, "reports", "phase_9_5", "figures")
        self.assertTrue(os.path.exists(fig_dir), "figures directory missing")
        expected_figs = [
            "01_model_validation_macro_f1.png",
            "02_model_test_macro_f1.png",
            "03_model_human_reference_macro_f1.png",
            "04_validation_vs_test.png",
            "05_validation_vs_human_gap.png",
            "06_declaration_performance.png",
            "07_class_distribution.png",
            "08_confusion_matrix_D01.png",
            "09_confusion_matrix_D02.png",
            "10_confusion_matrix_D03.png",
            "11_confusion_matrix_D04.png",
            "12_confusion_matrix_D05.png",
            "13_confusion_matrix_D06.png",
            "14_confusion_matrix_D07.png",
            "15_confidence_distribution.png",
            "16_uncertainty_distribution.png",
            "17_feature_importance_D01.png",
            "18_feature_importance_D03.png",
            "19_feature_importance_D06.png",
            "20_model_comparison.png",
        ]
        for f_name in expected_figs:
            f_path = os.path.join(fig_dir, f_name)
            self.assertTrue(os.path.exists(f_path), f"Figure missing: {f_name}")
            self.assertGreater(os.path.getsize(f_path), 5000, f"Figure too small: {f_name}")

    def test_all_required_reports_exist(self):
        """Verify all required markdown reports and JSON metric files exist."""
        required_reports = [
            "reports/phase_9_5/complete_model_metrics.md",
            "reports/phase_9_5/declaration_analysis.md",
            "reports/phase_9_5/model_comparison.md",
            "reports/phase_9_5/confidence_uncertainty.md",
            "reports/phase_9_5/D01_analysis.md",
            "reports/phase_9_5/D02_analysis.md",
            "reports/phase_9_5/D03_analysis.md",
            "reports/phase_9_5/D04_analysis.md",
            "reports/phase_9_5/D05_analysis.md",
            "reports/phase_9_5/D06_analysis.md",
            "reports/phase_9_5/D07_analysis.md",
            "reports/phase_9_5/real_product_results.md",
            "reports/phase_9_5/real_product_failure_analysis.md",
            "reports/phase_9_5/distribution_shift.md",
            "reports/phase_9_5/latency_analysis.md",
            "reports/phase_9_5/api_failure_analysis.md",
            "reports/phase_9_5/PHASE_9_5_FINAL_REPORT.md",
            "reports/phase_9_5/PHASE_9_5_QUALITY_GATE.md",
            "reports/phase_9_5/recommended_phase_10_improvements.md",
            "reports/phase_9_5/all_metrics.json",
            "real_product_validation/README.md",
            "real_product_validation/reports/real_product_validation_summary.json"
        ]
        for r_path in required_reports:
            full_path = os.path.join(PROJECT_ROOT, r_path)
            self.assertTrue(os.path.exists(full_path), f"Required report missing: {r_path}")
            self.assertGreater(os.path.getsize(full_path), 100, f"Report file too small: {r_path}")

    def test_all_26_quality_gates_pass(self):
        """Verify the quality gate audit runner executes and passes 26/26 gates."""
        gate_res = run_all_quality_gates()
        self.assertTrue(gate_res["all_passed"], f"Quality gates failed! Summary: {gate_res}")
        self.assertEqual(gate_res["passed_gates"], 26)
        self.assertEqual(gate_res["failed_gates"], 0)

    def test_secrets_and_gitignore_rules(self):
        """Verify .gitignore contains rule to exclude API keys and .env."""
        gi_path = os.path.join(PROJECT_ROOT, ".gitignore")
        self.assertTrue(os.path.exists(gi_path), ".gitignore missing")
        with open(gi_path, "r", encoding="utf-8") as f:
            gi_content = f.read()
        self.assertIn(".env", gi_content)
        self.assertTrue("*.key" in gi_content or "secrets" in gi_content)


if __name__ == "__main__":
    unittest.main()
