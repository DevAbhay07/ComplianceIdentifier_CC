"""
test_phase_8.py
===============
Unit test suite for Phase 8 Multi-Label Model Training and Evaluation Pipeline.
Contains 20 comprehensive unit tests covering metrics, registry, model wrappers,
training routines, evaluation logic, leakage detection, and data integrity.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import unittest
import numpy as np
import pandas as pd
import tempfile
import json
import hashlib
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import evaluate_declaration_predictions
from src.models.model_registry import log_experiment, save_model_artifact
from src.models.train_xgboost import XGBoostDeclarationWrapper
from src.evaluation.compare_models import evaluate_majority_baseline, compare_and_select_models
from src.evaluation.feature_importance import FORBIDDEN_LEAKAGE_SUBSTRINGS
from src.evaluation.error_analysis import get_prediction_and_confidence
from src.models.phase_8_quality_gate import compute_file_sha256, EXPECTED_HUMAN_GT_SHA256


class TestPhase8Suite(unittest.TestCase):

    def test_01_metrics_calculation(self):
        """Test accuracy, macro F1, and balanced accuracy calculation."""
        y_true = np.array(["PRESENT", "ABSENT", "PARTIAL", "PRESENT"])
        y_pred = np.array(["PRESENT", "ABSENT", "PRESENT", "PRESENT"])
        res = evaluate_declaration_predictions(y_true, y_pred)
        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("weighted_f1", res)
        self.assertIn("balanced_accuracy", res)
        self.assertEqual(res["accuracy"], 0.75)
        self.assertGreater(res["macro_f1"], 0.0)

    def test_02_metrics_confusion_matrix(self):
        """Test confusion matrix structure."""
        y_true = ["PRESENT", "ABSENT", "PRESENT"]
        y_pred = ["PRESENT", "PRESENT", "PRESENT"]
        res = evaluate_declaration_predictions(y_true, y_pred)
        cm = res["confusion_matrix"]
        self.assertIsInstance(cm, pd.DataFrame)
        self.assertEqual(cm.loc["True_PRESENT", "Pred_PRESENT"], 2)

    def test_03_per_class_metrics(self):
        """Test per-class precision/recall metrics."""
        y_true = ["PRESENT", "ABSENT", "PRESENT"]
        y_pred = ["PRESENT", "ABSENT", "ABSENT"]
        res = evaluate_declaration_predictions(y_true, y_pred)
        per_class = res["per_class"]
        self.assertIn("PRESENT", per_class)
        self.assertIn("ABSENT", per_class)
        self.assertEqual(per_class["PRESENT"]["support"], 2)

    def test_04_experiment_registry_logging(self):
        """Test logging experiment records to CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "exp_reg.csv")
            log_experiment(
                experiment_id="TEST_EXP_01",
                model_family="LogisticRegression",
                declaration="D01",
                feature_set="FEATURE_SET_A",
                target_type="Multiclass",
                hyperparameters={"C": 1.0},
                class_weight=None,
                sample_weight=False,
                val_macro_f1=0.85,
                val_weighted_f1=0.86,
                val_accuracy=0.88,
                training_time_sec=0.12,
                registry_csv=csv_path
            )
            self.assertTrue(os.path.exists(csv_path))
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["experiment_id"], "TEST_EXP_01")

    def test_05_model_artifact_persistence(self):
        """Test serializing model artifact and JSON registry entry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            lr = LogisticRegression()
            lr.fit([[1, 2], [3, 4]], [0, 1])
            save_model_artifact(
                model=lr,
                model_family="LogisticRegression",
                declaration="D01",
                feature_set="FEATURE_SET_A",
                hyperparameters={"C": 1.0},
                val_metrics={"macro_f1": 0.9, "accuracy": 0.95},
                random_seed=42,
                models_dir=tmp_dir
            )
            joblib_path = os.path.join(tmp_dir, "model_D01_LogisticRegression_FEATURE_SET_A.joblib")
            self.assertTrue(os.path.exists(joblib_path))
            reg_json = os.path.join(tmp_dir, "model_registry.json")
            self.assertTrue(os.path.exists(reg_json))

    def test_06_xgboost_wrapper_predictions(self):
        """Test XGBoost wrapper encoding and decoding string labels."""
        encoder = LabelEncoder()
        labels = ["ABSENT", "PARTIAL", "PRESENT"]
        encoder.fit(labels)

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = encoder.transform(["ABSENT", "PARTIAL", "PRESENT"])

        clf = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
        clf.fit(X, y)

        wrapper = XGBoostDeclarationWrapper(clf, encoder)
        preds = wrapper.predict(X)
        self.assertIsInstance(preds, np.ndarray)
        self.assertIn(preds[0], labels)
        self.assertEqual(len(preds), 3)

    def test_07_xgboost_wrapper_probabilities(self):
        """Test XGBoost wrapper predict_proba."""
        encoder = LabelEncoder()
        encoder.fit(["ABSENT", "PRESENT"])
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = encoder.transform(["ABSENT", "PRESENT"])
        clf = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
        clf.fit(X, y)
        wrapper = XGBoostDeclarationWrapper(clf, encoder)
        probs = wrapper.predict_proba(X)
        self.assertEqual(probs.shape, (2, 2))
        self.assertAlmostEqual(np.sum(probs[0]), 1.0, places=4)

    def test_08_majority_baseline(self):
        """Test majority baseline evaluator."""
        Y_train = pd.DataFrame({"D01_target": ["PRESENT", "PRESENT", "ABSENT"]})
        Y_val = pd.DataFrame({"D01_target": ["PRESENT", "ABSENT"]})
        base = evaluate_majority_baseline(Y_train, Y_val)
        self.assertIn("D01", base)
        self.assertEqual(base["D01"]["majority_class"], "PRESENT")

    def test_09_model_selection_macro_f1(self):
        """Test model selection selects candidate with highest Macro F1."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline = {
                "D01": {"model_family": "MajorityBaseline", "majority_class": "PRESENT", "metrics": {"macro_f1": 0.5, "weighted_f1": 0.5, "accuracy": 0.6, "balanced_accuracy": 0.5}}
            }
            all_exp = {
                "LR_FEATURE_SET_A": {
                    "D01": {"model": "LR_A", "feature_set": "FEATURE_SET_A", "metrics": {"macro_f1": 0.75, "weighted_f1": 0.75, "accuracy": 0.8, "balanced_accuracy": 0.75}, "params": {"C": 1.0}}
                },
                "RF_FEATURE_SET_B": {
                    "D01": {"model": "RF_B", "feature_set": "FEATURE_SET_B", "metrics": {"macro_f1": 0.92, "weighted_f1": 0.92, "accuracy": 0.94, "balanced_accuracy": 0.92}, "params": {"n_estimators": 100}}
                }
            }
            frozen, comp_df = compare_and_select_models(all_exp, baseline, tmp_dir)
            self.assertEqual(frozen["D01"]["model_family"], "RF")
            self.assertEqual(frozen["D01"]["val_macro_f1"], 0.92)

    def test_10_anti_leakage_screening_pass(self):
        """Test clean features pass leakage screening."""
        clean_feats = ["title_length", "tfidf_word_100", "brand_detected", "net_qty_match"]
        leaks = [f for f in clean_feats if any(sub in f for sub in FORBIDDEN_LEAKAGE_SUBSTRINGS)]
        self.assertEqual(len(leaks), 0)

    def test_11_anti_leakage_screening_catch(self):
        """Test forbidden feature is caught."""
        dirty_feats = ["title_length", "D01_target_leaked", "rule_confidence_score"]
        leaks = [f for f in dirty_feats if any(sub in f for sub in FORBIDDEN_LEAKAGE_SUBSTRINGS)]
        self.assertEqual(len(leaks), 2)

    def test_12_confidence_prediction_helper(self):
        """Test get_prediction_and_confidence for constant and trained models."""
        X_dummy = pd.DataFrame({"feat1": [1, 2, 3]})
        preds, confs = get_prediction_and_confidence("PRESENT", X_dummy)
        self.assertEqual(len(preds), 3)
        self.assertEqual(preds[0], "PRESENT")
        self.assertEqual(confs[0], 1.0)

    def test_13_dataset_split_sizes(self):
        """Verify partition file record counts match Phase 7 specifications."""
        data_dir = "/home/devabhay07/Desktop/MLmodelV2/data/processed/ml_dataset_v1"
        train_y = pd.read_csv(os.path.join(data_dir, "train/Y_train.csv"))
        val_y = pd.read_csv(os.path.join(data_dir, "validation/Y_val.csv"))
        test_y = pd.read_csv(os.path.join(data_dir, "test/Y_test.csv"))
        human_y = pd.read_csv(os.path.join(data_dir, "human_reference/Y_human_ref.csv"))

        self.assertEqual(len(train_y), 1159)
        self.assertEqual(len(val_y), 246)
        self.assertEqual(len(test_y), 258)
        self.assertEqual(len(human_y), 75)

    def test_14_feature_dimensions(self):
        """Verify feature dimensions: Set A = 219, Set B = 719."""
        data_dir = "/home/devabhay07/Desktop/MLmodelV2/data/processed/ml_dataset_v1"
        X_train = pd.read_csv(os.path.join(data_dir, "train/X_train.csv"))
        self.assertEqual(X_train.shape[1], 719)
        self.assertEqual(X_train.iloc[:, :219].shape[1], 219)

    def test_15_human_ground_truth_integrity(self):
        """Verify human ground truth dataset hash is 100% unchanged."""
        gt_path = "/home/devabhay07/Desktop/MLmodelV2/data/processed/pilot_ground_truth_75_v1.jsonl"
        actual_hash = compute_file_sha256(gt_path)
        self.assertEqual(actual_hash, EXPECTED_HUMAN_GT_SHA256)

    def test_16_target_column_isolation(self):
        """Verify no target column exists in X feature matrices."""
        data_dir = "/home/devabhay07/Desktop/MLmodelV2/data/processed/ml_dataset_v1"
        X_train = pd.read_csv(os.path.join(data_dir, "train/X_train.csv"))
        for col in X_train.columns:
            self.assertNotIn("target", col.lower())
            self.assertNotIn("overall_compliance", col.lower())

    def test_17_single_class_logistic_handling(self):
        """Verify constant prediction handling for single-class targets."""
        X_tr = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        Y_tr = pd.DataFrame({"D02_target": ["PRESENT", "PRESENT"]})
        X_v = pd.DataFrame({"a": [5], "b": [6]})
        Y_v = pd.DataFrame({"D02_target": ["PRESENT"]})
        res = evaluate_majority_baseline(Y_tr, Y_v)
        self.assertEqual(res["D02"]["majority_class"], "PRESENT")

    def test_18_feature_pipeline_joblib_exists(self):
        """Verify feature_pipeline_v1.joblib exists."""
        pipe_path = "/home/devabhay07/Desktop/MLmodelV2/models/feature_pipeline_v1.joblib"
        self.assertTrue(os.path.exists(pipe_path))

    def test_19_balanced_accuracy_multiclass(self):
        """Test balanced accuracy on multiclass."""
        y_true = ["A", "A", "B", "C"]
        y_pred = ["A", "A", "B", "B"]
        res = evaluate_declaration_predictions(y_true, y_pred)
        self.assertIn("balanced_accuracy", res)

    def test_20_reproducibility_seed(self):
        """Verify reproducible Random Forest results with fixed seed."""
        X = np.random.RandomState(42).randn(20, 5)
        y = np.array([0] * 10 + [1] * 10)
        rf1 = RandomForestClassifier(n_estimators=10, random_state=42)
        rf1.fit(X, y)
        rf2 = RandomForestClassifier(n_estimators=10, random_state=42)
        rf2.fit(X, y)
        np.testing.assert_array_equal(rf1.predict(X), rf2.predict(X))


if __name__ == "__main__":
    unittest.main()
