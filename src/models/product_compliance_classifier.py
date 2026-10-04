"""
product_compliance_classifier.py
================================
Trains and benchmarks direct supervised ML models (Logistic Regression, Random Forest, XGBoost)
for product-level compliance prediction, evaluates uncertainty abstention,
and persists Phase 10 model artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml_dataset_v1")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "phase_10")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_partition(split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    prefix = "human_ref" if split == "human_reference" else ("val" if split == "validation" else split)
    x_b = pd.read_csv(os.path.join(DATA_DIR, split, f"X_{prefix}.csv"))
    x_a = pd.read_csv(os.path.join(DATA_DIR, split, f"X_{prefix}_set_a.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, split, f"Y_{prefix}.csv"))["overall_target"]
    return x_a, x_b, y


def train_and_evaluate_direct_models() -> Dict[str, Any]:
    print("=================================================================")
    print("   NITIMARK-ML — DIRECT ML MODEL EXPERIMENTS (PHASE 10)         ")
    print("=================================================================\n")

    X_train_a, X_train_b, y_train = load_partition("train")
    X_val_a, X_val_b, y_val = load_partition("validation")
    X_test_a, X_test_b, y_test = load_partition("test")
    X_human_a, X_human_b, y_human = load_partition("human_reference")

    print(f"Train samples: {len(X_train_a)} | Validation: {len(X_val_a)} | Test: {len(X_test_a)} | Human Ref: {len(X_human_a)}")
    print(f"Train class distribution: {dict(y_train.value_counts())}\n")

    # Encode labels for XGBoost: map NON_COMPLIANT -> 0, UNCERTAIN -> 1 (COMPLIANT -> 2 if exists)
    unique_classes = sorted(list(set(y_train.unique())))
    label2idx = {c: i for i, c in enumerate(unique_classes)}
    idx2label = {i: c for i, c in enumerate(unique_classes)}

    y_train_num = y_train.map(label2idx)
    y_val_num = y_val.map(lambda x: label2idx.get(x, 0))
    y_test_num = y_test.map(lambda x: label2idx.get(x, 0))

    experiments = {}

    # 1. Majority Baseline
    maj_class = y_train.mode()[0]
    y_val_pred_maj = [maj_class] * len(y_val)
    experiments["MajorityBaseline"] = {
        "model_name": "MajorityBaseline",
        "feature_set": "NONE",
        "val_acc": accuracy_score(y_val, y_val_pred_maj),
        "val_macro_f1": f1_score(y_val, y_val_pred_maj, average="macro", zero_division=0),
        "model": None,
        "is_baseline": True,
        "constant_value": maj_class
    }

    # 2. Logistic Regression (Set A)
    lr_a = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_a.fit(X_train_a, y_train)
    y_val_pred_lr = lr_a.predict(X_val_a)
    experiments["LogisticRegression_SetA"] = {
        "model_name": "LogisticRegression",
        "feature_set": "FEATURE_SET_A",
        "val_acc": accuracy_score(y_val, y_val_pred_lr),
        "val_macro_f1": f1_score(y_val, y_val_pred_lr, average="macro", zero_division=0),
        "model": lr_a,
        "is_baseline": False
    }

    # 3. Logistic Regression (Set B)
    lr_b = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_b.fit(X_train_b, y_train)
    y_val_pred_lrb = lr_b.predict(X_val_b)
    experiments["LogisticRegression_SetB"] = {
        "model_name": "LogisticRegression",
        "feature_set": "FEATURE_SET_B",
        "val_acc": accuracy_score(y_val, y_val_pred_lrb),
        "val_macro_f1": f1_score(y_val, y_val_pred_lrb, average="macro", zero_division=0),
        "model": lr_b,
        "is_baseline": False
    }

    # 4. Random Forest (Set A)
    rf_a = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight="balanced")
    rf_a.fit(X_train_a, y_train)
    y_val_pred_rfa = rf_a.predict(X_val_a)
    experiments["RandomForest_SetA"] = {
        "model_name": "RandomForest",
        "feature_set": "FEATURE_SET_A",
        "val_acc": accuracy_score(y_val, y_val_pred_rfa),
        "val_macro_f1": f1_score(y_val, y_val_pred_rfa, average="macro", zero_division=0),
        "model": rf_a,
        "is_baseline": False
    }

    # 5. Random Forest (Set B)
    rf_b = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight="balanced")
    rf_b.fit(X_train_b, y_train)
    y_val_pred_rfb = rf_b.predict(X_val_b)
    experiments["RandomForest_SetB"] = {
        "model_name": "RandomForest",
        "feature_set": "FEATURE_SET_B",
        "val_acc": accuracy_score(y_val, y_val_pred_rfb),
        "val_macro_f1": f1_score(y_val, y_val_pred_rfb, average="macro", zero_division=0),
        "model": rf_b,
        "is_baseline": False
    }

    # 6. XGBoost (Set A)
    xgb_a = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss")
    xgb_a.fit(X_train_a, y_train_num)
    y_val_pred_xgba_num = xgb_a.predict(X_val_a)
    y_val_pred_xgba = [idx2label[i] for i in y_val_pred_xgba_num]
    experiments["XGBoost_SetA"] = {
        "model_name": "XGBoost",
        "feature_set": "FEATURE_SET_A",
        "val_acc": accuracy_score(y_val, y_val_pred_xgba),
        "val_macro_f1": f1_score(y_val, y_val_pred_xgba, average="macro", zero_division=0),
        "model": xgb_a,
        "is_baseline": False,
        "idx2label": idx2label
    }

    # Select Best Direct ML Model based on Validation Macro F1
    best_exp_key = max(
        (k for k in experiments if not experiments[k]["is_baseline"]),
        key=lambda k: experiments[k]["val_macro_f1"]
    )
    best_exp = experiments[best_exp_key]

    print(f"Validation Results:")
    for k, v in experiments.items():
        print(f" - {k:<25} (Feats: {v['feature_set']:<14}) | Val Acc: {v['val_acc']*100:.2f}% | Val Macro F1: {v['val_macro_f1']:.4f}")

    print(f"\nSelected Best Direct ML Model: {best_exp_key} (Val Macro F1: {best_exp['val_macro_f1']:.4f})")

    # Persist Best Direct ML Model
    model_artifact_path = os.path.join(MODELS_DIR, "product_compliance_model.joblib")
    joblib.dump(best_exp["model"], model_artifact_path)

    # Save registry
    registry = {
        "selected_direct_model": best_exp_key,
        "model_family": best_exp["model_name"],
        "feature_set": best_exp["feature_set"],
        "validation_macro_f1": best_exp["val_macro_f1"],
        "validation_accuracy": best_exp["val_acc"],
        "class_mapping": label2idx,
        "experiments": {
            k: {
                "model_name": v["model_name"],
                "feature_set": v["feature_set"],
                "val_acc": v["val_acc"],
                "val_macro_f1": v["val_macro_f1"]
            }
            for k, v in experiments.items()
        }
    }

    with open(os.path.join(MODELS_DIR, "model_registry.json"), "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    # Save Decision Policy Config
    from src.decision.product_compliance_decision import DEFAULT_POLICY_CONFIG
    with open(os.path.join(MODELS_DIR, "decision_policy.json"), "w", encoding="utf-8") as f:
        json.dump(DEFAULT_POLICY_CONFIG, f, indent=2)

    print(f"\nPersisted Phase 10 artifacts to {MODELS_DIR}/")
    return registry


if __name__ == "__main__":
    train_and_evaluate_direct_models()
