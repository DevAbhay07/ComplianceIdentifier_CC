"""
train_xgboost.py
=================
Trains and tunes XGBoost classifiers for declarations D01 through D07
across Feature Set A (tabular) and Feature Set B (multi-modal).

Handles LabelEncoding per declaration and inverse-transforms predictions
to canonical semantic string labels.

Evaluates hyperparameters:
- n_estimators in [50, 100]
- max_depth in [3, 6]
- learning_rate in [0.05, 0.1]
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from src.evaluation.metrics import evaluate_declaration_predictions
from src.models.model_registry import log_experiment, save_model_artifact


DECLARATIONS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]


class XGBoostDeclarationWrapper:
    """Wrapper encapsulating XGBClassifier with its fitted LabelEncoder."""
    def __init__(self, model: xgb.XGBClassifier, encoder: LabelEncoder):
        self.model = model
        self.encoder = encoder

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        int_preds = self.model.predict(X)
        return self.encoder.inverse_transform(int_preds)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def train_xgboost_grid(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    feature_set_name: str = "FEATURE_SET_B",
    random_seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Run grid search for XGBoost on each declaration.
    Returns best model and metrics per declaration selected strictly on Validation Macro F1.
    """
    print(f"\n--- Training XGBoost Models ({feature_set_name}) ---")

    n_estimators_list = [50, 100]
    max_depth_list = [3, 6]
    learning_rate_list = [0.05, 0.1]

    best_models_per_decl = {}

    for did in DECLARATIONS:
        target_col = f"{did}_target"
        y_train_raw = Y_train[target_col].values
        y_val_raw = Y_val[target_col].values

        unique_train_classes = sorted(list(set(y_train_raw)))

        # Single-class case
        if len(unique_train_classes) <= 1:
            majority_class = unique_train_classes[0]
            val_preds = [majority_class] * len(y_val_raw)
            val_metrics = evaluate_declaration_predictions(y_val_raw, val_preds)

            best_models_per_decl[did] = {
                "model": majority_class,
                "params": {"constant": majority_class},
                "metrics": val_metrics,
                "feature_set": feature_set_name
            }
            log_experiment(
                experiment_id=f"XGB_{did}_{feature_set_name}_constant",
                model_family="XGBoost",
                declaration=did,
                feature_set=feature_set_name,
                target_type="SingleClass",
                hyperparameters={"constant": majority_class},
                class_weight=None,
                sample_weight=False,
                val_macro_f1=val_metrics["macro_f1"],
                val_weighted_f1=val_metrics["weighted_f1"],
                val_accuracy=val_metrics["accuracy"],
                training_time_sec=0.001
            )
            continue

        # Fit LabelEncoder on training targets
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_raw)

        # Map validation classes (handle potential unseen classes safely)
        val_mask = np.isin(y_val_raw, le.classes_)
        y_val_encoded = np.zeros(len(y_val_raw), dtype=int)
        y_val_encoded[val_mask] = le.transform(y_val_raw[val_mask])

        num_classes = len(le.classes_)
        objective = "binary:logistic" if num_classes == 2 else "multi:softprob"

        best_f1 = -1.0
        best_candidate = None

        for n_est in n_estimators_list:
            for md in max_depth_list:
                for lr in learning_rate_list:
                    t0 = time.time()
                    model = xgb.XGBClassifier(
                        n_estimators=n_est,
                        max_depth=md,
                        learning_rate=lr,
                        objective=objective,
                        random_state=random_seed,
                        eval_metric="logloss" if num_classes == 2 else "mlogloss",
                        n_jobs=-1,
                        verbosity=0
                    )
                    model.fit(X_train, y_train_encoded)
                    train_time = time.time() - t0

                    wrapper = XGBoostDeclarationWrapper(model, le)
                    val_preds = wrapper.predict(X_val)
                    val_metrics = evaluate_declaration_predictions(y_val_raw, val_preds)

                    exp_id = f"XGB_{did}_{feature_set_name}_nest{n_est}_md{md}_lr{lr}"
                    log_experiment(
                        experiment_id=exp_id,
                        model_family="XGBoost",
                        declaration=did,
                        feature_set=feature_set_name,
                        target_type="Multiclass",
                        hyperparameters={"n_estimators": n_est, "max_depth": md, "learning_rate": lr},
                        class_weight=None,
                        sample_weight=False,
                        val_macro_f1=val_metrics["macro_f1"],
                        val_weighted_f1=val_metrics["weighted_f1"],
                        val_accuracy=val_metrics["accuracy"],
                        training_time_sec=train_time
                    )

                    # Selection based on Validation Macro F1
                    if val_metrics["macro_f1"] > best_f1:
                        best_f1 = val_metrics["macro_f1"]
                        best_candidate = {
                            "model": wrapper,
                            "raw_model": model,
                            "encoder": le,
                            "params": {"n_estimators": n_est, "max_depth": md, "learning_rate": lr},
                            "metrics": val_metrics,
                            "feature_set": feature_set_name
                        }

        best_models_per_decl[did] = best_candidate
        save_model_artifact(
            model=best_candidate["model"],
            model_family="XGBoost",
            declaration=did,
            feature_set=feature_set_name,
            hyperparameters=best_candidate["params"],
            val_metrics=best_candidate["metrics"],
            target_mapping={str(cls): int(idx) for idx, cls in enumerate(le.classes_)},
            random_seed=random_seed
        )
        print(f"  [{did}] Best XGB: n_est={best_candidate['params'].get('n_estimators')}, "
              f"md={best_candidate['params'].get('max_depth')}, lr={best_candidate['params'].get('learning_rate')} -> "
              f"Val Macro F1: {best_candidate['metrics']['macro_f1']}, Acc: {best_candidate['metrics']['accuracy']}")

    return best_models_per_decl
