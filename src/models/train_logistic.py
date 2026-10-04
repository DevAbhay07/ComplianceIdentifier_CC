"""
train_logistic.py
==================
Trains and tunes Logistic Regression models for declarations D01 through D07
across Feature Set A (tabular) and Feature Set B (multi-modal).

Evaluates hyperparameters:
- C in [0.01, 0.1, 1.0, 10.0]
- class_weight in [None, 'balanced']
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import evaluate_declaration_predictions
from src.models.model_registry import log_experiment, save_model_artifact


DECLARATIONS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]


def train_logistic_grid(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    feature_set_name: str = "FEATURE_SET_B",
    random_seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Run grid search for Logistic Regression on each declaration.
    Returns best model and metrics per declaration selected strictly on Validation Macro F1.
    """
    print(f"\n--- Training Logistic Regression Models ({feature_set_name}) ---")
    
    C_values = [0.01, 0.1, 1.0, 10.0]
    class_weight_options = [None, "balanced"]

    best_models_per_decl = {}

    for did in DECLARATIONS:
        target_col = f"{did}_target"
        y_train = Y_train[target_col].values
        y_val = Y_val[target_col].values

        unique_train_classes = sorted(list(set(y_train)))

        # If training target has only 1 class (e.g. D02, D05)
        if len(unique_train_classes) <= 1:
            majority_class = unique_train_classes[0]
            val_preds = [majority_class] * len(y_val)
            val_metrics = evaluate_declaration_predictions(y_val, val_preds)
            
            # Simple dummy baseline representation
            dummy_model = LogisticRegression()
            # Fit on trivial 1-sample per class trick or store constant predictor
            best_models_per_decl[did] = {
                "model": majority_class,  # constant predictor
                "params": {"constant": majority_class},
                "metrics": val_metrics,
                "feature_set": feature_set_name
            }
            log_experiment(
                experiment_id=f"LR_{did}_{feature_set_name}_constant",
                model_family="LogisticRegression",
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

        best_f1 = -1.0
        best_candidate = None

        for C in C_values:
            for cw in class_weight_options:
                t0 = time.time()
                model = LogisticRegression(
                    C=C,
                    class_weight=cw,
                    max_iter=1000,
                    random_state=random_seed,
                    solver="lbfgs"
                )
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                val_preds = model.predict(X_val)
                val_metrics = evaluate_declaration_predictions(y_val, val_preds)

                exp_id = f"LR_{did}_{feature_set_name}_C{C}_cw{cw}"
                log_experiment(
                    experiment_id=exp_id,
                    model_family="LogisticRegression",
                    declaration=did,
                    feature_set=feature_set_name,
                    target_type="Multiclass",
                    hyperparameters={"C": C, "class_weight": cw, "max_iter": 1000},
                    class_weight=cw,
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
                        "model": model,
                        "params": {"C": C, "class_weight": cw, "max_iter": 1000},
                        "metrics": val_metrics,
                        "feature_set": feature_set_name
                    }

        best_models_per_decl[did] = best_candidate
        save_model_artifact(
            model=best_candidate["model"],
            model_family="LogisticRegression",
            declaration=did,
            feature_set=feature_set_name,
            hyperparameters=best_candidate["params"],
            val_metrics=best_candidate["metrics"],
            random_seed=random_seed
        )
        print(f"  [{did}] Best LR: C={best_candidate['params'].get('C')}, "
              f"cw={best_candidate['params'].get('class_weight')} -> "
              f"Val Macro F1: {best_candidate['metrics']['macro_f1']}, Acc: {best_candidate['metrics']['accuracy']}")

    return best_models_per_decl
