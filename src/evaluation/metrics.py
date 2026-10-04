"""
metrics.py
===========
Multi-class and multi-label evaluation metrics for Legal Metrology prediction.
Computes overall, macro, weighted, balanced accuracy, per-class metrics,
and labeled confusion matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix
)


ALL_POSSIBLE_STATES = ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"]


def evaluate_declaration_predictions(
    y_true: List[str] | np.ndarray | pd.Series,
    y_pred: List[str] | np.ndarray | pd.Series,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for a single declaration prediction task.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        observed_labels = sorted(list(set(y_true).union(set(y_pred))))
    else:
        observed_labels = list(labels)

    # If only 1 label observed across true and pred
    if len(observed_labels) <= 1:
        acc = float(accuracy_score(y_true, y_pred))
        cm = pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=observed_labels),
            index=[f"True_{l}" for l in observed_labels],
            columns=[f"Pred_{l}" for l in observed_labels]
        )
        per_class = {}
        for lbl in observed_labels:
            supp = int((y_true == lbl).sum())
            per_class[lbl] = {
                "precision": 1.0 if acc == 1.0 else 0.0,
                "recall": 1.0 if acc == 1.0 else 0.0,
                "f1": 1.0 if acc == 1.0 else 0.0,
                "support": supp
            }
        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(acc, 4),
            "macro_recall": round(acc, 4),
            "macro_f1": round(acc, 4),
            "weighted_precision": round(acc, 4),
            "weighted_recall": round(acc, 4),
            "weighted_f1": round(acc, 4),
            "balanced_accuracy": round(acc, 4),
            "per_class": per_class,
            "confusion_matrix": cm,
            "labels": observed_labels
        }

    # Standard multi-class evaluation
    acc = float(accuracy_score(y_true, y_pred))
    
    # Balanced accuracy (handles single class or zero division gracefully)
    try:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal_acc = acc

    # Macro & Weighted metrics with zero_division=0
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=observed_labels, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=observed_labels, average="weighted", zero_division=0
    )

    # Per-class metrics
    prec_cls, rec_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=observed_labels, average=None, zero_division=0
    )

    per_class = {}
    for i, lbl in enumerate(observed_labels):
        supp = int(supp_cls[i])
        per_class[lbl] = {
            "precision": round(float(prec_cls[i]), 4),
            "recall": round(float(rec_cls[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "support": supp
        }

    # Confusion Matrix
    cm_arr = confusion_matrix(y_true, y_pred, labels=observed_labels)
    cm_df = pd.DataFrame(cm_arr, index=[f"True_{l}" for l in observed_labels], columns=[f"Pred_{l}" for l in observed_labels])

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "per_class": per_class,
        "confusion_matrix": cm_df,
        "labels": observed_labels
    }
