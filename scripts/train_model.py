"""
scripts/train_model.py
======================
Reproducible training script for the NitiMark Product Compliance ML model.

This script re-trains the ProductComplianceModel using the existing processed
dataset at data/processed/product_compliance_training.jsonl and persists the
best model to models/product_compliance/best_product_compliance_model.joblib.

Usage
-----
    python scripts/train_model.py

The existing model artifact is already trained and frozen. Run this script only
if you need to retrain from scratch (e.g., with new data or updated features).

WARNING: Overwriting the existing model will replace the frozen production artifact.
         Back up models/product_compliance/ before running.

Dataset
-------
    data/processed/product_compliance_training.jsonl
    - Contains Flipkart catalog records and YOLO physical packaging records
    - Unified ProductComplianceTrainingRecord format (JSONL)
    - Labels: COMPLIANT, NON_COMPLIANT

Model
-----
    Algorithm : RandomForestClassifier
    Feature Set: SET_A (16 features — statutory declaration presence & confidence)
    Features  : d01_present, d01_confidence, ..., d07_confidence,
                mandatory_declarations_present_count, all_mandatory_present

Saved Artifacts
---------------
    models/product_compliance/best_product_compliance_model.joblib
    models/product_compliance/model_registry.json
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.models.product_compliance_model import ProductComplianceModel

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "product_compliance_training.jsonl"
OUTPUT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "product_compliance" / "best_product_compliance_model.joblib"
)
REGISTRY_PATH = PROJECT_ROOT / "models" / "product_compliance" / "model_registry.json"
FEATURE_SET = "SET_A"
RANDOM_SEED = 42
TEST_SIZE = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[ProductComplianceTrainingRecord]:
    """Load ProductComplianceTrainingRecord objects from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Ensure data/processed/product_compliance_training.jsonl exists."
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(ProductComplianceTrainingRecord.from_dict(data))
            except Exception as exc:
                logger.warning("Skipping line %d due to parse error: %s", line_no, exc)
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def train(records: list[ProductComplianceTrainingRecord]) -> dict:
    """Train RandomForestClassifier on SET_A features and return metrics + model."""
    extractor = ProductComplianceFeatureExtractor(default_feature_set=FEATURE_SET)
    label_to_int = {"COMPLIANT": 0, "NON_COMPLIANT": 1}

    # Filter to labelled records only
    labelled = [
        r for r in records
        if r.overall_target in label_to_int and r.target_provenance != "UNKNOWN"
    ]
    logger.info("Labelled records: %d / %d total", len(labelled), len(records))

    if len(labelled) < 10:
        raise ValueError(
            f"Insufficient labelled data for training: {len(labelled)} records. Need ≥ 10."
        )

    X = extractor.transform(labelled, feature_set=FEATURE_SET)
    y_str = [r.overall_target for r in labelled]
    y = np.array([label_to_int[lbl] for lbl in y_str])
    classes = ["COMPLIANT", "NON_COMPLIANT"]

    # Stratified train/validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(X, y))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    logger.info("Train: %d | Val: %d", len(X_train), len(X_val))

    # Train RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_val)
    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_val, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_val, y_pred, average="macro", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_val, y_pred)), 4),
        "compliant_precision": round(
            float(precision_score(y_val, y_pred, pos_label=0, zero_division=0)), 4
        ),
        "compliant_recall": round(
            float(recall_score(y_val, y_pred, pos_label=0, zero_division=0)), 4
        ),
        "non_compliant_precision": round(
            float(precision_score(y_val, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "non_compliant_recall": round(
            float(recall_score(y_val, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "support": {
            "COMPLIANT": int(np.sum(y_val == 0)),
            "NON_COMPLIANT": int(np.sum(y_val == 1)),
        },
    }
    logger.info("Validation metrics: %s", json.dumps(metrics, indent=2))

    # Wrap in ProductComplianceModel
    feature_names = extractor.get_feature_names(FEATURE_SET)
    model = ProductComplianceModel(
        base_estimator=rf,
        feature_set=FEATURE_SET,
        feature_names=feature_names,
        classes=classes,
        abstention_threshold=0.65,
        margin_threshold=0.20,
        model_name="RandomForest",
        model_version="v2.0-product-compliance",
    )

    return {"model": model, "metrics": metrics, "feature_names": feature_names}


def save_artifacts(model: ProductComplianceModel, metrics: dict, feature_names: list) -> None:
    """Persist model and update model registry."""
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(OUTPUT_MODEL_PATH))
    logger.info("Model saved to: %s", OUTPUT_MODEL_PATH)

    registry = {
        "best_model_id": "RandomForest_SET_A",
        "best_model_path": str(OUTPUT_MODEL_PATH),
        "feature_set": FEATURE_SET,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "validation_metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated at: %s", REGISTRY_PATH)


def main() -> None:
    logger.info("=" * 60)
    logger.info("NitiMark — Product Compliance Model Training Script")
    logger.info("=" * 60)
    logger.info("Dataset  : %s", DATA_PATH)
    logger.info("Output   : %s", OUTPUT_MODEL_PATH)
    logger.info("Feature  : %s", FEATURE_SET)

    records = load_dataset(DATA_PATH)
    result = train(records)
    save_artifacts(result["model"], result["metrics"], result["feature_names"])

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(
        "Final Balanced Accuracy: %.4f | Macro F1: %.4f",
        result["metrics"]["balanced_accuracy"],
        result["metrics"]["macro_f1"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
