"""
model_registry.py
==================
Model registry and experiment tracking manager for Phase 8.
Handles model serialization, metadata versioning, and validation logging.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


EXPERIMENT_CSV_PATH = "experiments/phase_8/experiment_registry.csv"
MODEL_REGISTRY_JSON_PATH = "models/phase_8/model_registry.json"


def log_experiment(
    experiment_id: str,
    model_family: str,
    declaration: str,
    feature_set: str,
    target_type: str,
    hyperparameters: Dict[str, Any],
    class_weight: Any,
    sample_weight: bool,
    val_macro_f1: float,
    val_weighted_f1: float,
    val_accuracy: float,
    training_time_sec: float,
    status: str = "EXPERIMENT",
    registry_csv: Optional[str] = None
):
    """Append experiment entry to experiment_registry.csv."""
    csv_path = registry_csv or EXPERIMENT_CSV_PATH
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    
    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_family": model_family,
        "declaration": declaration,
        "feature_set": feature_set,
        "target_type": target_type,
        "hyperparameters": json.dumps(hyperparameters),
        "class_weight": str(class_weight),
        "sample_weight": sample_weight,
        "validation_macro_f1": round(val_macro_f1, 4),
        "validation_weighted_f1": round(val_weighted_f1, 4),
        "validation_accuracy": round(val_accuracy, 4),
        "training_time_sec": round(training_time_sec, 4),
        "status": status
    }

    df_row = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)


def save_model_artifact(
    model: Any,
    model_family: str,
    declaration: str,
    feature_set: str,
    hyperparameters: Dict[str, Any],
    val_metrics: Dict[str, Any],
    target_mapping: Optional[Dict[str, int]] = None,
    random_seed: int = 42,
    status: str = "CANDIDATE",
    models_dir: Optional[str] = None
) -> str:
    """Save trained model and its configuration metadata."""
    if models_dir is not None:
        save_dir = models_dir
    else:
        save_dir = f"models/phase_8/{model_family.lower()}/{declaration}"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"model_{declaration}_{model_family}_{feature_set}.joblib" if models_dir else "model.joblib")
    joblib.dump(model, model_path)

    metadata = {
        "model_family": model_family,
        "declaration": declaration,
        "feature_set": feature_set,
        "hyperparameters": hyperparameters,
        "target_mapping": target_mapping,
        "random_seed": random_seed,
        "validation_metrics": {
            "accuracy": val_metrics.get("accuracy"),
            "macro_f1": val_metrics.get("macro_f1"),
            "weighted_f1": val_metrics.get("weighted_f1"),
            "balanced_accuracy": val_metrics.get("balanced_accuracy")
        },
        "status": status,
        "training_dataset_version": "v1",
        "feature_pipeline_version": "v1",
        "training_timestamp": datetime.now(timezone.utc).isoformat()
    }

    meta_path = os.path.join(save_dir, "config.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Update central model registry JSON
    reg_json_path = os.path.join(models_dir, "model_registry.json") if models_dir else MODEL_REGISTRY_JSON_PATH
    update_model_registry(f"{model_family}_{declaration}_{feature_set}", metadata, registry_json_path=reg_json_path)
    return model_path


def update_model_registry(registry_key: str, entry: Dict[str, Any], registry_json_path: Optional[str] = None):
    json_path = registry_json_path or MODEL_REGISTRY_JSON_PATH
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    registry = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {}

    registry[registry_key] = entry
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def load_model_artifact(model_family: str, declaration: str) -> tuple[Any, Dict[str, Any]]:
    """Load model artifact with validation."""
    save_dir = f"models/phase_8/{model_family.lower()}/{declaration}"
    model_path = os.path.join(save_dir, "model.joblib")
    meta_path = os.path.join(save_dir, "config.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Config metadata not found at {meta_path}")

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return model, config
