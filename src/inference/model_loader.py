"""
model_loader.py
===============
Loads frozen Phase 8 ML model artifacts and baselines for declarations D01 through D07
without altering, retraining, or modifying model binaries.
"""

import os
import json
import joblib
from typing import Dict, Any, Optional

DECLARATIONS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]
FROZEN_MODELS_DIR = "models/phase_8"
REGISTRY_JSON_PATH = os.path.join(FROZEN_MODELS_DIR, "model_registry.json")

# Explicit fallback mapping defined by Phase 8 Model Freeze Record
FROZEN_SELECTION_SPEC = {
    "D01": {
        "model_family": "RandomForest",
        "model_type": "RandomForestClassifier",
        "feature_set": "FEATURE_SET_A",
        "is_baseline": False,
        "artifact_rel_path": "randomforest/D01/model.joblib",
        "constant_value": None
    },
    "D02": {
        "model_family": "MajorityBaseline",
        "model_type": "MAJORITY_BASELINE",
        "feature_set": "NONE",
        "is_baseline": True,
        "artifact_rel_path": None,
        "constant_value": "PRESENT"
    },
    "D03": {
        "model_family": "RandomForest",
        "model_type": "RandomForestClassifier",
        "feature_set": "FEATURE_SET_A",
        "is_baseline": False,
        "artifact_rel_path": "randomforest/D03/model.joblib",
        "constant_value": None
    },
    "D04": {
        "model_family": "MajorityBaseline",
        "model_type": "MAJORITY_BASELINE",
        "feature_set": "NONE",
        "is_baseline": True,
        "artifact_rel_path": None,
        "constant_value": "CANNOT_DETERMINE"
    },
    "D05": {
        "model_family": "MajorityBaseline",
        "model_type": "MAJORITY_BASELINE",
        "feature_set": "NONE",
        "is_baseline": True,
        "artifact_rel_path": None,
        "constant_value": "PRESENT"
    },
    "D06": {
        "model_family": "RandomForest",
        "model_type": "RandomForestClassifier",
        "feature_set": "FEATURE_SET_B",
        "is_baseline": False,
        "artifact_rel_path": "randomforest/D06/model.joblib",
        "constant_value": None
    },
    "D07": {
        "model_family": "MajorityBaseline",
        "model_type": "MAJORITY_BASELINE",
        "feature_set": "NONE",
        "is_baseline": True,
        "artifact_rel_path": None,
        "constant_value": "CANNOT_DETERMINE"
    }
}


def load_frozen_models(
    models_base_dir: str = FROZEN_MODELS_DIR
) -> Dict[str, Dict[str, Any]]:
    """
    Loads all frozen Phase 8 models into memory and returns their descriptors.

    Returns:
        Dict[did, Dict[str, Any]] mapping "D01"-"D07" to model objects and metadata.
    """
    loaded_models = {}

    for did, spec in FROZEN_SELECTION_SPEC.items():
        if spec["is_baseline"]:
            loaded_models[did] = {
                "declaration": did,
                "model_family": spec["model_family"],
                "model_type": spec["model_type"],
                "is_baseline": True,
                "feature_set": spec["feature_set"],
                "model_object": spec["constant_value"],
                "classes": [spec["constant_value"]],
                "artifact_path": "BASELINE_CONSTANT",
                "version": "phase_8_v1"
            }
            continue

        artifact_path = os.path.join(models_base_dir, spec["artifact_rel_path"])
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"FROZEN_MODEL_MISSING: Artifact for {did} not found at {artifact_path}. "
                "Ensure Phase 8 model training has completed."
            )

        model_obj = joblib.load(artifact_path)
        classes = list(getattr(model_obj, "classes_", []))

        loaded_models[did] = {
            "declaration": did,
            "model_family": spec["model_family"],
            "model_type": spec["model_type"],
            "is_baseline": False,
            "feature_set": spec["feature_set"],
            "model_object": model_obj,
            "classes": classes,
            "artifact_path": artifact_path,
            "version": "phase_8_v1"
        }

    return loaded_models
