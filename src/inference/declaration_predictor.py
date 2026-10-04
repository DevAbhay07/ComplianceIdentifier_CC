"""
declaration_predictor.py
========================
Executes inference on frozen Phase 8 models and baselines for Legal Metrology
declarations D01 through D07, returning predictions, probabilities, and model types.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.inference.uncertainty import compute_prediction_uncertainty

DECLARATIONS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]

DECLARATION_NAMES = {
    "D01": "Name & Address of Manufacturer / Packer / Importer",
    "D02": "Generic or Common Name of Commodity",
    "D03": "Net Quantity in Standard Units",
    "D04": "Date of Manufacture / Packing / Expiry",
    "D05": "Maximum Retail Price (MRP)",
    "D06": "Consumer Care Contact Details",
    "D07": "Country of Origin"
}


def predict_single_declaration(
    model_desc: Dict[str, Any],
    X_set_a: pd.DataFrame,
    X_set_b: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate prediction and uncertainty for a single declaration descriptor.
    """
    did = model_desc["declaration"]
    is_baseline = model_desc["is_baseline"]
    model_type = model_desc["model_type"]
    feature_set = model_desc["feature_set"]

    if is_baseline:
        pred_label = str(model_desc["model_object"])
        prob_dict = {pred_label: 1.0}
        unc = compute_prediction_uncertainty(prob_dict)

        return {
            "declaration_id": did,
            "declaration_name": DECLARATION_NAMES.get(did, did),
            "prediction": pred_label,
            "confidence": 1.0,
            "decision": "PREDICT",
            "probabilities": prob_dict,
            "margin": 1.0,
            "entropy": 0.0,
            "uncertainty_reason": None,
            "model_family": model_desc["model_family"],
            "model_type": model_type,
            "feature_set": feature_set,
            "is_baseline": True,
            "model_version": model_desc.get("version", "phase_8_v1")
        }

    # Trained ML Model
    model_obj = model_desc["model_object"]
    X = X_set_b if feature_set == "FEATURE_SET_B" else X_set_a

    # Generate predictions
    raw_pred = model_obj.predict(X)[0]
    pred_label = str(raw_pred)

    # Probabilities
    classes = list(getattr(model_obj, "classes_", [pred_label]))
    if hasattr(model_obj, "predict_proba"):
        raw_probs = model_obj.predict_proba(X)[0]
        prob_dict = {str(classes[i]): round(float(raw_probs[i]), 4) for i in range(len(classes))}
    else:
        prob_dict = {pred_label: 1.0}

    unc = compute_prediction_uncertainty(prob_dict)

    return {
        "declaration_id": did,
        "declaration_name": DECLARATION_NAMES.get(did, did),
        "prediction": pred_label,
        "confidence": unc["confidence"],
        "decision": unc["decision"],
        "probabilities": prob_dict,
        "margin": unc["margin"],
        "entropy": unc["entropy"],
        "uncertainty_reason": unc["uncertainty_reason"],
        "model_family": model_desc["model_family"],
        "model_type": model_type,
        "feature_set": feature_set,
        "is_baseline": False,
        "model_version": model_desc.get("version", "phase_8_v1")
    }


def predict_all_declarations(
    loaded_models: Dict[str, Dict[str, Any]],
    X_set_a: pd.DataFrame,
    X_set_b: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Executes prediction across all 7 declarations D01 through D07.
    """
    predictions = {}
    for did in DECLARATIONS:
        if did not in loaded_models:
            continue
        predictions[did] = predict_single_declaration(
            model_desc=loaded_models[did],
            X_set_a=X_set_a,
            X_set_b=X_set_b
        )
    return predictions
