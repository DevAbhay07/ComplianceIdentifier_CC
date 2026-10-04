"""
export.py
=========
Component for exporting and downloading sanitized product compliance analysis JSON.
Safely serializes complex, numpy, and dataclass objects into compliant JSON.
"""

import json
import streamlit as st
import numpy as np
from typing import Dict, Any


def _json_default_serializer(obj):
    """Custom serializer for numpy types, sets, and custom classes."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def render_export_button(result: Dict[str, Any]):
    """
    Renders download button with sanitized analysis JSON payload.
    """
    if not isinstance(result, dict):
        return

    # Create clean, serializable audit payload
    export_payload = {
        "project": "NitiMark-ML",
        "phase": "Product Compliance Inference & Audit",
        "overall_prediction": result.get("overall_prediction"),
        "confidence": result.get("confidence"),
        "uncertainty": result.get("uncertainty"),
        "primary_reason": result.get("primary_reason"),
        "decision_basis": result.get("decision_basis"),
        "declarations": result.get("declarations"),
        "evidence_summary": result.get("evidence_summary"),
        "yolo_summary": {
            "detection_count": result.get("yolo", {}).get("detection_count", 0) if isinstance(result.get("yolo"), dict) else 0,
            "detections": result.get("yolo", {}).get("detections", []) if isinstance(result.get("yolo"), dict) else []
        },
        "ocr_summary": {
            "status": result.get("ocr", {}).get("status") if isinstance(result.get("ocr"), dict) else "N/A",
            "confidence": result.get("ocr", {}).get("confidence") if isinstance(result.get("ocr"), dict) else 0.0,
            "raw_text_length": len(result.get("ocr", {}).get("raw_text", "")) if isinstance(result.get("ocr"), dict) else 0
        },
        "model_id": result.get("model_id"),
        "model_version": result.get("model_version"),
        "feature_set": result.get("feature_set"),
        "probabilities": result.get("probabilities")
    }

    try:
        json_str = json.dumps(export_payload, indent=2, default=_json_default_serializer)
    except Exception as e:
        json_str = json.dumps({"error": f"Failed to serialize audit payload: {e}"}, indent=2)

    st.download_button(
        label="📥 Download Analysis Audit JSON",
        data=json_str,
        file_name="nitimark_compliance_analysis.json",
        mime="application/json",
        use_container_width=True
    )
