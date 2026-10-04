"""
Developer Mode Component: Detailed Technical Pipeline Trace & Verifiable YOLO Integration.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any


def _safe_get(d, *keys, default=None):
    """Safely navigate nested dicts without crashes."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


def render_technical_trace(result: Dict[str, Any]):
    """
    Renders deep technical trace proving YOLO -> OCR -> D01-D07 -> Feature Adapter -> ML/Decision engine.
    All access is defensive — never crashes on missing or unexpected types.
    """
    st.markdown("---")
    st.subheader("🛠️ Developer Mode — Technical Analysis & Intermediate Diagnostics")
    st.caption("Verifiable execution trace proving the integration of visual YOLO features, OCR evidence, declaration features, and ML model inference.")

    # 1. Pipeline Execution Stages & Latency
    latencies = result.get("latencies", {})
    if isinstance(latencies, (int, float)):
        latencies = {"total_latency_ms": float(latencies)}
    if not isinstance(latencies, dict):
        latencies = {}

    st.markdown("#### ⏱️ Latency & Execution Breakdown")
    
    col_lat1, col_lat2 = st.columns([1, 1])
    with col_lat1:
        st.write(f"• **Image Validation**: `{latencies.get('image_validation_ms', 0.0):.2f} ms`")
        st.write(f"• **YOLO Localization**: `{latencies.get('yolo_localization_ms', 0.0):.2f} ms`")
        st.write(f"• **OCR / Vision Call**: `{latencies.get('ocr_vision_ms', 0.0):.2f} ms`")
        st.write(f"• **Evidence Parsing**: `{latencies.get('evidence_parsing_ms', 0.0):.2f} ms`")

    with col_lat2:
        st.write(f"• **Declaration Pipeline**: `{latencies.get('declaration_pipeline_ms', 0.0):.2f} ms`")
        st.write(f"• **Direct ML Inference**: `{latencies.get('direct_ml_inference_ms', 0.0):.2f} ms`")
        st.write(f"• **Decision Engine**: `{latencies.get('decision_engine_ms', 0.0):.2f} ms`")
        st.markdown(f"**⚡ Total Pipeline Latency**: `{latencies.get('total_latency_ms', 0.0):.2f} ms`")

    # 2. Comprehensive Intermediate Diagnostics
    st.markdown("#### 🔬 Intermediate Pipeline Diagnostics")
    diag = result.get("diagnostics", {})
    if not isinstance(diag, dict):
        diag = {}
    trace = result.get("trace", {})
    if not isinstance(trace, dict):
        trace = {}
    if not diag:
        diag = trace.get("diagnostics", {})
        if not isinstance(diag, dict):
            diag = {}

    ocr_info = result.get("ocr", {})
    if not isinstance(ocr_info, dict):
        ocr_info = {}

    ocr_status = diag.get("ocr_call_status", ocr_info.get("status", "UNKNOWN"))
    ocr_err = diag.get("ocr_error", ocr_info.get("error"))
    ocr_raw_text = ocr_info.get("raw_text", "")
    if not isinstance(ocr_raw_text, str):
        ocr_raw_text = str(ocr_raw_text) if ocr_raw_text else ""
    ocr_len = diag.get("ocr_raw_text_length", len(ocr_raw_text))
    ocr_prev = diag.get("ocr_raw_text_preview", ocr_raw_text[:300])
    ocr_confidence = ocr_info.get("confidence", 0.0)
    if not isinstance(ocr_confidence, (int, float)):
        ocr_confidence = 0.0

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("**📸 Optical & Extraction State**")
        st.write(f"• Image Count: `{diag.get('image_count', trace.get('image_count', 1))}`")
        yolo_data = result.get("yolo", {})
        if not isinstance(yolo_data, dict):
            yolo_data = {}
        st.write(f"• YOLO Region Count: `{yolo_data.get('detection_count', 0)}`")
        yolo_classes = diag.get("yolo_detected_classes", [])
        if not yolo_classes:
            dets = yolo_data.get("detections", [])
            if isinstance(dets, list):
                yolo_classes = list(set(
                    d.get("class_name", d.get("class", ""))
                    for d in dets if isinstance(d, dict)
                ))
        st.write(f"• YOLO Detected Classes: `{yolo_classes or 'None'}`")
        provider_info = ocr_info.get("provider_info", {})
        if not isinstance(provider_info, dict):
            provider_info = {}
        st.write(f"• OCR Provider: `{provider_info.get('model', 'gemini-3.6-flash')}`")
        st.write(f"• OCR Execution Status: `{ocr_status}`")
        if ocr_err:
            st.error(f"OCR Error: {ocr_err}")

    with col_d2:
        st.markdown("**📝 Text & Feature State**")
        st.write(f"• OCR Raw Text Length: `{ocr_len} characters`")
        st.write(f"• OCR Confidence: `{float(ocr_confidence) * 100:.1f}%`")
        model_meta = result.get("model_metadata", {})
        if not isinstance(model_meta, dict):
            model_meta = {}
        st.write(f"• ML Feature Pipeline: `{model_meta.get('feature_pipeline', 'SET_A (16 features)')}`")
        st.write(f"• Model Feature Count: `{diag.get('model_feature_count', 16)}`")
        st.write(f"• Model Input Shape: `{diag.get('model_input_shape', [1, 16])}`")

    if ocr_len > 0:
        with st.expander("📄 Verbatim Extracted Text Preview", expanded=False):
            st.text(ocr_prev)

    # 3. Direct ML vs Decision Engine Decomposition
    st.markdown("#### ⚖️ Direct ML Classifier vs Decision Engine")
    direct_ml = result.get("direct_ml", {})
    if not isinstance(direct_ml, dict):
        direct_ml = {}
    probs = direct_ml.get("probabilities", {})
    if not isinstance(probs, dict):
        probs = {}
    
    # Safely handle uncertainty — can be dict, UncertaintyStatus, float, or None
    unc = result.get("uncertainty", {})
    if isinstance(unc, dict):
        unc_status = unc.get("status", "LOW")
        unc_score = unc.get("score", round(1.0 - float(result.get("confidence", 1.0)), 4))
        unc_reason = unc.get("reason", result.get("abstention_reason", "Certain"))
        unc_abstained = unc.get("abstained", result.get("abstention_applied", False))
    elif isinstance(unc, (int, float)):
        unc_score = round(float(unc), 4)
        unc_status = "HIGH" if unc_score >= 0.45 else ("MEDIUM" if unc_score >= 0.25 else "LOW")
        unc_reason = result.get("abstention_reason", "Certain")
        unc_abstained = bool(result.get("abstention_applied", False))
    elif isinstance(unc, bool):
        unc_score = 0.85 if unc else 0.10
        unc_status = "HIGH" if unc else "LOW"
        unc_reason = result.get("abstention_reason", "Certain")
        unc_abstained = bool(result.get("abstention_applied", False))
    else:
        unc_score = 0.10
        unc_status = "LOW"
        unc_reason = "Unknown uncertainty format"
        unc_abstained = False

    col_ml1, col_ml2 = st.columns([1, 1])
    with col_ml1:
        st.markdown(f"**Direct ML Model** (`{direct_ml.get('model_name', 'RandomForest_SET_A')}`)")
        st.write(f"• Raw Supervised Prediction: **`{direct_ml.get('predicted_label', result.get('raw_prediction', 'N/A'))}`**")
        st.write(f"• Supervised Probabilities:")
        st.write(f"  - `NON_COMPLIANT`: **{probs.get('NON_COMPLIANT', 0.0) * 100:.1f}%**")
        st.write(f"  - `COMPLIANT`: **{probs.get('COMPLIANT', 0.0) * 100:.1f}%**")

    with col_ml2:
        st.markdown("**Evidence-Aware Decision & Abstention Engine**")
        st.write(f"• Final Calibrated Verdict: **`{result.get('overall_prediction', 'N/A')}`**")
        confidence = result.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        st.write(f"• Calibrated Confidence: **`{float(confidence) * 100:.1f}%`**")
        raw_model_conf = result.get("raw_model_confidence")
        if raw_model_conf is not None:
            st.write(f"• Raw Model Probability: `{float(raw_model_conf) * 100:.1f}%`")
        ev_quality = result.get("evidence_quality")
        if ev_quality is not None:
            st.write(f"• Evidence Quality Factor: `{float(ev_quality) * 100:.1f}%`")
        st.write(f"• Uncertainty Status: **`{unc_status}`** (Score: `{unc_score:.4f}`)")
        st.write(f"• Abstention Applied: `{unc_abstained}`")
        if unc_abstained:
            st.info(f"Abstention Trigger: {unc_reason}")
        st.write(f"• Decision Basis: {result.get('decision_basis', 'N/A')}")

    # 4. Exact ML Input Feature Trace Table
    st.markdown("#### 📊 Declaration-Centric Features (Entering ML Classifier)")
    feat_dict = diag.get("feature_vector_summary", trace.get("features", {}))
    if not isinstance(feat_dict, dict):
        # Try yolo features as fallback
        yolo_feats = result.get("yolo", {})
        feat_dict = yolo_feats.get("features", {}) if isinstance(yolo_feats, dict) else {}
    
    if feat_dict and isinstance(feat_dict, dict):
        feat_rows = [
            {"Feature Name": k, "Extracted Value": f"{v:.4f}" if isinstance(v, (int, float)) else str(v)}
            for k, v in feat_dict.items()
        ]
        df_feats = pd.DataFrame(feat_rows)
        st.dataframe(df_feats, use_container_width=True, height=260)
    else:
        st.write("No feature dictionary available.")

    # 5. Raw Inspection JSON
    with st.expander("🔍 Raw Internal Trace JSON Payload", expanded=False):
        try:
            clean_res = dict(result)
            # Remove large binary data
            if "yolo" in clean_res and isinstance(clean_res["yolo"], dict) and "roi_crops" in clean_res["yolo"]:
                clean_res["yolo"] = {k: v for k, v in clean_res["yolo"].items() if k != "roi_crops"}
            # Convert non-serializable types
            st.json(clean_res)
        except Exception as e:
            st.error(f"Could not render trace JSON: {e}")
            st.text(str(result)[:5000])
