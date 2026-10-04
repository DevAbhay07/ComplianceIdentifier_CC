"""
why_explanation.py
==================
Component for rendering the 'Why this result?' evidence explanation.
Defensively counts declaration statuses across schema variations and explains multimodal signals.
"""

import streamlit as st
from typing import Dict, Any


def render_why_explanation(result: Dict[str, Any]):
    """
    Renders structured, evidence-backed explanation for the product compliance decision.
    """
    if not isinstance(result, dict):
        return

    verdict = str(result.get("overall_prediction", "UNCERTAIN") or "UNCERTAIN")
    primary_reason = str(result.get("primary_reason", "") or result.get("decision_basis", "") or "")
    if not primary_reason:
        primary_reason = "Evaluated against Rule 6 mandatory requirements."

    declarations = result.get("declarations", {}) or {}
    evidence_summary = result.get("evidence_summary", {}) or {}
    
    # YOLO information
    yolo_data = result.get("yolo", {}) or {}
    trace = result.get("trace", {}) or {}
    yolo_dets = yolo_data.get("detections", []) or trace.get("yolo_detections", []) or []
    detected_classes = [
        d.get("class_name", d.get("class", "")) 
        for d in yolo_dets if isinstance(d, dict)
    ]
    detected_classes_str = ", ".join(sorted(set(filter(None, detected_classes)))) or "None detected"

    # OCR information
    ocr_data = result.get("ocr", {}) or {}
    ocr_status = ocr_data.get("status", "SUCCESS")
    try:
        ocr_conf = float(ocr_data.get("confidence", 0.0) or 0.0)
    except (ValueError, TypeError):
        ocr_conf = 0.0
    ocr_raw_text = str(ocr_data.get("raw_text", "") or "")
    if not ocr_raw_text and "trace" in result:
        ocr_raw_text = str(trace.get("combined_ocr_text", "") or "")

    # Declaration counts - defensive across schema variants
    present_cnt = evidence_summary.get("present_count")
    absent_cnt = evidence_summary.get("absent_count")
    uncertain_cnt = evidence_summary.get("cannot_determine_count")

    if present_cnt is None:
        present_cnt = 0
        absent_cnt = 0
        uncertain_cnt = 0
        for d in declarations.values():
            if isinstance(d, dict):
                st_val = d.get("status") or d.get("prediction")
                if hasattr(st_val, "value"):
                    st_val = st_val.value
                st_val = str(st_val).upper()
                if st_val == "PRESENT":
                    present_cnt += 1
                elif st_val == "ABSENT":
                    absent_cnt += 1
                else:
                    uncertain_cnt += 1

    with st.expander("💡 Why this result? (Detailed Statutory Analysis)", expanded=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Statutory Findings")
            st.write(f"• **Core Legal Basis**: {primary_reason}")
            st.write(f"• **Mandatory Declarations Verified Present**: `{present_cnt} / 7`")
            st.write(f"• **Mandatory Declarations Identified Absent**: `{absent_cnt} / 7`")
            st.write(f"• **Unresolved / Ambiguous Declarations**: `{uncertain_cnt} / 7`")

        with col2:
            st.markdown("#### Multimodal Evidence Signals")
            st.write(f"• **YOLO Packaging Regions Found**: `{len(detected_classes)}` regions ({detected_classes_str})")
            st.write(f"• **OCR Quality / Confidence**: `{ocr_conf * 100:.1f}%` ({ocr_status})")
            st.write(f"• **Transcribed Text Length**: `{len(ocr_raw_text)}` characters")

        st.info(
            "ℹ️ **Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 6)** requires that all pre-packaged commodities carry unambiguous declarations of Manufacturer Identity, Commodity Name, Net Weight, MRP, Month/Year of Manufacture, and Consumer Care helpline."
        )
