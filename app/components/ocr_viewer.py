"""
ocr_viewer.py
=============
Component for displaying OCR / Vision extracted textual evidence.
Defensively renders text buffers, multiple image OCR transcriptions, and API statuses.
"""

import streamlit as st
from typing import Dict, Any


def render_ocr_viewer(result: Dict[str, Any]):
    """
    Renders structured OCR fields and raw transcribed text with non-fatal status notifications.
    """
    if not isinstance(result, dict):
        return

    ocr_data = result.get("ocr", {}) or {}
    trace = result.get("trace", {}) or {}
    
    raw_text = str(ocr_data.get("raw_text", "") or trace.get("combined_ocr_text", "") or "")
    status = str(ocr_data.get("status", "SUCCESS") or trace.get("ocr_call_status", "SUCCESS"))
    
    try:
        conf = float(ocr_data.get("confidence", trace.get("overall_ocr_confidence", 0.90)) or 0.90)
    except (ValueError, TypeError):
        conf = 0.90

    provider_info = ocr_data.get("provider_info", {}) or {}
    provider_name = provider_info.get("provider") or trace.get("ocr_provider", "GoogleGeminiVision")
    model_name = provider_info.get("model", "gemini-3.6-flash")

    st.subheader("📝 OCR / Vision Textual Extraction")

    if status in ("NO_API_KEY_FALLBACK", "FALLBACK_MOCK"):
        st.warning("⚠️ Vision API Key not configured. Running in fallback mode. Configure `VISION_API_KEY` in environment variables or sidebar for live Gemini Vision extraction.")
    elif status == "OFFLINE_MODE":
        st.info("ℹ️ Running in Offline / Benchmark Reference Mode.")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"**Provider**: `{provider_name}` (`{model_name}`)")
        st.markdown(f"**Transcription Status**: `{status}`")
        st.markdown(f"**Optical Clarity Confidence**: `{conf * 100:.1f}%`")
        st.markdown(f"**Extracted Text Volume**: `{len(raw_text)} characters / {len(raw_text.split())} words`")

    with col2:
        if raw_text:
            st.text_area("Transcribed Packaging Text Preview", raw_text, height=130, disabled=True)
        else:
            st.info("No optical text could be transcribed from the provided image(s).")

    with st.expander("📄 View Raw Full OCR Transcription Buffer", expanded=False):
        if raw_text:
            st.code(raw_text, language="text")
        else:
            st.write("Empty text buffer.")
