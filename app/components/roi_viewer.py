"""
Component for displaying cropped Region-of-Interest (ROI) image patches from YOLO detections.
"""

import streamlit as st
from typing import Dict, Any, List


def render_roi_viewer(result: Dict[str, Any]):
    """
    Renders cropped ROI image patches extracted from YOLO detections.
    """
    yolo_data = result.get("yolo", {})
    roi_crops = yolo_data.get("roi_crops", [])

    with st.expander("🔍 Cropped Packaging ROIs (Region-of-Interest Patches)", expanded=False):
        if not roi_crops:
            st.info("No packaging declaration ROIs were cropped for this product image.")
            return

        st.caption("Cropped high-resolution image patches extracted from YOLO bounding boxes for focused OCR transcription.")

        cols = st.columns(min(len(roi_crops), 4))
        for i, crop_info in enumerate(roi_crops):
            col = cols[i % min(len(roi_crops), 4)]
            cls_name = crop_info.get("class_name", f"Region #{i+1}").upper()
            conf = float(crop_info.get("confidence", 0.0))
            roi_img = crop_info.get("roi_image")

            with col:
                if roi_img:
                    st.image(roi_img, caption=f"{cls_name} ({conf:.2f})", use_container_width=True)
                else:
                    st.write(f"**{cls_name}** (No image)")
