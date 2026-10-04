"""
yolo_viewer.py
==============
Component for rendering YOLOv8 visual packaging detection results and bounding box annotations.
Defensively handles missing models, empty detections, and multiple images.
"""

import streamlit as st
import pandas as pd
from PIL import Image
from typing import Dict, Any, Optional
from app.utils.visualizer import draw_bounding_boxes, CLASS_COLORS


def render_yolo_viewer(original_image: Optional[Image.Image], result: Dict[str, Any]):
    """
    Renders the annotated bounding box image alongside the detections table.
    """
    if not isinstance(result, dict) or original_image is None:
        return

    yolo_data = result.get("yolo", {}) or {}
    trace = result.get("trace", {}) or {}
    detections = yolo_data.get("detections") or trace.get("yolo_detections") or []
    det_count = len(detections)

    st.subheader("🎯 YOLOv8 Packaging Visual Localization")
    
    col_img, col_info = st.columns([1.2, 1])

    with col_img:
        try:
            annotated_img = draw_bounding_boxes(original_image, detections)
            st.image(annotated_img, caption=f"YOLO Packaging Detections ({det_count} regions found)", use_container_width=True)
        except Exception as e:
            st.image(original_image, caption="Original Image (Annotation Rendering Error)", use_container_width=True)
            st.caption(f"Note: Could not draw bounding boxes: {e}")

    with col_info:
        st.markdown(f"**Detected Declaration Zones ({det_count} Total)**")
        if not detections:
            st.info("No packaging declaration bounding boxes detected with confidence $\ge 0.25$. Note: Invariant holds — absence of YOLO bounding boxes does not mean declaration absence if text is transcribed.")
        else:
            rows = []
            for i, d in enumerate(detections, start=1):
                if not isinstance(d, dict):
                    continue
                cls = str(d.get("class_name", d.get("class", "unknown"))).lower()
                try:
                    conf = float(d.get("confidence", 0.0) or 0.0)
                except (ValueError, TypeError):
                    conf = 0.0
                bbox = d.get("bbox") or d.get("bbox_normalized", [])
                
                rows.append({
                    "#": i,
                    "Class": cls.upper(),
                    "Confidence": f"{conf:.2f}",
                    "Bounding Box": f"[{', '.join(f'{v:.1f}' for v in bbox)}]" if (isinstance(bbox, (list, tuple)) and len(bbox) == 4) else "N/A"
                })
            
            df_dets = pd.DataFrame(rows)
            st.dataframe(df_dets, use_container_width=True, hide_index=True)

        st.caption("YOLOv8 Model: `yolov8n-packaging` (9 classes: Brand, MRP, Weight, Address, Consumer, Origin, Details, Tax, Barcode).")
