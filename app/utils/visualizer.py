"""
Visualizer utility for drawing color-coded YOLO packaging detection bounding boxes.
"""

from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont

CLASS_COLORS = {
    "brand": "#3b82f6",          # Blue
    "mrp details": "#10b981",    # Green
    "weight": "#8b5cf6",         # Purple
    "address": "#f59e0b",        # Amber
    "consumer": "#ef4444",       # Red
    "origin": "#06b6d4",         # Cyan
    "details": "#64748b",        # Slate
    "tax": "#ec4899",            # Pink
    "barcode": "#6b7280"         # Gray
}


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def draw_bounding_boxes(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    line_width: int = 3
) -> Image.Image:
    """
    Draws labeled, color-coded bounding boxes on a copy of the input PIL image.
    """
    annotated = image.copy()
    if not detections:
        return annotated

    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    for det in detections:
        cls_name = str(det.get("class_name", det.get("class", "unknown"))).lower().strip()
        conf = float(det.get("confidence", 0.0) or 0.0)
        bbox = det.get("bbox", det.get("bbox_normalized", []))

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        
        # Check if normalized coordinates (0..1)
        if max(x1, y1, x2, y2) <= 1.05:
            x1 = x1 * w
            y1 = y1 * h
            x2 = x2 * w
            y2 = y2 * h

        color_hex = CLASS_COLORS.get(cls_name, "#22c55e")
        color_rgb = hex_to_rgb(color_hex)

        # Draw bounding rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=line_width)

        # Label text
        label_text = f"{cls_name.upper()} {conf:.2f}"
        
        # Estimate text bbox
        text_w = len(label_text) * 7 + 8
        text_h = 16
        
        label_y1 = max(0, y1 - text_h)
        label_y2 = label_y1 + text_h
        label_x2 = min(w, x1 + text_w)

        # Background fill for readability
        draw.rectangle([x1, label_y1, label_x2, label_y2], fill=color_rgb)
        draw.text((x1 + 4, label_y1 + 1), label_text, fill="white")

    return annotated
