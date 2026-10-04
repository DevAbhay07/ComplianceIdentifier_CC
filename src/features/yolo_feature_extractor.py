"""
YOLOv8 Feature Extractor for Phase 11 Multimodal Product Compliance Prediction.

Extracts structured visual features from YOLO packaging detections:
1. Detection Presence:
   - brand_detected
   - mrp_detected
   - net_weight_detected
   - address_detected
   - consumer_care_detected
   - country_origin_detected
   - product_details_detected
   - tax_info_detected
   - barcode_detected

2. Detection Confidence:
   - brand_detection_confidence
   - mrp_detection_confidence
   - net_weight_detection_confidence
   - address_detection_confidence
   - consumer_care_detection_confidence
   - country_origin_detection_confidence
   - product_details_detection_confidence
   - tax_info_detection_confidence
   - barcode_detection_confidence

3. Geometry Features:
   - total_region_area
   - normalized_region_area
   - max_region_width
   - max_region_height
   - mean_aspect_ratio

4. Aggregate Statistics:
   - total_detection_count
   - average_detection_confidence
   - maximum_detection_confidence
   - number_of_unique_classes
   - visual_evidence_coverage
   - text_region_density

Strict Anti-Leakage Guarantee:
Visual detections represent ROI bounding boxes and class detector scores.
They DO NOT encode legal compliance target states (D01-D07).
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union


class YOLOFeatureExtractor:
    """
    Extracts structured numerical features from YOLOv8 packaging detection outputs.
    """
    
    YOLO_CLASSES = [
        "brand",
        "mrp details",
        "weight",
        "address",
        "consumer",
        "origin",
        "details",
        "tax",
        "barcode"
    ]
    
    CLASS_CANONICAL_MAP = {
        "brand": "brand",
        "mrp": "mrp details",
        "mrp details": "mrp details",
        "weight": "weight",
        "net weight": "weight",
        "address": "address",
        "consumer": "consumer",
        "consumer care": "consumer",
        "origin": "origin",
        "country of origin": "origin",
        "details": "details",
        "product details": "details",
        "tax": "tax",
        "tax info": "tax",
        "barcode": "barcode"
    }

    def __init__(self):
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names = []
        # 1. Presence (9)
        for cls in ["brand", "mrp", "net_weight", "address", "consumer_care", "country_origin", "product_details", "tax_info", "barcode"]:
            names.append(f"{cls}_detected")
        # 2. Confidence (9)
        for cls in ["brand", "mrp", "net_weight", "address", "consumer_care", "country_origin", "product_details", "tax_info", "barcode"]:
            names.append(f"{cls}_detection_confidence")
        # 3. Geometry (5)
        names.extend([
            "total_region_area",
            "normalized_region_area",
            "max_region_width",
            "max_region_height",
            "mean_aspect_ratio"
        ])
        # 4. Aggregates (6)
        names.extend([
            "total_detection_count",
            "average_detection_confidence",
            "maximum_detection_confidence",
            "number_of_unique_classes",
            "visual_evidence_coverage",
            "text_region_density"
        ])
        return names

    def extract_from_detections(self, detections: List[Dict[str, Any]], image_width: int = 640, image_height: int = 640) -> Dict[str, float]:
        """
        Extract numerical YOLO features from a list of bounding box detection dicts.
        """
        feats = {name: 0.0 for name in self.feature_names}
        
        if not detections or not isinstance(detections, list):
            return feats

        # Track per-class detections and confidences
        class_detections: Dict[str, List[float]] = {cls: [] for cls in self.YOLO_CLASSES}
        box_areas = []
        box_widths = []
        box_heights = []
        aspect_ratios = []

        total_image_area = float(max(image_width * image_height, 1))

        for det in detections:
            if not isinstance(det, dict):
                continue
            raw_cls = str(det.get("class_name", det.get("class", "")) or "").lower().strip()
            canonical_cls = self.CLASS_CANONICAL_MAP.get(raw_cls, raw_cls)
            try:
                conf = float(det.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0

            if canonical_cls in class_detections and conf > 0:
                class_detections[canonical_cls].append(conf)

            bbox = det.get("bbox", det.get("bbox_normalized", []))
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    # Bounding box can be absolute [x1, y1, x2, y2] or normalized [0..1]
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    
                    # Check if normalized
                    if max(x1, y1, x2, y2) <= 1.05:
                        norm_area = w * h
                        abs_w = w * image_width
                        abs_h = h * image_height
                    else:
                        norm_area = (w * h) / total_image_area
                        abs_w = w
                        abs_h = h

                    box_areas.append(norm_area)
                    box_widths.append(abs_w)
                    box_heights.append(abs_h)
                    ar = abs_w / max(abs_h, 1e-4)
                    aspect_ratios.append(ar)
                except Exception:
                    pass

        # 1. Presence & 2. Confidence
        cls_key_map = {
            "brand": "brand",
            "mrp details": "mrp",
            "weight": "net_weight",
            "address": "address",
            "consumer": "consumer_care",
            "origin": "country_origin",
            "details": "product_details",
            "tax": "tax_info",
            "barcode": "barcode"
        }

        unique_detected_classes = 0
        all_confs = []

        for c_internal, c_prefix in cls_key_map.items():
            confs = class_detections[c_internal]
            if confs:
                feats[f"{c_prefix}_detected"] = 1.0
                feats[f"{c_prefix}_detection_confidence"] = float(max(confs))
                unique_detected_classes += 1
                all_confs.extend(confs)
            else:
                feats[f"{c_prefix}_detected"] = 0.0
                feats[f"{c_prefix}_detection_confidence"] = 0.0

        # 3. Geometry
        if box_areas:
            feats["total_region_area"] = float(sum(box_areas))
            feats["normalized_region_area"] = float(min(sum(box_areas), 1.0))
            feats["max_region_width"] = float(max(box_widths))
            feats["max_region_height"] = float(max(box_heights))
            feats["mean_aspect_ratio"] = float(np.mean(aspect_ratios))
        else:
            feats["total_region_area"] = 0.0
            feats["normalized_region_area"] = 0.0
            feats["max_region_width"] = 0.0
            feats["max_region_height"] = 0.0
            feats["mean_aspect_ratio"] = 0.0

        # 4. Aggregate Statistics
        total_detections = len(all_confs)
        feats["total_detection_count"] = float(total_detections)
        feats["average_detection_confidence"] = float(np.mean(all_confs)) if all_confs else 0.0
        feats["maximum_detection_confidence"] = float(max(all_confs)) if all_confs else 0.0
        feats["number_of_unique_classes"] = float(unique_detected_classes)
        feats["visual_evidence_coverage"] = float(unique_detected_classes / len(self.YOLO_CLASSES))
        feats["text_region_density"] = float(min(total_detections / 10.0, 1.0))

        return feats

    def extract_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical feature array from a dataframe with 'visual_regions' column.
        """
        matrix = []
        for _, row in df.iterrows():
            v_regions = row.get("visual_regions", [])
            if isinstance(v_regions, str):
                try:
                    v_regions = json.loads(v_regions)
                except Exception:
                    v_regions = []
            if not isinstance(v_regions, list):
                v_regions = []
            feats_dict = self.extract_from_detections(v_regions)
            matrix.append([feats_dict[name] for name in self.feature_names])
        return np.array(matrix, dtype=np.float32)
