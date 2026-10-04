"""
Modular YOLOv8 Packaging Visual Localizer.
Implements Phase 11F requirements:
- Robust model loading with CPU support and graceful fallbacks.
- Structured bounding box detections with pixel & normalized coordinates.
- Class mapping to statutory Legal Metrology packaging declaration zones (D01–D07).
- Error isolation (corrupt images, missing weights, empty detections).
- ROI extraction for downstream OCR/Vision evidence pipeline.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("YOLODetector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WORKSPACE_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "models/phase_11/yolov8_packaging.pt"
FALLBACK_MODEL_PATH = WORKSPACE_ROOT / "experiments/phase_11/yolo_train/packaging_yolov8n/weights/best.pt"

CLASS_NAMES = [
    "address",      # 0: D02 / D07 (Manufacturer/Packer Address)
    "barcode",      # 1: Non-statutory barcode
    "brand",        # 2: D01 (Brand Name / Identity)
    "consumer",     # 3: D06 (Consumer Care / Helpline)
    "details",      # 4: General Specification / Declarations
    "mrp details",  # 5: D05 (MRP Declarations)
    "origin",       # 6: D07 (Country of Origin)
    "tax",          # 7: D05 Support (Inclusive of all taxes)
    "weight"        # 8: D03 (Net Quantity / Weight)
]


class YOLOPackagingDetector:
    """
    Production-grade YOLOv8 packaging visual localization engine.
    Localizes statutory declaration zones without performing compliance classification.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        device: str = "cpu"
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.model_loaded = False
        self.model_path_used = None
        
        # Determine candidate model paths
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.append(DEFAULT_MODEL_PATH)
        candidates.append(FALLBACK_MODEL_PATH)
        candidates.append(Path("yolov8n.pt"))
        
        # Attempt to load model
        for p in candidates:
            if p.exists() or str(p) == "yolov8n.pt":
                try:
                    from ultralytics import YOLO
                    logger.info(f"Loading YOLO model from: {p}")
                    self.model = YOLO(str(p))
                    self.model_loaded = True
                    self.model_path_used = str(p)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load YOLO model from {p}: {e}")
                    
        if not self.model_loaded:
            logger.warning("No YOLO model could be loaded. Detector will operate in graceful fallback mode.")

    def detect(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Execute visual detection on a single image.
        Returns structured detections with coordinates, confidence, and latency.
        """
        start_time = time.perf_counter()
        
        # Load and validate PIL image
        pil_image = None
        img_w, img_h = 0, 0
        try:
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    return {
                        "status": "ERROR_FILE_NOT_FOUND",
                        "error": f"Image file does not exist: {img_path}",
                        "detections": [],
                        "detection_count": 0,
                        "latency_ms": 0.0
                    }
                pil_image = Image.open(img_path).convert("RGB")
            elif isinstance(image_input, Image.Image):
                pil_image = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                pil_image = Image.fromarray(image_input).convert("RGB")
            else:
                return {
                    "status": "ERROR_INVALID_INPUT_TYPE",
                    "error": f"Unsupported input type: {type(image_input)}",
                    "detections": [],
                    "detection_count": 0,
                    "latency_ms": 0.0
                }
            img_w, img_h = pil_image.size
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return {
                "status": "ERROR_CORRUPT_IMAGE",
                "error": str(e),
                "detections": [],
                "detection_count": 0,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0
            }
            
        # If model is not loaded, return empty detections gracefully
        if not self.model_loaded or self.model is None:
            return {
                "status": "WARNING_NO_MODEL_LOADED",
                "detections": [],
                "detection_count": 0,
                "image_width": img_w,
                "image_height": img_h,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                "model_path": None
            }
            
        # Run inference
        detections = []
        try:
            results = self.model.predict(
                source=pil_image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    
                    x1 = max(0, min(float(xyxy[0]), float(img_w)))
                    y1 = max(0, min(float(xyxy[1]), float(img_h)))
                    x2 = max(0, min(float(xyxy[2]), float(img_w)))
                    y2 = max(0, min(float(xyxy[3]), float(img_h)))
                    
                    x1_norm = x1 / max(1.0, float(img_w))
                    y1_norm = y1 / max(1.0, float(img_h))
                    x2_norm = x2 / max(1.0, float(img_w))
                    y2_norm = y2 / max(1.0, float(img_h))
                    
                    w_box = max(0.0, x2 - x1)
                    h_box = max(0.0, y2 - y1)
                    area = w_box * h_box
                    area_norm = (w_box / max(1.0, float(img_w))) * (h_box / max(1.0, float(img_h)))
                    
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                    
                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "bbox_normalized": [round(x1_norm, 4), round(y1_norm, 4), round(x2_norm, 4), round(y2_norm, 4)],
                        "area_pixels": round(area, 1),
                        "area_normalized": round(area_norm, 4)
                    })
                    
            # Sort detections by confidence descending
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            status = "SUCCESS"
        except Exception as e:
            logger.error(f"Inference error in YOLOPackagingDetector: {e}")
            status = f"ERROR_INFERENCE: {str(e)}"
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "status": status,
            "detections": detections,
            "detection_count": len(detections),
            "image_width": img_w,
            "image_height": img_h,
            "latency_ms": round(latency_ms, 2),
            "model_path": self.model_path_used
        }

    def extract_rois(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        detections: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract cropped PIL Image regions of interest (ROIs) for each detection.
        """
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(image_input).convert("RGB")
        else:
            return []
            
        if detections is None:
            res = self.detect(pil_image)
            detections = res.get("detections", [])
            
        rois = []
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                if x2 > x1 and y2 > y1:
                    crop = pil_image.crop((x1, y1, x2, y2))
                    rois.append({
                        "class_name": det.get("class_name"),
                        "confidence": det.get("confidence"),
                        "bbox": bbox,
                        "roi_image": crop
                    })
        return rois
