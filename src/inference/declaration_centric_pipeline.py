"""
declaration_centric_pipeline.py
================================
Authoritative declaration-centric end-to-end inference pipeline for NitiMark-ML.

DATA FLOW:
PHYSICAL PRODUCT IMAGE(S)
         ↓
  IMAGE VALIDATION
         ↓
VISUAL LOCALIZATION (YOLOv8) -> Packaging region supervision & crops
         ↓
   OCR / VISION -> Raw verbatim text transcription
         ↓
DECLARATION EXTRACTION -> D01-D07 structured evidence with preserved actual values
         ↓
MULTI-IMAGE FUSION -> Front/back/side evidence union, conflict detection
         ↓
DECLARATION FEATURES -> Sets A through E (semantic values, OCR quality, visual support)
         ↓
PRODUCT-LEVEL ML -> Supervised prediction with confidence-based abstention
         ↓
COMPLIANT / NON_COMPLIANT / UNCERTAIN
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from PIL import Image

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.image_validator import validate_product_image
from src.vision.yolo_detector import YOLOPackagingDetector
from src.inference.vision_ocr import GoogleGeminiVisionProvider, MockVisionOCRProvider
from src.extraction.declaration_schema import (
    ProductEvidenceRecord, DeclarationEvidence, DeclarationStatus, EvidenceSource
)
from src.extraction.declaration_extractor import DeclarationExtractor
from src.extraction.evidence_fusion import EvidenceFusionEngine
from src.models.declaration_model_predictor import DeclarationModelPredictor

logger = logging.getLogger("DeclarationCentricPipeline")


class DeclarationCentricPipeline:
    """
    Authoritative inference pipeline implementing Declaration-Centric Product Compliance ML.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        yolo_path: Optional[str] = None,
        feature_set: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_mock_ocr_fallback: bool = True
    ):
        # 1. Initialize YOLO Packaging Detector
        try:
            self.yolo_detector = YOLOPackagingDetector(model_path=yolo_path)
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}. Falling back to non-visual extraction.")
            self.yolo_detector = None

        # 2. Initialize Vision/OCR Providers
        self.gemini_provider = GoogleGeminiVisionProvider(api_key=api_key, model_name=model_name)
        self.mock_provider = MockVisionOCRProvider() if use_mock_ocr_fallback else None

        # 3. Initialize Declaration Extractor & Fusion Engine
        self.extractor = DeclarationExtractor()
        self.fusion_engine = EvidenceFusionEngine()

        # 4. Initialize ML Predictor
        self.predictor = DeclarationModelPredictor(
            model_path=model_path,
            feature_set=feature_set
        )

    def analyze_product(
        self,
        image_paths: Union[str, List[str]],
        product_id: Optional[str] = None,
        product_title: Optional[str] = None,
        category: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end product compliance inference over one or more packaging images.
        """
        pipeline_start = time.time()

        if model_name:
            self.gemini_provider.set_model(model_name)

        if isinstance(image_paths, (str, Path)):
            image_paths = [str(image_paths)]
        else:
            image_paths = [str(p) for p in image_paths]

        product_id = product_id or f"PROD_{int(time.time())}"
        per_image_evidence: List[ProductEvidenceRecord] = []
        yolo_detections_all: List[Dict[str, Any]] = []
        ocr_results_all: List[Dict[str, Any]] = []

        # 1. Process each image view independently
        for idx, img_path in enumerate(image_paths):
            image_id = f"image_{idx+1}"

            # A. Validate image
            val_res = validate_product_image(img_path)
            if not val_res.get("valid", True):
                logger.warning(f"Image {img_path} validation warning: {val_res.get('reasons')}")

            # B. YOLO Visual Localization (if available)
            yolo_dets = []
            if self.yolo_detector and self.yolo_detector.model_loaded:
                try:
                    det_res = self.yolo_detector.detect(img_path)
                    yolo_dets = det_res.get("detections", [])
                    yolo_detections_all.extend(yolo_dets)
                except Exception as e:
                    logger.warning(f"YOLO detection error on {img_path}: {e}")

            # C. OCR / Vision extraction
            raw_text = ""
            ocr_conf = 0.90
            ocr_src = "GEMINI_VISION"
            try:
                ocr_out = self.gemini_provider.extract_from_image(img_path)
                raw_text = ocr_out.get("extracted_text", "")
                ocr_conf = ocr_out.get("ocr_confidence", 0.90)
                ocr_src = ocr_out.get("source", "GEMINI_VISION")
            except Exception as e:
                logger.warning(f"Primary Vision OCR failed on {img_path}: {e}")
                if self.mock_provider:
                    ocr_out = self.mock_provider.extract_from_image(img_path)
                    raw_text = ocr_out.get("extracted_text", "")
                    ocr_conf = 0.85
                    ocr_src = "MOCK_OCR_FALLBACK"

            ocr_results_all.append({
                "image_id": image_id,
                "image_path": img_path,
                "extracted_text": raw_text,
                "ocr_confidence": ocr_conf,
                "source": ocr_src
            })

            # D. Declaration Extraction for this image view
            img_rec = self.extractor.extract_declarations(
                raw_text=raw_text,
                product_id=product_id,
                source_image=image_id,
                ocr_confidence=ocr_conf,
                visual_regions=yolo_dets
            )
            per_image_evidence.append(img_rec)

        # 2. Multi-Image Evidence Fusion
        fused_record = self.fusion_engine.fuse_product_evidence(
            records=per_image_evidence,
            product_id=product_id,
            product_title=product_title,
            category=category
        )

        # 3. Product-Level ML Prediction
        ml_result = self.predictor.predict(fused_record)

        total_latency_ms = (time.time() - pipeline_start) * 1000.0

        # 4. Formulate structured response
        response = {
            "overall_prediction": ml_result["predicted_label"],
            "raw_prediction": ml_result["raw_label"],
            "confidence": ml_result["confidence"],
            "uncertainty": ml_result["predicted_label"] == "UNCERTAIN",
            "abstention_applied": ml_result["abstention_applied"],
            "abstention_reason": ml_result["abstention_reason"],
            "decision_basis": ml_result["decision_basis"],
            "declarations": {
                did: decl.to_dict() for did, decl in fused_record.declarations.items()
            },
            "evidence_summary": {
                "present_count": sum(1 for d in fused_record.declarations.values() if d.status == DeclarationStatus.PRESENT),
                "absent_count": sum(1 for d in fused_record.declarations.values() if d.status == DeclarationStatus.ABSENT),
                "cannot_determine_count": sum(1 for d in fused_record.declarations.values() if d.status == DeclarationStatus.CANNOT_DETERMINE),
                "completeness_ratio": fused_record.declaration_completeness_ratio,
                "overall_ocr_confidence": fused_record.overall_ocr_confidence,
                "conflicts_detected": fused_record.conflicting_fields
            },
            "model_id": ml_result["model_name"],
            "model_version": ml_result["model_version"],
            "feature_pipeline_version": "v1.0-declaration-centric",
            "feature_set": ml_result["feature_set"],
            "trace": {
                "image_count": len(image_paths),
                "yolo_detections_count": len(yolo_detections_all),
                "ocr_sources_count": len(ocr_results_all),
                "declaration_features_count": ml_result["feature_count"],
                "yolo_detections": yolo_detections_all,
                "ocr_results": ocr_results_all,
                "feature_vector_sample": {
                    k: ml_result["feature_trace"][k] for k in list(ml_result["feature_trace"].keys())[:15]
                }
            },
            "latency_ms": round(total_latency_ms, 2)
        }

        return response
