"""
active_learning_selector.py
===========================
Active Learning sample selector using margin and uncertainty sampling.
Selects the most informative, borderline, and ambiguous product samples
from unannotated candidate pools for minimal-effort human review.

WORKFLOW:
1. Automated batch feature extraction across candidate pool
2. Margin & Uncertainty ranking:
   - Margin sampling: |P(C) - P(NC)| < margin_threshold
   - High uncertainty: score >= 0.40 or abstention triggered
   - Borderline declaration states (PARTIAL / CANNOT_DETERMINE)
3. Outputs pre-filled AI annotation draft for human verification (NOT blank annotation)
4. Saves selected batch to data/interim/active_learning_selected_batch.jsonl
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.models.product_compliance_model import ProductComplianceModel
from src.extraction.declaration_extractor import DeclarationExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ActiveLearningSelector")

RAW_FLIPKART_CSV = PROJECT_ROOT / "data/raw/flipkart/flipkart_com-ecommerce_sample.csv"
CANDIDATE_POOL_CSV = PROJECT_ROOT / "data/interim/automated_candidate_pool.csv"
MODEL_PATH = PROJECT_ROOT / "models/product_compliance/best_product_compliance_model.joblib"
OUTPUT_BATCH_JSONL = PROJECT_ROOT / "data/interim/active_learning_selected_batch.jsonl"
REPORTS_PATH = PROJECT_ROOT / "reports/product_compliance/active_learning_selection_report.json"


class ActiveLearningSelector:
    """
    Selects high-value samples for human-in-the-loop review using uncertainty and margin sampling.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        margin_threshold: float = 0.25,
        uncertainty_threshold: float = 0.40
    ):
        self.model_path = model_path or MODEL_PATH
        self.margin_threshold = margin_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.model = None
        self.feature_extractor = ProductComplianceFeatureExtractor()
        self.extractor = DeclarationExtractor()
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            self.model = ProductComplianceModel.load(str(self.model_path))
            logger.info(f"Loaded compliance model: {self.model.model_name} (version: {self.model.model_version})")
        else:
            logger.warning(f"Model not found at {self.model_path}. Operating in rule-based uncertainty mode.")

    def score_candidate(self, record: ProductComplianceTrainingRecord) -> Dict[str, Any]:
        """
        Scores a candidate record on uncertainty, margin, and evidentiary richness.
        """
        # 1. Feature extraction
        feat_vector = self.feature_extractor.transform([record], feature_set="SET_A")

        # 1. Model prediction & margin
        if self.model and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat_vector)[0]
            # Assumes classes ['COMPLIANT', 'NON_COMPLIANT']
            p_comp = float(probs[0]) if len(probs) > 0 else 0.5
            p_nonc = float(probs[1]) if len(probs) > 1 else 0.5
            margin = abs(p_comp - p_nonc)
            top_prob = max(p_comp, p_nonc)
            confidence = round(top_prob * float(record.ocr_quality), 4)
            uncertainty_score = round(1.0 - confidence, 4)
            pred_class = "COMPLIANT" if p_comp >= p_nonc else "NON_COMPLIANT"
        else:
            p_comp, p_nonc = 0.5, 0.5
            margin = 0.0
            confidence = 0.5
            uncertainty_score = 0.5
            pred_class = "UNCERTAIN"

        # 2. Ambiguity signals in declarations
        decls = record.declarations or {}
        partial_cnt = sum(1 for d in decls.values() if d.get("status") in ["PARTIAL", "CANNOT_DETERMINE"])
        present_cnt = sum(1 for d in decls.values() if d.get("status") == "PRESENT")

        # Active learning informativeness score:
        # Higher score = more uncertain / more borderline / higher priority for human verification
        margin_score = max(0.0, 1.0 - (margin / self.margin_threshold))
        ambiguity_score = (partial_cnt / 7.0) * 0.5 + (0.5 if present_cnt in [3, 4, 5] else 0.0)
        informative_score = round((margin_score * 0.6) + (ambiguity_score * 0.4), 4)

        return {
            "product_id": record.product_id,
            "product_title": record.product_title,
            "category": record.category,
            "pred_class": pred_class,
            "p_compliant": round(p_comp, 4),
            "p_non_compliant": round(p_nonc, 4),
            "margin": round(margin, 4),
            "confidence": confidence,
            "uncertainty_score": uncertainty_score,
            "informative_score": informative_score,
            "partial_count": partial_cnt,
            "present_count": present_cnt,
            "is_informative": (margin <= self.margin_threshold or uncertainty_score >= self.uncertainty_threshold or partial_cnt >= 2)
        }

    def select_batch(self, candidate_records: List[ProductComplianceTrainingRecord], batch_size: int = 40) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Ranks candidates by informativeness and selects the top batch for human review.
        """
        logger.info(f"Scoring {len(candidate_records)} candidates for active learning selection...")
        scored = []
        for r in candidate_records:
            sc = self.score_candidate(r)
            sc["_record"] = r
            scored.append(sc)

        # Sort descending by informativeness
        scored.sort(key=lambda x: x["informative_score"], reverse=True)
        selected = scored[:batch_size]

        # Format output batch with pre-filled AI draft for human verification
        output_records = []
        for s in selected:
            rec = s["_record"]
            draft_record = {
                "product_id": rec.product_id,
                "source_dataset": rec.source_dataset,
                "product_title": rec.product_title,
                "category": rec.category,
                "raw_text": rec.raw_text,
                "ai_suggested_target": s["pred_class"],
                "ai_confidence": s["confidence"],
                "ai_margin": s["margin"],
                "ai_uncertainty_score": s["uncertainty_score"],
                "ai_informative_score": s["informative_score"],
                "draft_declarations": rec.declarations,
                "draft_d01_values": rec.d01_values,
                "draft_d02_values": rec.d02_values,
                "draft_d03_values": rec.d03_values,
                "draft_d04_values": rec.d04_values,
                "draft_d05_values": rec.d05_values,
                "draft_d06_values": rec.d06_values,
                "draft_d07_values": rec.d07_values,
                "human_verified": False,
                "human_verified_target": None,  # To be filled by human reviewer
                "human_review_notes": "",
                "selection_timestamp": datetime.now(timezone.utc).isoformat()
            }
            output_records.append(draft_record)

        # Build summary report
        report = {
            "total_candidates_evaluated": len(candidate_records),
            "batch_size_selected": len(output_records),
            "margin_threshold": self.margin_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "average_informative_score": round(float(np.mean([s["informative_score"] for s in selected])), 4) if selected else 0.0,
            "average_margin": round(float(np.mean([s["margin"] for s in selected])), 4) if selected else 0.0,
            "class_distribution_suggested": dict(Counter(s["pred_class"] for s in selected)),
            "output_path": str(OUTPUT_BATCH_JSONL)
        }

        # Write selected batch
        OUTPUT_BATCH_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_BATCH_JSONL, "w", encoding="utf-8") as f:
            for r in output_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved {len(output_records)} active learning selected candidates to {OUTPUT_BATCH_JSONL}")
        return output_records, report


def run_active_learning_selection(sample_limit: int = 500, batch_size: int = 40):
    """CLI runner to select informative batch from Flipkart candidates."""
    selector = ActiveLearningSelector()

    # Load candidate pool from Flipkart
    records = []
    if RAW_FLIPKART_CSV.exists():
        df = pd.read_csv(RAW_FLIPKART_CSV, nrows=sample_limit)
        extractor = DeclarationExtractor()
        for _, row in df.iterrows():
            pid = str(row.get("uniq_id", f"prod_{_}"))
            title = str(row.get("product_name", "") or "")
            desc = str(row.get("description", "") or "")
            specs = str(row.get("product_specifications", "") or "")
            comb_text = f"{title}\n{desc}\n{specs}"

            decls = extractor.extract_from_text_and_vision(
                raw_ocr_text=comb_text,
                title=title,
                description=desc,
                ocr_confidence=0.85
            )
            d_dict = {did: d.to_dict() for did, d in decls.items()}

            rec = ProductComplianceTrainingRecord(
                product_id=pid,
                source_dataset="flipkart_candidate_pool",
                target_provenance="CANDIDATE_POOL",
                overall_target="UNCERTAIN",
                category=str(row.get("product_category_tree", "General")).split(">>")[0].strip("['\" "),
                brand=str(row.get("brand", "")),
                product_title=title,
                raw_text=comb_text,
                declarations=d_dict,
                d01_values=decls["D01"].normalized_value,
                d02_values=decls["D02"].normalized_value,
                d03_values=decls["D03"].normalized_value,
                d04_values=decls["D04"].normalized_value,
                d05_values=decls["D05"].normalized_value,
                d06_values=decls["D06"].normalized_value,
                d07_values=decls["D07"].normalized_value,
                packaging_features={"yolo_detection_count": 0, "yolo_class_counts": {}, "yolo_total_bbox_area": 0.0},
                ocr_quality=0.85,
                text_length=len(comb_text),
                image_count=1
            )
            records.append(rec)

    if records:
        selector.select_batch(records, batch_size=batch_size)


if __name__ == "__main__":
    run_active_learning_selection()
