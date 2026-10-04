"""
build_common_training_dataset.py
================================
Builds the unified Common Product Compliance Training Dataset.
Integrates:
1. Flipkart catalog metadata records (~1,663 samples)
2. YOLO physical packaging dataset (249 physical packaging images, 482 region annotations)

Outputs:
- data/processed/product_compliance_training.jsonl
- data/processed/product_compliance_training.csv
- reports/product_compliance/pretraining_target_audit.json
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.product_training_representation import (
    ProductComplianceTrainingRecord, TargetProvenance, DatasetSource
)
from src.extraction.declaration_extractor import DeclarationExtractor
from src.extraction.declaration_schema import DeclarationStatus, DECLARATION_DEFINITIONS

DATA_DIR = PROJECT_ROOT / "data/processed/ml_dataset_v1"
WEAK_JSONL = PROJECT_ROOT / "data/processed/flipkart_weak_labels_v1.jsonl"
RAW_CSV = PROJECT_ROOT / "data/raw/flipkart/flipkart_com-ecommerce_sample.csv"
YOLO_DIR = PROJECT_ROOT / "data/external/yolo_candidate_dataset"
HUMAN_REF_PATH = PROJECT_ROOT / "data/processed/pilot_ground_truth_75_v1.jsonl"
REAL_PROD_DIR = PROJECT_ROOT / "real_product_validation"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
REPORTS_DIR = PROJECT_ROOT / "reports/product_compliance"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

YOLO_CLASS_NAMES = [
    "address",      # 0: D01/D07
    "barcode",      # 1: Auxiliary
    "brand",        # 2: D01
    "consumer",     # 3: D06
    "details",      # 4: D02/Specs
    "mrp details",  # 5: D05
    "origin",       # 6: D07
    "tax",          # 7: D05
    "weight"        # 8: D03
]


def audit_pretraining_targets() -> Dict[str, Any]:
    """
    Executes a hard pre-training audit of target availability across all datasets.
    """
    print("=================================================================")
    print("   PRE-TRAINING TARGET & DATASET AVAILABILITY AUDIT             ")
    print("=================================================================\n")

    # 1. Flipkart Weak Dataset
    flipkart_count = 0
    flipkart_targets = {}
    if WEAK_JSONL.exists():
        with open(WEAK_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    flipkart_count += 1
                    t = r.get("overall_target", "UNKNOWN")
                    flipkart_targets[t] = flipkart_targets.get(t, 0) + 1

    # 2. YOLO Physical Packaging Dataset
    yolo_images = []
    for split in ["train", "valid", "test"]:
        img_dir = YOLO_DIR / split / "images"
        if img_dir.exists():
            yolo_images.extend(list(img_dir.glob("*.*")))

    yolo_count = len(yolo_images)

    # 3. Human Reference
    hr_count = 0
    if HUMAN_REF_PATH.exists():
        with open(HUMAN_REF_PATH, "r", encoding="utf-8") as f:
            hr_count = sum(1 for line in f if line.strip())

    # 4. Real Packaging Benchmark
    rp_count = len(glob.glob(str(REAL_PROD_DIR / "manual_reference/*.json")))

    audit_summary = {
        "flipkart_raw_records": 20002,
        "flipkart_weak_silver_records": flipkart_count,
        "flipkart_target_provenance": "WEAKLY_DERIVED_SILVER",
        "flipkart_targets_distribution": flipkart_targets,
        "yolo_physical_packaging_images": yolo_count,
        "yolo_raw_annotations_type": "PACKAGING_REGION_BOUNDING_BOXES (9 classes)",
        "yolo_raw_target_provenance": "YOLO_REGION_ANNOTATION (No pre-existing product compliance labels)",
        "human_reference_samples": hr_count,
        "human_reference_role": "IMMUTABLE_EVALUATION_ONLY (N=75)",
        "real_packaging_benchmark_samples": rp_count,
        "real_packaging_benchmark_role": "IMMUTABLE_EVALUATION_ONLY (N=20)",
        "audit_decision": "YOLO packaging data contributes physical packaging evidence and visual region features. Training targets for Flipkart are WEAKLY_DERIVED_SILVER."
    }

    print(f"Flipkart Weak Records: {flipkart_count} (Provenance: WEAKLY_DERIVED_SILVER)")
    print(f"YOLO Physical Packaging Images: {yolo_count} (Provenance: YOLO_REGION_ANNOTATION / Physical Evidence)")
    print(f"Human Reference Benchmark: {hr_count} (Evaluation-Only)")
    print(f"Real Packaging Benchmark: {rp_count} (Evaluation-Only)\n")

    with open(REPORTS_DIR / "pretraining_target_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return audit_summary


def build_common_records() -> List[ProductComplianceTrainingRecord]:
    """
    Constructs standardized ProductComplianceTrainingRecord objects for both
    Flipkart records and YOLO physical packaging records.
    """
    extractor = DeclarationExtractor()
    records: List[ProductComplianceTrainingRecord] = []

    # -------------------------------------------------------------
    # 1. Process Flipkart Catalog Records
    # -------------------------------------------------------------
    print("Processing Flipkart catalog records into common representation...")
    raw_text_by_id = {}
    if RAW_CSV.exists():
        df_raw = pd.read_csv(
            RAW_CSV,
            usecols=["uniq_id", "product_name", "description", "product_specifications", "retail_price", "discounted_price", "brand"]
        )
        for _, row in df_raw.iterrows():
            raw_text_by_id[row["uniq_id"]] = {
                "product_name": str(row.get("product_name", "") or ""),
                "description": str(row.get("description", "") or ""),
                "product_specifications": str(row.get("product_specifications", "") or ""),
                "retail_price": float(row.get("retail_price", 0.0) or 0.0),
                "discounted_price": float(row.get("discounted_price", 0.0) or 0.0),
                "brand": str(row.get("brand", "") or "")
            }

    if WEAK_JSONL.exists():
        with open(WEAK_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                w_rec = json.loads(line)
                pid = w_rec["product_id"]
                raw_t = raw_text_by_id.get(pid, {})
                title = raw_t.get("product_name") or w_rec.get("product_name", "")
                desc = raw_t.get("description") or w_rec.get("description", "")
                specs = raw_t.get("product_specifications") or w_rec.get("product_specifications", "")
                brand = raw_t.get("brand") or w_rec.get("brand", "")
                category = w_rec.get("category", "Other")
                target = w_rec.get("overall_target", "NON_COMPLIANT")

                comb_text = f"{title}\n{desc}\n{specs}"
                decls = extractor.extract_from_text_and_vision(
                    raw_ocr_text=comb_text,
                    title=title,
                    description=desc,
                    ocr_confidence=0.90
                )

                d_dict = {did: d.to_dict() for did, d in decls.items()}
                d01_v = decls["D01"].normalized_value
                d02_v = decls["D02"].normalized_value
                d03_v = decls["D03"].normalized_value
                d04_v = decls["D04"].normalized_value
                d05_v = decls["D05"].normalized_value
                d06_v = decls["D06"].normalized_value
                d07_v = decls["D07"].normalized_value

                rec = ProductComplianceTrainingRecord(
                    product_id=pid,
                    source_dataset=DatasetSource.FLIPKART.value,
                    target_provenance=TargetProvenance.WEAKLY_DERIVED_SILVER.value,
                    overall_target=target,
                    category=category,
                    brand=brand,
                    product_title=title,
                    raw_text=comb_text,
                    declarations=d_dict,
                    d01_values=d01_v,
                    d02_values=d02_v,
                    d03_values=d03_v,
                    d04_values=d04_v,
                    d05_values=d05_v,
                    d06_values=d06_v,
                    d07_values=d07_v,
                    packaging_features={
                        "yolo_region_count": 0,
                        "yolo_detected_classes": [],
                        "yolo_total_bbox_area": 0.0,
                        "has_physical_image": False
                    },
                    ocr_quality=0.90,
                    text_length=len(comb_text),
                    image_count=1,
                    metadata={"split": w_rec.get("split", "train")}
                )
                records.append(rec)

    # -------------------------------------------------------------
    # 2. Process YOLO Physical Packaging Dataset (249 images)
    # -------------------------------------------------------------
    print("Processing YOLO physical packaging images & annotations into common representation...")
    yolo_count = 0
    for split in ["train", "valid", "test"]:
        img_dir = YOLO_DIR / split / "images"
        lbl_dir = YOLO_DIR / split / "labels"
        if not img_dir.exists():
            continue

        for img_p in sorted(img_dir.glob("*.*")):
            stem = img_p.stem
            lbl_p = lbl_dir / f"{stem}.txt"
            
            # Read YOLO bounding-box annotations
            boxes = []
            detected_classes = []
            total_area = 0.0
            class_counts = {c: 0 for c in YOLO_CLASS_NAMES}

            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as lf:
                    for l_line in lf:
                        parts = l_line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            cname = YOLO_CLASS_NAMES[cid] if cid < len(YOLO_CLASS_NAMES) else f"class_{cid}"
                            detected_classes.append(cname)
                            class_counts[cname] = class_counts.get(cname, 0) + 1
                            area = w * h
                            total_area += area
                            boxes.append({
                                "class": cname,
                                "class_id": cid,
                                "bbox": [xc, yc, w, h],
                                "area": area
                            })

            # Create physical packaging evidence
            visual_regions_for_extractor = [
                {"class": b["class"], "confidence": 0.90, "bbox": b["bbox"]}
                for b in boxes
            ]

            # Parse filename for text / identity hints
            clean_name = stem.replace("ComputerVision_v11i_yolov8_", "").replace("Batch-Details_v1i_yolov8_", "").replace(".rf.", " ")
            clean_title = clean_name.split()[0] if clean_name else "Physical Package"

            # Extract structured declarations using visual region support + text
            decls = extractor.extract_from_text_and_vision(
                raw_ocr_text=f"Packaging sample {clean_name}",
                title=clean_title,
                visual_regions=visual_regions_for_extractor,
                image_name=img_p.name,
                ocr_confidence=0.85
            )

            d_dict = {did: d.to_dict() for did, d in decls.items()}
            
            # Target for physical packaging sample:
            # If mandatory statutory regions (MRP, Weight, Address) are present in packaging evidence -> COMPLIANT / NON_COMPLIANT weak target
            has_mandatory_regions = class_counts.get("mrp details", 0) > 0 and class_counts.get("weight", 0) > 0 and class_counts.get("address", 0) > 0
            yolo_target = "COMPLIANT" if has_mandatory_regions else "NON_COMPLIANT"

            rec = ProductComplianceTrainingRecord(
                product_id=f"YOLO_{split.upper()}_{stem[:16]}",
                source_dataset=DatasetSource.YOLO_PHYSICAL_PACKAGING.value,
                target_provenance=TargetProvenance.WEAKLY_DERIVED_SILVER.value,
                overall_target=yolo_target,
                category="Physical Packaging",
                brand=clean_title,
                product_title=f"Physical Package ({clean_title})",
                raw_text=f"Physical packaging image {img_p.name} with {len(boxes)} annotated packaging regions: {', '.join(detected_classes[:5])}",
                declarations=d_dict,
                d01_values=decls["D01"].normalized_value,
                d02_values=decls["D02"].normalized_value,
                d03_values=decls["D03"].normalized_value,
                d04_values=decls["D04"].normalized_value,
                d05_values=decls["D05"].normalized_value,
                d06_values=decls["D06"].normalized_value,
                d07_values=decls["D07"].normalized_value,
                packaging_features={
                    "yolo_region_count": len(boxes),
                    "yolo_detected_classes": list(set(detected_classes)),
                    "yolo_class_counts": class_counts,
                    "yolo_total_bbox_area": round(total_area, 4),
                    "has_physical_image": True,
                    "image_path": str(img_p)
                },
                ocr_quality=0.85,
                text_length=len(clean_name),
                image_count=1,
                image_paths=[str(img_p)],
                metadata={"split": split, "stem": stem}
            )
            records.append(rec)
            yolo_count += 1

    print(f"Successfully constructed {len(records)} total records (Flipkart: {len(records) - yolo_count}, YOLO Physical Packaging: {yolo_count}).")
    return records


def save_common_dataset(records: List[ProductComplianceTrainingRecord]):
    """Saves records to jsonl and csv format."""
    jsonl_path = PROCESSED_DIR / "product_compliance_training.jsonl"
    csv_path = PROCESSED_DIR / "product_compliance_training.csv"

    print(f"Writing {len(records)} records to {jsonl_path}...")
    rows = []
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in records:
            d = r.to_dict()
            jf.write(json.dumps(d) + "\n")
            
            # Flat row for CSV inspection
            rows.append({
                "product_id": r.product_id,
                "source_dataset": r.source_dataset,
                "target_provenance": r.target_provenance,
                "overall_target": r.overall_target,
                "category": r.category,
                "brand": r.brand,
                "yolo_region_count": r.packaging_features.get("yolo_region_count", 0),
                "has_physical_image": r.packaging_features.get("has_physical_image", False),
                "d01_status": r.declarations.get("D01", {}).get("status", "CANNOT_DETERMINE"),
                "d03_status": r.declarations.get("D03", {}).get("status", "CANNOT_DETERMINE"),
                "d05_status": r.declarations.get("D05", {}).get("status", "CANNOT_DETERMINE")
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Writing CSV overview to {csv_path}...")


if __name__ == "__main__":
    audit_pretraining_targets()
    records = build_common_records()
    save_common_dataset(records)
