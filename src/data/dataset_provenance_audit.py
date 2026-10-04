"""
dataset_provenance_audit.py
===========================
Authoritative Dynamic Dataset Provenance & Target Availability Audit for NitiMark-ML.
Calculates exact counts from current workspace files without relying on stale reports.
"""

import os
import sys
import json
import glob
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

PROJECT_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data/processed"
COMMON_DATA_PATH = DATA_DIR / "product_compliance_training.jsonl"
YOLO_DIR = PROJECT_ROOT / "data/external/yolo_candidate_dataset"
HUMAN_REF_PATH = DATA_DIR / "pilot_ground_truth_75_v1.jsonl"
REAL_PROD_DIR = PROJECT_ROOT / "real_product_validation"
REPORTS_DIR = PROJECT_ROOT / "reports/product_compliance"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_dataset_provenance_audit() -> Dict[str, Any]:
    print("=================================================================")
    print("      NITIMARK-ML — AUTHORITATIVE DATASET PROVENANCE AUDIT       ")
    print("=================================================================\n")

    # 1. Inspect Common Training Representation
    common_records = []
    if COMMON_DATA_PATH.exists():
        with open(COMMON_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    common_records.append(json.loads(line))

    total_common = len(common_records)
    print(f"1. Common Training Representation Total: {total_common}")

    sources = Counter(r.get("source_dataset", "UNKNOWN") for r in common_records)
    targets = Counter(r.get("overall_target", "UNKNOWN") for r in common_records)
    provenances = Counter(r.get("target_provenance", "UNKNOWN") for r in common_records)

    source_target_matrix = Counter(
        (r.get("source_dataset", "UNKNOWN"), r.get("overall_target", "UNKNOWN"))
        for r in common_records
    )
    source_prov_matrix = Counter(
        (r.get("source_dataset", "UNKNOWN"), r.get("target_provenance", "UNKNOWN"))
        for r in common_records
    )

    print("\n--- Source Dataset Breakdown ---")
    for s, c in sources.items():
        print(f"  • {s}: {c}")

    print("\n--- Overall Target Breakdown ---")
    for t, c in targets.items():
        print(f"  • {t}: {c}")

    print("\n--- Target Provenance Breakdown ---")
    for p, c in provenances.items():
        print(f"  • {p}: {c}")

    print("\n--- Source x Target Matrix ---")
    for (s, t), c in sorted(source_target_matrix.items()):
        print(f"  • ({s}, {t}): {c}")

    print("\n--- Source x Target Provenance Matrix ---")
    for (s, p), c in sorted(source_prov_matrix.items()):
        print(f"  • ({s}, {p}): {c}")

    # 2. Inspect Raw YOLO Dataset Directory
    yolo_images = []
    yolo_labels = []
    for split in ["train", "valid", "test"]:
        s_img = YOLO_DIR / split / "images"
        s_lbl = YOLO_DIR / split / "labels"
        if s_img.exists():
            yolo_images.extend(list(s_img.glob("*.*")))
        if s_lbl.exists():
            yolo_labels.extend(list(s_lbl.glob("*.txt")))

    print(f"\n2. Raw YOLO Dataset on Disk:")
    print(f"  • Total Images on Disk: {len(yolo_images)}")
    print(f"  • Total Label Files on Disk: {len(yolo_labels)}")

    # 3. Audit YOLO COMPLIANT Records in Common Representation
    yolo_compliant_recs = [
        r for r in common_records
        if r.get("source_dataset") == "yolo_physical_packaging" and r.get("overall_target") == "COMPLIANT"
    ]
    print(f"\n3. Audit of Existing YOLO COMPLIANT Records ({len(yolo_compliant_recs)} found):")
    yolo_compliant_audit = []
    for i, r in enumerate(yolo_compliant_recs):
        item_info = {
            "index": i + 1,
            "product_id": r.get("product_id"),
            "category": r.get("category"),
            "brand": r.get("brand"),
            "target_provenance": r.get("target_provenance"),
            "overall_target": r.get("overall_target"),
            "packaging_features": r.get("packaging_features"),
            "declarations": {
                did: {"status": d.get("status"), "confidence": d.get("confidence")}
                for did, d in r.get("declarations", {}).items()
            }
        }
        yolo_compliant_audit.append(item_info)
        print(f"  [{i+1}] ID: {r.get('product_id')}")
        print(f"      Category: {r.get('category')}, Provenance: {r.get('target_provenance')}")
        print(f"      Packaging Image: {r.get('packaging_features', {}).get('image_path')}")
        print(f"      Detected Classes: {r.get('packaging_features', {}).get('yolo_detected_classes')}")

    # 4. Human Reference Benchmark & Real Packaging Benchmark
    hr_count = 0
    if HUMAN_REF_PATH.exists():
        with open(HUMAN_REF_PATH, "r", encoding="utf-8") as f:
            hr_count = sum(1 for line in f if line.strip())

    rp_count = len(glob.glob(str(REAL_PROD_DIR / "manual_reference/*.json")))
    rp_images = len(glob.glob(str(REAL_PROD_DIR / "images/*.png")))

    print(f"\n4. Benchmark Partitions:")
    print(f"  • Human Ground Truth 75 Benchmark: {hr_count} records")
    print(f"  • Real Product Validation 20 Benchmark: {rp_count} references, {rp_images} images")

    report_payload = {
        "timestamp": "2026-09-05T20:30:00Z",
        "common_representation": {
            "total_records": total_common,
            "sources": dict(sources),
            "targets": dict(targets),
            "provenances": dict(provenances),
            "source_target_matrix": {f"{s}::{t}": c for (s, t), c in source_target_matrix.items()},
            "source_provenance_matrix": {f"{s}::{p}": c for (s, p), c in source_prov_matrix.items()}
        },
        "yolo_raw_dataset": {
            "total_images": len(yolo_images),
            "total_labels": len(yolo_labels)
        },
        "yolo_compliant_audit": yolo_compliant_audit,
        "benchmarks": {
            "human_75_count": hr_count,
            "real_packaging_20_count": rp_count
        }
    }

    out_json = REPORTS_DIR / "dataset_provenance_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nAudit complete. Authoritative report saved to: {out_json}")
    return report_payload


if __name__ == "__main__":
    run_dataset_provenance_audit()
