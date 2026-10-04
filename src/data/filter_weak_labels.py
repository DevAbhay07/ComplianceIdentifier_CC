"""
filter_weak_labels.py
======================
Applies quality filtering, deduplication / family clustering, and serializes
the canonical Version 1.0.0 weak-label dataset in both JSONL and CSV formats.

Inputs:
    data/interim/raw_weak_labels.jsonl

Outputs:
    data/processed/flipkart_weak_labels_v1.jsonl
    data/processed/flipkart_weak_labels_v1.csv
    reports/phase_6/weak_label_manifest.json
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd
from typing import List, Dict, Any

from src.data.deduplicate_weak_dataset import deduplicate_and_assign_families


def flatten_weak_record_for_csv(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten structured weak label record into tabular CSV row."""
    decls = rec.get("declarations", {})
    row = {
        "product_id": rec.get("product_id"),
        "product_url": rec.get("product_url"),
        "category": rec.get("category"),
        "product_name": rec.get("product_name"),
        "brand": rec.get("brand"),
        "retail_price": rec.get("retail_price"),
        "discounted_price": rec.get("discounted_price"),
        "duplicate_group_id": rec.get("duplicate_group_id"),
        "product_family_id": rec.get("product_family_id"),
        
        # Declarations
        "D01_label": decls.get("D01", {}).get("label"),
        "D01_confidence": decls.get("D01", {}).get("confidence"),
        "D01_evidence": " | ".join(decls.get("D01", {}).get("evidence_text", [])),

        "D02_label": decls.get("D02", {}).get("label"),
        "D02_confidence": decls.get("D02", {}).get("confidence"),
        "D02_evidence": " | ".join(decls.get("D02", {}).get("evidence_text", [])),

        "D03_label": decls.get("D03", {}).get("label"),
        "D03_confidence": decls.get("D03", {}).get("confidence"),
        "D03_evidence": " | ".join(decls.get("D03", {}).get("evidence_text", [])),

        "D04_label": decls.get("D04", {}).get("label"),
        "D04_confidence": decls.get("D04", {}).get("confidence"),
        "D04_evidence": " | ".join(decls.get("D04", {}).get("evidence_text", [])),

        "D05_label": decls.get("D05", {}).get("label"),
        "D05_confidence": decls.get("D05", {}).get("confidence"),
        "D05_evidence": " | ".join(decls.get("D05", {}).get("evidence_text", [])),

        "D06_label": decls.get("D06", {}).get("label"),
        "D06_confidence": decls.get("D06", {}).get("confidence"),
        "D06_evidence": " | ".join(decls.get("D06", {}).get("evidence_text", [])),

        "D07_label": decls.get("D07", {}).get("label"),
        "D07_confidence": decls.get("D07", {}).get("confidence"),
        "D07_evidence": " | ".join(decls.get("D07", {}).get("evidence_text", [])),

        "overall_label": rec.get("overall_label"),
        "overall_confidence": rec.get("overall_confidence"),
        "label_type": rec.get("label_type", "WEAK_LABEL"),
        "rule_engine_version": rec.get("rule_engine_version", "v1"),
        "evidence_version": rec.get("evidence_version", "v1"),
        "extraction_version": rec.get("extraction_version", "v1"),
        "quality_tier": rec.get("quality_tier"),
    }
    return row


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_filter_and_export(
    raw_weak_jsonl_path: str,
    output_jsonl_path: str,
    output_csv_path: str,
    manifest_path: str
):
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    print(f"Loading raw weak labels from {raw_weak_jsonl_path}...")
    raw_records = []
    with open(raw_weak_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    print(f"Loaded {len(raw_records)} raw records.")

    # Deduplication and clustering
    enriched_records = deduplicate_and_assign_families(raw_records)

    # Filter out REJECTED tier
    accepted_records = [r for r in enriched_records if r.get("quality_tier") != "REJECTED"]
    rejected_count = len(enriched_records) - len(accepted_records)

    print(f"Accepted {len(accepted_records)} records ({rejected_count} rejected).")

    # Export JSONL
    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for rec in accepted_records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved processed JSONL dataset to {output_jsonl_path}")

    # Export CSV
    flat_rows = [flatten_weak_record_for_csv(rec) for rec in accepted_records]
    df = pd.DataFrame(flat_rows)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved processed CSV dataset to {output_csv_path}")

    # Compute Checksums & Write Manifest
    jsonl_sha = compute_sha256(output_jsonl_path)
    csv_sha = compute_sha256(output_csv_path)

    manifest = {
        "dataset_name": "Flipkart Weak-Label Dataset (Phase 6)",
        "version": "1.0.0",
        "release_tag": "flipkart_weak_labels_v1",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(accepted_records),
        "rejected_records": rejected_count,
        "declarations_covered": ["D01", "D02", "D03", "D04", "D05", "D06", "D07"],
        "label_type": "WEAK_LABEL",
        "provenance": "AUTOMATED_WEAK_LABEL_PIPELINE_V1",
        "files": {
            "jsonl": {
                "path": output_jsonl_path,
                "sha256": jsonl_sha,
                "record_count": len(accepted_records)
            },
            "csv": {
                "path": output_csv_path,
                "sha256": csv_sha,
                "record_count": len(accepted_records)
            }
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved weak dataset manifest to {manifest_path}")

    return accepted_records, manifest


if __name__ == "__main__":
    RAW_JSONL = "data/interim/raw_weak_labels.jsonl"
    OUT_JSONL = "data/processed/flipkart_weak_labels_v1.jsonl"
    OUT_CSV = "data/processed/flipkart_weak_labels_v1.csv"
    MANIFEST_JSON = "reports/phase_6/weak_label_manifest.json"

    run_filter_and_export(RAW_JSONL, OUT_JSONL, OUT_CSV, MANIFEST_JSON)
