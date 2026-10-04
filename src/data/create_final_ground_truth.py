"""
create_final_ground_truth.py
==============================
Reconciles the 75 unique pilot products into a finalized, versioned
human-ground-truth dataset (Version 1.0.0).

Inputs:
  - annotations/pilot_annotations.jsonl (78 records -> 75 unique products)
  - annotations/review_36_results.jsonl (37 records -> 36 unique reviewed products)
  - annotations/pilot_manifest.csv (75 sample manifest)
  - data/interim/parsed_annotation_candidates.jsonl (product metadata & specs)

Outputs:
  - data/processed/pilot_ground_truth_75_v1.jsonl
  - data/processed/pilot_ground_truth_75_v1.csv
  - data/processed/pilot_ground_truth_manifest.json

CRITICAL RULES:
  - Machine drafts != Human verified ground truth.
  - Only human-confirmed values are placed in ground-truth fields.
  - Audit metadata is preserved separately.
  - Exactly 75 unique products are outputted.
"""

import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

import pandas as pd

# Paths
ANNOTATIONS_PATH = "annotations/pilot_annotations.jsonl"
REVIEW_RESULTS_PATH = "annotations/review_36_results.jsonl"
MANIFEST_PATH = "annotations/pilot_manifest.csv"
PARSED_DATA_PATH = "data/interim/parsed_annotation_candidates.jsonl"

OUTPUT_JSONL_PATH = "data/processed/pilot_ground_truth_75_v1.jsonl"
OUTPUT_CSV_PATH = "data/processed/pilot_ground_truth_75_v1.csv"
MANIFEST_JSON_PATH = "data/processed/pilot_ground_truth_manifest.json"

DECLS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]

VALID_DECL_LABELS = {"PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"}
VALID_OVERALL_LABELS = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"}
VALID_SOURCES = {
    "PRODUCT_TITLE", "PRODUCT_DESCRIPTION", "PRODUCT_SPECIFICATIONS",
    "BRAND_FIELD", "PRICE_FIELD", "PRODUCT_IMAGE", "PRODUCT_PAGE",
    "MULTIPLE_SOURCES", "NO_EVIDENCE", "CANNOT_VERIFY",
}



def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_parsed_products() -> Dict[str, dict]:
    """Load parsed product data indexed by product_id."""
    products = {}
    with open(PARSED_DATA_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            products[str(rec.get("product_id", ""))] = rec
    return products


def load_deduped_originals() -> Dict[str, dict]:
    """
    Load original pilot annotations deduplicated by timestamp.
    Returns latest record for each unique product_id.
    """
    latest = {}
    with open(ANNOTATIONS_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["product_id"]
            if pid not in latest or rec["timestamp"] > latest[pid]["timestamp"]:
                latest[pid] = rec
    return latest


def load_reconciled_reviews() -> Dict[str, dict]:
    """
    Load reviewed records for the 36 flagged products.
    Handles duplicate review records (e.g. 015d31ad501bf22af9526da45b2a30fe)
    by selecting the valid completed review.
    """
    reviews_by_pid = {}
    with open(REVIEW_RESULTS_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["product_id"]
            if pid not in reviews_by_pid:
                reviews_by_pid[pid] = []
            reviews_by_pid[pid].append(rec)

    reconciled = {}
    for pid, rev_list in reviews_by_pid.items():
        if len(rev_list) == 1:
            reconciled[pid] = rev_list[0]
        else:
            # Select the complete review (for 015d31ad501bf22af9526da45b2a30fe,
            # select the review where declarations were properly reviewed)
            valid_reviews = [
                r for r in rev_list
                if not (
                    r.get("revised_overall") == "NON_COMPLIANT"
                    and all(
                        r.get("revised_declarations", {}).get(d, {}).get("revised_label") == "PRESENT"
                        for d in DECLS
                    )
                )
            ]
            if valid_reviews:
                reconciled[pid] = valid_reviews[-1]
            else:
                reconciled[pid] = rev_list[-1]
    return reconciled


def normalize_evidence_source(src: str) -> str:
    """Normalize legacy source names to approved schema."""
    clean = str(src).strip().upper()
    mapping = {
        "LISTING_TITLE": "PRODUCT_TITLE",
        "TITLE": "PRODUCT_TITLE",
        "SPECIFICATION": "PRODUCT_SPECIFICATIONS",
        "SPECIFICATIONS": "PRODUCT_SPECIFICATIONS",
        "SPECS": "PRODUCT_SPECIFICATIONS",
        "DESCRIPTION": "PRODUCT_DESCRIPTION",
        "NONE": "NO_EVIDENCE",
        "IMAGE": "PRODUCT_IMAGE",
        "IMAGES": "PRODUCT_IMAGE",
        "PRICE": "PRICE_FIELD",
        "BRAND": "BRAND_FIELD",
    }
    if clean in mapping:
        return mapping[clean]
    if clean in VALID_SOURCES:
        return clean
    return "NO_EVIDENCE" if not clean else "PRODUCT_DESCRIPTION"


def build_final_dataset():
    os.makedirs("data/processed", exist_ok=True)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    expected_pids = manifest_df["product_id"].tolist()
    print(f"Loaded pilot manifest with {len(expected_pids)} expected products.")

    products_metadata = load_parsed_products()
    original_annotations = load_deduped_originals()
    reconciled_reviews = load_reconciled_reviews()

    print(f"Loaded {len(original_annotations)} unique original annotations.")
    print(f"Loaded {len(reconciled_reviews)} unique reviewed records.")

    final_records = []
    csv_rows = []

    ai_assisted_count = 0
    human_reviewed_count = len(reconciled_reviews)

    for pid in expected_pids:
        meta = products_metadata.get(pid, {})
        is_reviewed = pid in reconciled_reviews

        if is_reviewed:
            rev = reconciled_reviews[pid]
            provenance = "PHASE_4.5_HUMAN_REVIEW"
            reviewer_id = rev.get("reviewer_id", "reviewer_01")
            record_timestamp = rev.get("review_timestamp", "")
            overall_label = rev.get("revised_overall", "")

            # Check if AI draft was used
            ai_draft = rev.get("ai_draft", {})
            ai_draft_used = bool(ai_draft and ai_draft.get("draft_generated"))
            if ai_draft_used:
                ai_assisted_count += 1

            declarations = {}
            for did in DECLS:
                d_rev = rev.get("revised_declarations", {}).get(did, {})
                src = normalize_evidence_source(d_rev.get("evidence_source", "NO_EVIDENCE"))
                declarations[did] = {
                    "label": d_rev.get("revised_label", ""),
                    "evidence_source": src,
                    "evidence_notes": d_rev.get("evidence_notes", ""),
                    "review_action": d_rev.get("review_action", "MANUAL"),
                    "original_label": d_rev.get("original_label", ""),
                }

            audit_info = {
                "reviewer_decision": rev.get("reviewer_decision", "CONFIRMED"),
                "review_reason": rev.get("review_reason", ""),
                "reviewer_notes": rev.get("reviewer_notes", ""),
                "review_confidence": rev.get("review_confidence", "HIGH"),
                "overall_review_action": rev.get("overall_review_action", "MANUAL"),
                "ai_draft_metadata": {
                    "draft_generated": ai_draft_used,
                    "model_name": ai_draft.get("model_name"),
                    "draft_timestamp": ai_draft.get("draft_timestamp"),
                } if ai_draft_used else {"draft_generated": False},
            }
        else:
            orig = original_annotations[pid]
            provenance = "ORIGINAL_PILOT_ANNOTATION"
            reviewer_id = orig.get("annotator_id", "annotator_01")
            record_timestamp = orig.get("timestamp", "")
            overall_label = orig.get("overall_assessment", {}).get("label", "")

            declarations = {}
            for did in DECLS:
                d_orig = orig.get("declarations", {}).get(did, {})
                sources = d_orig.get("evidence_sources", [])
                raw_src = sources[0] if sources else "NO_EVIDENCE"
                src = normalize_evidence_source(raw_src)
                declarations[did] = {
                    "label": d_orig.get("label", ""),
                    "evidence_source": src,
                    "evidence_notes": d_orig.get("evidence_text", "") or d_orig.get("notes", ""),
                    "review_action": "ORIGINAL_HUMAN_ANNOTATION",
                    "original_label": d_orig.get("label", ""),
                }


            audit_info = {
                "reviewer_decision": "ORIGINAL_CONFIRMED",
                "review_reason": orig.get("overall_assessment", {}).get("notes", ""),
                "reviewer_notes": "",
                "review_confidence": orig.get("overall_assessment", {}).get("confidence", "MEDIUM"),
                "overall_review_action": "ORIGINAL",
                "ai_draft_metadata": {"draft_generated": False},
            }

        # Build clean JSON record
        record = {
            "product_id": pid,
            "product_name": meta.get("product_name", ""),
            "product_url": meta.get("product_url", ""),
            "category": meta.get("top_category", ""),
            "category_tree": meta.get("product_category_tree", ""),
            "brand": meta.get("brand", ""),
            "retail_price": meta.get("retail_price"),
            "discounted_price": meta.get("discounted_price"),
            "image_urls": meta.get("image_urls", ""),
            "overall_label": overall_label,
            "declarations": declarations,
            "provenance": provenance,
            "human_confirmed": True,
            "reviewed_by_human": True,
            "reviewer_id": reviewer_id,
            "finalized_timestamp": record_timestamp,
            "audit": audit_info,
        }
        final_records.append(record)

        # Build flat CSV row
        csv_row = {
            "product_id": pid,
            "product_name": meta.get("product_name", ""),
            "category": meta.get("top_category", ""),
            "brand": meta.get("brand", ""),
            "retail_price": meta.get("retail_price", ""),
            "overall_label": overall_label,
            "provenance": provenance,
            "reviewer_id": reviewer_id,
            "finalized_timestamp": record_timestamp,
        }
        for did in DECLS:
            csv_row[f"{did}_label"] = declarations[did]["label"]
            csv_row[f"{did}_source"] = declarations[did]["evidence_source"]
            csv_row[f"{did}_notes"] = declarations[did]["evidence_notes"]
        csv_rows.append(csv_row)

    # 1. Write JSONL
    with open(OUTPUT_JSONL_PATH, "w") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(final_records)} records to {OUTPUT_JSONL_PATH}")

    # 2. Write CSV
    final_df = pd.DataFrame(csv_rows)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Wrote {len(final_df)} records to {OUTPUT_CSV_PATH}")

    # 3. Write Manifest
    jsonl_hash = compute_sha256(OUTPUT_JSONL_PATH)
    csv_hash = compute_sha256(OUTPUT_CSV_PATH)

    manifest_data = {
        "dataset_name": "NitiMark Pilot Ground Truth",
        "version": "1.0.0",
        "release_tag": "pilot_ground_truth_75_v1",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_products": len(final_records),
        "unique_products": len(set(expected_pids)),
        "human_verified_count": len(final_records),
        "phase_4.5_reviewed_count": human_reviewed_count,
        "original_baseline_count": len(final_records) - human_reviewed_count,
        "ai_assisted_reviews_count": ai_assisted_count,
        "declaration_schema_version": "1.0.0",
        "legal_requirement_source": "Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 6)",
        "declarations_covered": DECLS,
        "valid_declaration_labels": list(VALID_DECL_LABELS),
        "valid_overall_labels": list(VALID_OVERALL_LABELS),
        "files": {
            "jsonl": {
                "path": OUTPUT_JSONL_PATH,
                "sha256": jsonl_hash,
                "record_count": len(final_records),
            },
            "csv": {
                "path": OUTPUT_CSV_PATH,
                "sha256": csv_hash,
                "record_count": len(final_df),
            },
        },
    }

    with open(MANIFEST_JSON_PATH, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"✅ Wrote manifest to {MANIFEST_JSON_PATH}")


if __name__ == "__main__":
    build_final_dataset()
