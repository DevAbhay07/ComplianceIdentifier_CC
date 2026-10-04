"""
create_review_dataset.py
==========================
Creates the review dataset (review_36.csv) for the 36 inconsistent
annotations identified in Phase 4.

This script:
1. Loads the deduplicated annotations
2. Identifies records where overall=NON_COMPLIANT but all declarations are PRESENT/NOT_APPLICABLE
3. Enriches each record with product listing evidence
4. Creates a review CSV with blank revision fields for human review

DOES NOT modify the original annotations.
DOES NOT auto-populate revised fields.
"""

import json
import csv
import os
import pandas as pd

ANNOTATIONS_PATH = "annotations/pilot_annotations.jsonl"
PARSED_DATA_PATH = "data/interim/parsed_annotation_candidates.jsonl"
OUTPUT_PATH = "annotations/review_36.csv"

DECLS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]

DECL_NAMES = {
    "D01": "Manufacturer/Packer/Importer Name & Address",
    "D02": "Common/Generic Name of Commodity",
    "D03": "Net Quantity",
    "D04": "Month/Year of Manufacture",
    "D05": "MRP",
    "D06": "Consumer Care Details",
    "D07": "Country of Origin",
}


def main():
    # Load deduplicated annotations
    all_anns = []
    with open(ANNOTATIONS_PATH) as f:
        for line in f:
            all_anns.append(json.loads(line))

    latest = {}
    for ann in all_anns:
        pid = ann["product_id"]
        if pid not in latest or ann["timestamp"] > latest[pid]["timestamp"]:
            latest[pid] = ann

    # Identify the 36 inconsistent records
    flagged = []
    for ann in latest.values():
        labels = {d: ann["declarations"].get(d, {}).get("label", "") for d in DECLS}
        overall = ann["overall_assessment"]["label"]
        if overall == "NON_COMPLIANT" and all(
            l in ("PRESENT", "NOT_APPLICABLE") for l in labels.values()
        ):
            flagged.append(ann)

    print(f"Flagged inconsistent annotations: {len(flagged)}")

    # Load product data for evidence enrichment
    products = {}
    with open(PARSED_DATA_PATH) as f:
        for line in f:
            rec = json.loads(line)
            products[str(rec["product_id"])] = rec

    # Build the review CSV
    rows = []
    for ann in flagged:
        pid = ann["product_id"]
        product = products.get(pid, {})
        snapshot = ann.get("product_snapshot", {})

        # Extract evidence sources per declaration
        decl_data = ann.get("declarations", {})

        row = {
            # Product identification
            "product_id": pid,
            "product_name": snapshot.get("name", product.get("product_name", "")),
            "product_category": snapshot.get("category", product.get("top_category", "")),
            "brand": snapshot.get("brand", product.get("brand", "")),
            "retail_price": snapshot.get("retail_price", product.get("retail_price", "")),

            # Original annotation
            "original_overall": ann["overall_assessment"]["label"],
            "original_overall_notes": ann["overall_assessment"].get("notes", ""),
            "original_confidence": ann["overall_assessment"].get("confidence", ""),
            "annotation_timestamp": ann.get("timestamp", ""),
            "schema_version": ann.get("schema_version", ""),
        }

        # Original declaration labels and evidence
        for did in DECLS:
            d = decl_data.get(did, {})
            row[f"original_{did}"] = d.get("label", "")
            row[f"evidence_{did}"] = d.get("evidence_text", "")
            row[f"sources_{did}"] = str(d.get("evidence_sources", []))

        # Flag reason
        row["flag_reason"] = (
            "Overall is NON_COMPLIANT but all 7 declarations are PRESENT or NOT_APPLICABLE. "
            "This is internally inconsistent per the annotation guidelines."
        )

        # Machine guidance info (if draft was used)
        mg = ann.get("machine_guidance", {})
        row["draft_generated"] = mg.get("draft_generated", False)
        if mg.get("draft_generated"):
            row["draft_overall_label"] = mg.get("overall", {}).get("machine_label", "")
            row["overall_human_modified"] = mg.get("overall", {}).get("human_modified", "")
            for did in DECLS:
                dm = mg.get(did, {})
                row[f"draft_{did}"] = dm.get("machine_label", "")
                row[f"modified_{did}"] = dm.get("human_modified", "")
        else:
            row["draft_overall_label"] = ""
            row["overall_human_modified"] = ""
            for did in DECLS:
                row[f"draft_{did}"] = ""
                row[f"modified_{did}"] = ""

        # Available product evidence summary
        specs = product.get("parsed_specs", {})
        row["spec_keys"] = str(list(specs.keys())) if specs else "[]"
        row["description_length"] = len(product.get("description", ""))
        row["image_urls_available"] = bool(product.get("image_urls", ""))

        # === BLANK REVIEW FIELDS (human must fill these) ===
        row["review_status"] = ""  # REVIEWED / PENDING
        row["reviewer_id"] = ""
        row["reviewer_decision"] = ""  # CONFIRMED / REVISED / NEEDS_MORE_EVIDENCE
        row["revised_overall"] = ""  # COMPLIANT / NON_COMPLIANT / UNCERTAIN
        for did in DECLS:
            row[f"revised_{did}"] = ""  # blank until human reviews
        row["review_evidence_checked"] = ""  # which sources were checked
        row["review_reason"] = ""  # why the reviewer made this decision
        row["reviewer_notes"] = ""  # any additional notes

        rows.append(row)

    # Sort by product_id for reproducibility
    rows.sort(key=lambda r: r["product_id"])

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Review dataset saved: {OUTPUT_PATH}")
    print(f"Records: {len(rows)}")
    print(f"Columns: {len(fieldnames)}")
    print(f"\nBlank review fields for human annotation:")
    print(f"  review_status, reviewer_id, reviewer_decision")
    print(f"  revised_overall, revised_D01-D07")
    print(f"  review_evidence_checked, review_reason, reviewer_notes")


if __name__ == "__main__":
    main()
