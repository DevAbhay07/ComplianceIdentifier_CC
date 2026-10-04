"""
validate_annotations.py
=========================
Validates saved annotation JSONL files for schema compliance and data quality.

Checks:
  - Missing declaration labels
  - Invalid labels (not in allowed values)
  - Missing overall label
  - Invalid evidence sources
  - Duplicate annotations (same product annotated multiple times)
  - Missing product IDs
  - Schema violations
  - Missing product snapshots
  - Consistency checks

Usage:
    python src/data/validate_annotations.py

Does NOT modify annotations. Reports problems only.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime

# -------------------------------------------------------------------------
# Constants (must match annotation tool)
# -------------------------------------------------------------------------
ANNOTATIONS_PATH = "annotations/pilot_annotations.jsonl"

VALID_DECLARATION_LABELS = {"PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"}
VALID_OVERALL_LABELS = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"}
VALID_EVIDENCE_SOURCES = {"LISTING_TITLE", "DESCRIPTION", "SPECIFICATION", "PRICE_FIELD", "BRAND_FIELD", "IMAGE", "OTHER", "NONE"}
REQUIRED_DECLARATIONS = {"D01", "D02", "D03", "D04", "D05", "D06", "D07"}
REQUIRED_SNAPSHOT_FIELDS = {"name", "category", "brand"}


def validate_single_annotation(annotation: dict, line_num: int) -> list:
    """Validate a single annotation record. Returns list of issues."""
    issues = []
    prefix = f"Line {line_num}"

    # 1. Check required top-level fields
    for field in ["annotation_id", "product_id", "annotator_id", "timestamp", "schema_version"]:
        if field not in annotation or not annotation[field]:
            issues.append(f"{prefix}: Missing required field '{field}'")

    # 2. Check product_id
    pid = annotation.get("product_id", "")
    if not pid or not str(pid).strip():
        issues.append(f"{prefix}: Empty product_id")

    # 3. Check product snapshot
    snapshot = annotation.get("product_snapshot", {})
    if not snapshot:
        issues.append(f"{prefix} [{pid}]: Missing product_snapshot")
    else:
        for field in REQUIRED_SNAPSHOT_FIELDS:
            if field not in snapshot:
                issues.append(f"{prefix} [{pid}]: Missing snapshot field '{field}'")

    # 4. Check declarations
    declarations = annotation.get("declarations", {})
    if not declarations:
        issues.append(f"{prefix} [{pid}]: Missing declarations section")
    else:
        # Check all required declarations are present
        present_decls = set(declarations.keys())
        missing_decls = REQUIRED_DECLARATIONS - present_decls
        if missing_decls:
            issues.append(f"{prefix} [{pid}]: Missing declarations: {missing_decls}")

        extra_decls = present_decls - REQUIRED_DECLARATIONS
        if extra_decls:
            issues.append(f"{prefix} [{pid}]: Unexpected declarations: {extra_decls}")

        # Validate each declaration
        for decl_id, decl_data in declarations.items():
            if not isinstance(decl_data, dict):
                issues.append(f"{prefix} [{pid}]: Declaration {decl_id} is not a dict")
                continue

            # Check label
            label = decl_data.get("label", "")
            if not label:
                issues.append(f"{prefix} [{pid}]: Missing label for {decl_id}")
            elif label not in VALID_DECLARATION_LABELS:
                issues.append(f"{prefix} [{pid}]: Invalid label '{label}' for {decl_id}")

            # Check evidence sources
            sources = decl_data.get("evidence_sources", [])
            if isinstance(sources, list):
                for src in sources:
                    if src not in VALID_EVIDENCE_SOURCES:
                        issues.append(f"{prefix} [{pid}]: Invalid evidence source '{src}' for {decl_id}")

    # 5. Check overall assessment
    overall = annotation.get("overall_assessment", {})
    if not overall:
        issues.append(f"{prefix} [{pid}]: Missing overall_assessment")
    else:
        overall_label = overall.get("label", "")
        if not overall_label:
            issues.append(f"{prefix} [{pid}]: Missing overall label")
        elif overall_label not in VALID_OVERALL_LABELS:
            issues.append(f"{prefix} [{pid}]: Invalid overall label '{overall_label}'")

    # 6. Check timestamp format
    ts = annotation.get("timestamp", "")
    if ts:
        try:
            datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            issues.append(f"{prefix} [{pid}]: Invalid timestamp format '{ts}'")

    # 7. Check that machine_guidance is separate from human labels
    if "machine_guidance" in annotation:
        mg = annotation["machine_guidance"]
        if isinstance(mg, dict):
            for decl_id, mg_data in mg.items():
                if isinstance(mg_data, dict) and "label" in mg_data:
                    issues.append(
                        f"{prefix} [{pid}]: machine_guidance for {decl_id} contains 'label' — "
                        "machine guidance should NOT contain compliance labels"
                    )

    return issues


def validate_cross_annotation(annotations: list) -> list:
    """Validate across all annotations (duplicates, consistency)."""
    issues = []

    # Check for duplicate product annotations
    pid_counts = Counter()
    for ann in annotations:
        pid_counts[ann.get("product_id", "")] += 1

    duplicates = {pid: count for pid, count in pid_counts.items() if count > 1}
    if duplicates:
        for pid, count in duplicates.items():
            issues.append(
                f"DUPLICATE: Product {pid} has {count} annotations. "
                "This is allowed (re-annotation) but only the latest should be used for training."
            )

    return issues


def generate_statistics(annotations: list) -> dict:
    """Generate descriptive statistics from annotations."""
    stats = {
        "total_annotations": len(annotations),
        "unique_products": len(set(ann.get("product_id") for ann in annotations)),
        "annotators": list(set(ann.get("annotator_id", "unknown") for ann in annotations)),
    }

    # Overall label distribution
    overall_dist = Counter()
    for ann in annotations:
        label = ann.get("overall_assessment", {}).get("label", "MISSING")
        overall_dist[label] += 1
    stats["overall_label_distribution"] = dict(overall_dist)

    # Per-declaration label distribution
    decl_dist = {}
    for decl_id in REQUIRED_DECLARATIONS:
        decl_dist[decl_id] = Counter()

    for ann in annotations:
        for decl_id, decl_data in ann.get("declarations", {}).items():
            if decl_id in decl_dist:
                decl_dist[decl_id][decl_data.get("label", "MISSING")] += 1

    stats["declaration_distributions"] = {
        decl_id: dict(dist) for decl_id, dist in decl_dist.items()
    }

    # Confidence distribution
    conf_dist = Counter()
    for ann in annotations:
        conf = ann.get("overall_assessment", {}).get("confidence", "MISSING")
        conf_dist[conf] += 1
    stats["confidence_distribution"] = dict(conf_dist)

    return stats


def main():
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"No annotations file found at: {ANNOTATIONS_PATH}")
        print("This is expected if no annotations have been made yet.")
        print("Start the annotation tool first:")
        print("  streamlit run src/annotation_tool.py")
        return

    # Load annotations
    annotations = []
    parse_errors = []

    with open(ANNOTATIONS_PATH, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                ann = json.loads(line)
                annotations.append(ann)
            except json.JSONDecodeError as e:
                parse_errors.append(f"Line {i}: JSON parse error: {e}")

    print(f"=== ANNOTATION VALIDATION REPORT ===")
    print(f"  File: {ANNOTATIONS_PATH}")
    print(f"  Total records: {len(annotations)}")
    print(f"  Parse errors: {len(parse_errors)}")
    print()

    if parse_errors:
        print("=== JSON PARSE ERRORS ===")
        for err in parse_errors:
            print(f"  ❌ {err}")
        print()

    # Validate each annotation
    all_issues = []
    for i, ann in enumerate(annotations, 1):
        issues = validate_single_annotation(ann, i)
        all_issues.extend(issues)

    # Cross-annotation validation
    cross_issues = validate_cross_annotation(annotations)
    all_issues.extend(cross_issues)

    # Report issues
    if all_issues:
        print(f"=== VALIDATION ISSUES ({len(all_issues)} found) ===")
        for issue in all_issues:
            severity = "⚠️" if "DUPLICATE" in issue else "❌"
            print(f"  {severity} {issue}")
    else:
        print("=== VALIDATION: ✅ ALL CHECKS PASSED ===")
    print()

    # Statistics (only if annotations exist)
    if annotations:
        stats = generate_statistics(annotations)
        print(f"=== ANNOTATION STATISTICS ===")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Unique products: {stats['unique_products']}")
        print(f"  Annotators: {stats['annotators']}")
        print()
        print(f"  Overall label distribution:")
        for label, count in sorted(stats['overall_label_distribution'].items()):
            print(f"    {label}: {count}")
        print()
        print(f"  Per-declaration label distribution:")
        for decl_id in sorted(stats['declaration_distributions'].keys()):
            dist = stats['declaration_distributions'][decl_id]
            print(f"    {decl_id}:")
            for label, count in sorted(dist.items()):
                print(f"      {label}: {count}")
        print()
        print(f"  Confidence distribution:")
        for conf, count in sorted(stats['confidence_distribution'].items()):
            print(f"    {conf}: {count}")
    else:
        print("No annotations to analyze yet.")

    # Summary
    print()
    error_count = len([i for i in all_issues if "DUPLICATE" not in i])
    warn_count = len([i for i in all_issues if "DUPLICATE" in i])
    print(f"=== SUMMARY ===")
    print(f"  Errors: {error_count}")
    print(f"  Warnings: {warn_count}")
    print(f"  Parse errors: {len(parse_errors)}")

    if error_count == 0 and len(parse_errors) == 0:
        print(f"  Status: ✅ VALID")
    else:
        print(f"  Status: ❌ HAS ISSUES")


if __name__ == "__main__":
    main()
