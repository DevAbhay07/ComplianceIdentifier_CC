"""
phase4_analysis.py
====================
Phase 4 — Pilot Annotation Analysis & Dataset Strategy Decision

Generates all Phase 4 deliverables from the pilot annotation data.

Usage:
    python src/data/phase4_analysis.py
"""

import json
import os
import csv
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
ANNOTATIONS_PATH = "annotations/pilot_annotations.jsonl"
MANIFEST_PATH = "annotations/pilot_manifest.csv"
PARSED_DATA_PATH = "data/interim/parsed_annotation_candidates.jsonl"

REPORTS_DIR = "reports"

DECLARATION_IDS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]
DECLARATION_NAMES = {
    "D01": "Manufacturer/Packer/Importer Name & Address",
    "D02": "Common/Generic Name of Commodity",
    "D03": "Net Quantity",
    "D04": "Month/Year of Manufacture",
    "D05": "MRP",
    "D06": "Consumer Care Details",
    "D07": "Country of Origin",
}
VALID_LABELS = {"PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"}
VALID_OVERALL = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"}


def load_deduped_annotations():
    """Load annotations, keeping only the latest per product."""
    all_anns = []
    with open(ANNOTATIONS_PATH) as f:
        for line in f:
            all_anns.append(json.loads(line))

    latest = {}
    for ann in all_anns:
        pid = ann["product_id"]
        if pid not in latest or ann["timestamp"] > latest[pid]["timestamp"]:
            latest[pid] = ann

    return list(latest.values()), all_anns


def load_parsed_products():
    """Load parsed product data keyed by product_id."""
    products = {}
    with open(PARSED_DATA_PATH) as f:
        for line in f:
            rec = json.loads(line)
            products[str(rec["product_id"])] = rec
    return products


# =====================================================================
# STEP 2: Quality Audit
# =====================================================================
def quality_audit(annotations):
    """Check for annotation quality issues that require human review."""
    issues = []

    for ann in annotations:
        pid = ann["product_id"]
        decls = ann.get("declarations", {})
        overall = ann.get("overall_assessment", {}).get("label", "")

        labels = {did: decls.get(did, {}).get("label", "") for did in DECLARATION_IDS}

        # Check: COMPLIANT but some declarations are ABSENT
        absent_decls = [d for d, l in labels.items() if l == "ABSENT"]
        if overall == "COMPLIANT" and absent_decls:
            issues.append({
                "product_id": pid,
                "issue_type": "COMPLIANT_WITH_ABSENT",
                "declaration": ", ".join(absent_decls),
                "annotated_value": ", ".join(f"{d}={labels[d]}" for d in absent_decls),
                "overall_assessment": overall,
                "reason_for_review": f"Overall is COMPLIANT but {', '.join(absent_decls)} marked ABSENT",
            })

        # Check: NON_COMPLIANT but all declarations are PRESENT
        if overall == "NON_COMPLIANT" and all(l in ("PRESENT", "NOT_APPLICABLE") for l in labels.values() if l):
            issues.append({
                "product_id": pid,
                "issue_type": "NON_COMPLIANT_ALL_PRESENT",
                "declaration": "ALL",
                "annotated_value": ", ".join(f"{d}={labels[d]}" for d in DECLARATION_IDS),
                "overall_assessment": overall,
                "reason_for_review": "Overall is NON_COMPLIANT but all declarations are PRESENT or NOT_APPLICABLE",
            })

        # Check: Missing declaration labels
        for did in DECLARATION_IDS:
            label = labels.get(did, "")
            if not label or label not in VALID_LABELS:
                issues.append({
                    "product_id": pid,
                    "issue_type": "INVALID_LABEL",
                    "declaration": did,
                    "annotated_value": label,
                    "overall_assessment": overall,
                    "reason_for_review": f"Invalid or missing label for {did}",
                })

        # Check: Missing evidence sources for PRESENT declarations
        for did in DECLARATION_IDS:
            label = labels.get(did, "")
            sources = decls.get(did, {}).get("evidence_sources", [])
            if label == "PRESENT" and (not sources or sources == ["NONE"]):
                issues.append({
                    "product_id": pid,
                    "issue_type": "PRESENT_NO_EVIDENCE",
                    "declaration": did,
                    "annotated_value": f"{did}=PRESENT, sources={sources}",
                    "overall_assessment": overall,
                    "reason_for_review": f"{did} marked PRESENT but no evidence source provided",
                })

    return issues


# =====================================================================
# STEP 3: Overall Label Distribution
# =====================================================================
def overall_distribution(annotations):
    dist = Counter(a["overall_assessment"]["label"] for a in annotations)
    total = len(annotations)
    rows = []
    for label in ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"]:
        count = dist.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        rows.append({"label": label, "count": count, "percentage": round(pct, 1)})
    return rows, dist


# =====================================================================
# STEP 4: Declaration-Level Statistics
# =====================================================================
def declaration_statistics(annotations):
    stats = defaultdict(Counter)
    for ann in annotations:
        for did in DECLARATION_IDS:
            label = ann.get("declarations", {}).get(did, {}).get("label", "MISSING")
            stats[did][label] += 1

    rows = []
    for did in DECLARATION_IDS:
        row = {"declaration": did, "name": DECLARATION_NAMES[did]}
        for label in ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"]:
            row[label] = stats[did].get(label, 0)
        rows.append(row)
    return rows, stats


# =====================================================================
# STEP 5: Evidence Source Analysis
# =====================================================================
def evidence_source_analysis(annotations):
    source_counts = defaultdict(Counter)  # did -> source -> count
    source_by_label = defaultdict(lambda: defaultdict(Counter))  # did -> label -> source -> count

    for ann in annotations:
        for did in DECLARATION_IDS:
            decl = ann.get("declarations", {}).get(did, {})
            label = decl.get("label", "")
            sources = decl.get("evidence_sources", [])
            for src in sources:
                source_counts[did][src] += 1
                source_by_label[did][label][src] += 1

    rows = []
    for did in DECLARATION_IDS:
        for src, count in sorted(source_counts[did].items(), key=lambda x: -x[1]):
            rows.append({"declaration": did, "source": src, "count": count})

    return rows, source_counts


# =====================================================================
# STEP 6: Modification Tracking (Draft vs Human)
# =====================================================================
def draft_modification_analysis(annotations):
    """Analyze how often humans modified the machine draft."""
    draft_anns = [a for a in annotations if a.get("machine_guidance", {}).get("draft_generated", False)]

    if not draft_anns:
        return None

    per_decl = defaultdict(lambda: {"total": 0, "modified": 0, "accepted": 0})
    overall_mods = {"total": 0, "modified": 0, "accepted": 0}

    for ann in draft_anns:
        mg = ann.get("machine_guidance", {})
        for did in DECLARATION_IDS:
            d = mg.get(did, {})
            if "human_modified" in d:
                per_decl[did]["total"] += 1
                if d["human_modified"]:
                    per_decl[did]["modified"] += 1
                else:
                    per_decl[did]["accepted"] += 1

        ov = mg.get("overall", {})
        if "human_modified" in ov:
            overall_mods["total"] += 1
            if ov["human_modified"]:
                overall_mods["modified"] += 1
            else:
                overall_mods["accepted"] += 1

    return {
        "total_draft_annotations": len(draft_anns),
        "per_declaration": dict(per_decl),
        "overall": overall_mods,
    }


# =====================================================================
# STEP 9: Edge Case Identification
# =====================================================================
def identify_edge_cases(annotations, products):
    """Identify annotation edge cases for the report."""
    edge_cases = []

    for ann in annotations:
        pid = ann["product_id"]
        product = products.get(pid, {})
        decls = ann.get("declarations", {})
        overall = ann.get("overall_assessment", {})
        snapshot = ann.get("product_snapshot", {})

        labels = {did: decls.get(did, {}).get("label", "") for did in DECLARATION_IDS}

        # Edge case: D01 PARTIAL (brand but no address)
        if labels.get("D01") == "PARTIAL":
            edge_cases.append({
                "product_id": pid,
                "category": snapshot.get("category", ""),
                "issue": "Manufacturer info partial — brand only, no address",
                "evidence_sources": str(decls.get("D01", {}).get("evidence_sources", [])),
                "annotation_decision": labels["D01"],
                "overall": overall.get("label", ""),
                "recommendation": "Image OCR may reveal full manufacturer/packer label",
            })

        # Edge case: D03 ABSENT (no quantity info)
        if labels.get("D03") == "ABSENT":
            edge_cases.append({
                "product_id": pid,
                "category": snapshot.get("category", ""),
                "issue": "No net quantity information available",
                "evidence_sources": str(decls.get("D03", {}).get("evidence_sources", [])),
                "annotation_decision": labels["D03"],
                "overall": overall.get("label", ""),
                "recommendation": "Check product image for quantity label; may be missing from listing metadata",
            })

        # Edge case: D04 CANNOT_DETERMINE (expected but important)
        if labels.get("D04") == "CANNOT_DETERMINE":
            edge_cases.append({
                "product_id": pid,
                "category": snapshot.get("category", ""),
                "issue": "Manufacturing date not available in listing",
                "evidence_sources": str(decls.get("D04", {}).get("evidence_sources", [])),
                "annotation_decision": labels["D04"],
                "overall": overall.get("label", ""),
                "recommendation": "Image OCR required to verify; never available in listing metadata",
            })

        # Edge case: D06 ABSENT (consumer care missing)
        if labels.get("D06") == "ABSENT":
            edge_cases.append({
                "product_id": pid,
                "category": snapshot.get("category", ""),
                "issue": "No consumer care/complaint contact details found",
                "evidence_sources": str(decls.get("D06", {}).get("evidence_sources", [])),
                "annotation_decision": labels["D06"],
                "overall": overall.get("label", ""),
                "recommendation": "Consumer care details typically only on packaging; image OCR needed",
            })

        # Edge case: D07 CANNOT_DETERMINE (origin unknown)
        if labels.get("D07") == "CANNOT_DETERMINE":
            edge_cases.append({
                "product_id": pid,
                "category": snapshot.get("category", ""),
                "issue": "Cannot determine if product is imported or domestic",
                "evidence_sources": str(decls.get("D07", {}).get("evidence_sources", [])),
                "annotation_decision": labels["D07"],
                "overall": overall.get("label", ""),
                "recommendation": "Image of packaging label may reveal country of origin",
            })

        # Edge case: COMPLIANT overall
        if overall.get("label") == "COMPLIANT":
            edge_cases.append({
                "product_id": pid,
                "category": snapshot.get("category", ""),
                "issue": "Product annotated as COMPLIANT (rare in this dataset)",
                "evidence_sources": "multiple",
                "annotation_decision": "COMPLIANT",
                "overall": "COMPLIANT",
                "recommendation": "Verify all 7 declarations are genuinely covered",
            })

    return edge_cases


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print("PHASE 4 — PILOT ANNOTATION ANALYSIS")
    print("=" * 60)
    print()

    # Load data
    annotations, all_raw = load_deduped_annotations()
    products = load_parsed_products()

    print(f"Loaded {len(annotations)} unique annotations (from {len(all_raw)} total records)")
    print()

    # --- STEP 2: Quality Audit ---
    print("--- STEP 2: QUALITY AUDIT ---")
    issues = quality_audit(annotations)
    if issues:
        print(f"  Review issues found: {len(issues)}")
        issue_types = Counter(i["issue_type"] for i in issues)
        for it, count in issue_types.items():
            print(f"    {it}: {count}")

        # Save review_required.csv
        df_issues = pd.DataFrame(issues)
        df_issues.to_csv("annotations/review_required.csv", index=False)
        print(f"  Saved: annotations/review_required.csv")
    else:
        print("  No quality issues found. ✅")
    print()

    # --- STEP 3: Overall Distribution ---
    print("--- STEP 3: OVERALL LABEL DISTRIBUTION ---")
    overall_rows, overall_dist = overall_distribution(annotations)
    for r in overall_rows:
        print(f"  {r['label']}: {r['count']} ({r['percentage']}%)")

    pd.DataFrame(overall_rows).to_csv(f"{REPORTS_DIR}/pilot_label_distribution.csv", index=False)
    print(f"  Saved: {REPORTS_DIR}/pilot_label_distribution.csv")
    print()

    # --- STEP 4: Declaration Statistics ---
    print("--- STEP 4: DECLARATION-LEVEL STATISTICS ---")
    decl_rows, decl_stats = declaration_statistics(annotations)
    df_decl = pd.DataFrame(decl_rows)
    print(df_decl.to_string(index=False))
    df_decl.to_csv(f"{REPORTS_DIR}/declaration_statistics.csv", index=False)
    print(f"\n  Saved: {REPORTS_DIR}/declaration_statistics.csv")
    print()

    # --- STEP 5: Evidence Source Analysis ---
    print("--- STEP 5: EVIDENCE SOURCE ANALYSIS ---")
    source_rows, source_counts = evidence_source_analysis(annotations)
    df_sources = pd.DataFrame(source_rows)
    print(df_sources.to_string(index=False))
    df_sources.to_csv(f"{REPORTS_DIR}/evidence_source_analysis.csv", index=False)
    print(f"\n  Saved: {REPORTS_DIR}/evidence_source_analysis.csv")
    print()

    # --- Draft Modification Analysis ---
    print("--- DRAFT MODIFICATION ANALYSIS ---")
    draft_analysis = draft_modification_analysis(annotations)
    if draft_analysis:
        print(f"  Draft-assisted annotations: {draft_analysis['total_draft_annotations']}")
        for did in DECLARATION_IDS:
            d = draft_analysis["per_declaration"].get(did, {})
            total = d.get("total", 0)
            modified = d.get("modified", 0)
            accepted = d.get("accepted", 0)
            mod_pct = modified / total * 100 if total > 0 else 0
            print(f"  {did}: {accepted} accepted, {modified} modified ({mod_pct:.0f}% modification rate)")

        ov = draft_analysis["overall"]
        ov_mod_pct = ov["modified"] / ov["total"] * 100 if ov["total"] > 0 else 0
        print(f"  OVERALL: {ov['accepted']} accepted, {ov['modified']} modified ({ov_mod_pct:.0f}% modification rate)")
    else:
        print("  No draft-assisted annotations found.")
    print()

    # --- Category Distribution ---
    print("--- CATEGORY DISTRIBUTION ---")
    cat_dist = Counter(a.get("product_snapshot", {}).get("category", "unknown") for a in annotations)
    for cat, count in cat_dist.most_common():
        print(f"  {cat}: {count}")
    print()

    # --- STEP 9: Edge Cases ---
    print("--- STEP 9: EDGE CASE ANALYSIS ---")
    edge_cases = identify_edge_cases(annotations, products)
    edge_type_counts = Counter(e["issue"] for e in edge_cases)
    for issue, count in edge_type_counts.most_common():
        print(f"  {issue}: {count}")

    # Save edge cases (deduplicate by keeping unique product_id + issue combos)
    seen = set()
    unique_edges = []
    for e in edge_cases:
        key = (e["product_id"], e["issue"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    pd.DataFrame(unique_edges).to_csv(f"{REPORTS_DIR}/pilot_edge_cases_data.csv", index=False)
    print(f"\n  Total unique edge case entries: {len(unique_edges)}")
    print(f"  Saved: {REPORTS_DIR}/pilot_edge_cases_data.csv")
    print()

    # --- Confidence Distribution ---
    print("--- CONFIDENCE DISTRIBUTION ---")
    conf_dist = Counter(a.get("overall_assessment", {}).get("confidence", "MISSING") for a in annotations)
    for conf, count in conf_dist.most_common():
        print(f"  {conf}: {count}")
    print()

    print("=" * 60)
    print("ANALYSIS COMPLETE — See reports/ directory for deliverables")
    print("=" * 60)

    # Return data for report generation
    return {
        "annotations": annotations,
        "issues": issues,
        "overall_rows": overall_rows,
        "overall_dist": overall_dist,
        "decl_rows": decl_rows,
        "decl_stats": decl_stats,
        "source_rows": source_rows,
        "draft_analysis": draft_analysis,
        "edge_cases": unique_edges,
        "cat_dist": cat_dist,
        "conf_dist": conf_dist,
    }


if __name__ == "__main__":
    main()
