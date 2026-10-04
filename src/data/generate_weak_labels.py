"""
generate_weak_labels.py
========================
Generates auditable, schema-compliant weak labels (silver labels) for Flipkart products.
Applies deterministic legal metrology rule heuristics to multi-source evidence streams.

Inputs:
    data/interim/product_evidence.jsonl
    configs/weak_label_rules.yaml

Outputs:
    data/interim/raw_weak_labels.jsonl
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import yaml
from datetime import datetime, timezone
from typing import Dict, Any, List

from src.data.detect_declaration_evidence import detect_all_declaration_evidence


def derive_overall_compliance(declarations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Derive deterministic overall compliance status and aggregate confidence.
    Follows Legal Metrology Rule 6 deterministic evaluation policy:
    - If ANY declaration is ABSENT -> NON_COMPLIANT
    - If ALL declarations are PRESENT (or NOT_APPLICABLE) -> COMPLIANT
    - Otherwise (contains PARTIAL or CANNOT_DETERMINE) -> UNCERTAIN
    """
    labels = {did: d["label"] for did, d in declarations.items()}
    confidences = [d.get("confidence", 0.0) for d in declarations.values()]

    absent_decls = [did for did, l in labels.items() if l == "ABSENT"]
    partial_decls = [did for did, l in labels.items() if l == "PARTIAL"]
    cannot_decls = [did for did, l in labels.items() if l == "CANNOT_DETERMINE"]

    if absent_decls:
        overall_label = "NON_COMPLIANT"
        reason = f"Required declaration(s) {', '.join(absent_decls)} are ABSENT in listing evidence."
    elif not partial_decls and not cannot_decls:
        overall_label = "COMPLIANT"
        reason = "All mandatory declarations (D01-D07) are evidenced as PRESENT."
    else:
        overall_label = "UNCERTAIN"
        uncertain_parts = []
        if partial_decls:
            uncertain_parts.append(f"PARTIAL ({', '.join(partial_decls)})")
        if cannot_decls:
            uncertain_parts.append(f"CANNOT_DETERMINE ({', '.join(cannot_decls)})")
        reason = f"Compliance cannot be confirmed: {'; '.join(uncertain_parts)}."

    # Compute overall confidence (mean of declaration interpretation confidences)
    overall_conf = round(float(sum(confidences) / len(confidences)), 3) if confidences else 0.0

    return {
        "overall_label": overall_label,
        "overall_confidence": overall_conf,
        "overall_reason": reason,
        "cannot_determine_count": len(cannot_decls),
        "absent_count": len(absent_decls),
        "present_count": sum(1 for l in labels.values() if l == "PRESENT"),
        "partial_count": len(partial_decls)
    }


def assign_quality_tier(
    overall_info: Dict[str, Any],
    declarations: Dict[str, Dict[str, Any]],
    rules_cfg: Dict[str, Any]
) -> str:
    """Assign quality tier based on overall confidence and uncertainty bounds."""
    tiers_cfg = rules_cfg.get("quality_tiers", {})
    conf = overall_info["overall_confidence"]
    cannot_count = overall_info["cannot_determine_count"]

    has_conflict = any(d.get("evidence_nature") == "CONFLICTING_EVIDENCE" for d in declarations.values())

    tier_a = tiers_cfg.get("TIER_A_HIGH_CONFIDENCE", {})
    if (
        conf >= tier_a.get("min_overall_confidence", 0.70)
        and cannot_count <= tier_a.get("max_cannot_determine", 2)
        and not has_conflict
    ):
        return "TIER_A_HIGH_CONFIDENCE"

    tier_b = tiers_cfg.get("TIER_B_MEDIUM_CONFIDENCE", {})
    if (
        conf >= tier_b.get("min_overall_confidence", 0.45)
        and cannot_count <= tier_b.get("max_cannot_determine", 4)
        and not has_conflict
    ):
        return "TIER_B_MEDIUM_CONFIDENCE"

    tier_c = tiers_cfg.get("TIER_C_LOW_CONFIDENCE", {})
    if conf >= tier_c.get("min_overall_confidence", 0.20):
        return "TIER_C_LOW_CONFIDENCE"

    return "REJECTED"


def generate_weak_labels_for_product(
    evidence_rec: Dict[str, Any],
    rules_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate complete weak label record with provenance for a single product."""
    product_id = evidence_rec["product_id"]
    declarations = detect_all_declaration_evidence(evidence_rec, rules_cfg)
    overall_info = derive_overall_compliance(declarations)
    quality_tier = assign_quality_tier(overall_info, declarations, rules_cfg)

    # Format declaration blocks
    formatted_decls = {}
    for did, d in declarations.items():
        formatted_decls[did] = {
            "label": d["label"],
            "confidence": d["confidence"],
            "evidence_nature": d["evidence_nature"],
            "evidence_sources": d["evidence_sources"],
            "evidence_text": d["evidence_text"],
            "rule_ids": d["rule_ids"],
            "label_type": "WEAK_LABEL",
            "extraction_method": "DETERMINISTIC_LEGAL_HEURISTICS_V1"
        }

    return {
        "product_id": product_id,
        "product_url": evidence_rec.get("product_url", ""),
        "pid": evidence_rec.get("pid", ""),
        "product_name": evidence_rec.get("title", ""),
        "category": evidence_rec.get("root_category", "Unknown"),
        "category_hierarchy": evidence_rec.get("category_hierarchy", []),
        "brand": evidence_rec.get("brand", ""),
        "retail_price": evidence_rec.get("retail_price"),
        "discounted_price": evidence_rec.get("discounted_price"),
        "image_urls": evidence_rec.get("image_urls", []),
        "image_evidence_status": evidence_rec.get("image_evidence_status", "NOT_ATTEMPTED"),
        "declarations": formatted_decls,
        "overall_label": overall_info["overall_label"],
        "overall_confidence": overall_info["overall_confidence"],
        "overall_reason": overall_info["overall_reason"],
        "quality_tier": quality_tier,
        "label_type": "WEAK_LABEL",
        "provenance": "AUTOMATED_WEAK_LABEL_PIPELINE_V1",
        "rule_engine_version": rules_cfg.get("rule_engine_version", "v1"),
        "evidence_version": rules_cfg.get("evidence_version", "v1"),
        "extraction_version": rules_cfg.get("extraction_version", "v1"),
        "generation_timestamp": datetime.now(timezone.utc).isoformat()
    }


def run_weak_label_generation(
    evidence_jsonl_path: str,
    rules_config_path: str,
    output_jsonl_path: str
):
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(rules_config_path, "r", encoding="utf-8") as f:
        rules_cfg = yaml.safe_load(f)

    print(f"Reading evidence stream from {evidence_jsonl_path}...")
    weak_records = []
    tier_counts = {"TIER_A_HIGH_CONFIDENCE": 0, "TIER_B_MEDIUM_CONFIDENCE": 0, "TIER_C_LOW_CONFIDENCE": 0, "REJECTED": 0}
    overall_counts = {"COMPLIANT": 0, "NON_COMPLIANT": 0, "UNCERTAIN": 0}

    with open(evidence_jsonl_path, "r", encoding="utf-8") as in_f, open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            evidence_rec = json.loads(line)
            weak_rec = generate_weak_labels_for_product(evidence_rec, rules_cfg)
            weak_records.append(weak_rec)
            out_f.write(json.dumps(weak_rec, ensure_ascii=False) + "\n")

            tier_counts[weak_rec["quality_tier"]] = tier_counts.get(weak_rec["quality_tier"], 0) + 1
            overall_counts[weak_rec["overall_label"]] = overall_counts.get(weak_rec["overall_label"], 0) + 1

    total = len(weak_records)
    print(f"Generated {total} weak label records.")
    print("Quality Tier Distribution:")
    for tier, count in tier_counts.items():
        pct = round(count / total * 100, 1) if total > 0 else 0
        print(f"  {tier}: {count} ({pct}%)")
    print("Overall Compliance Distribution:")
    for lab, count in overall_counts.items():
        pct = round(count / total * 100, 1) if total > 0 else 0
        print(f"  {lab}: {count} ({pct}%)")

    print(f"Saved raw weak labels to {output_jsonl_path}")
    return weak_records


if __name__ == "__main__":
    EVIDENCE_JSONL = "data/interim/product_evidence.jsonl"
    RULES_CONFIG = "configs/weak_label_rules.yaml"
    OUTPUT_JSONL = "data/interim/raw_weak_labels.jsonl"

    run_weak_label_generation(EVIDENCE_JSONL, RULES_CONFIG, OUTPUT_JSONL)
