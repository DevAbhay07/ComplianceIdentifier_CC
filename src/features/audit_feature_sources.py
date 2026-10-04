"""
audit_feature_sources.py
=========================
Comprehensive feature source auditor for Phase 7.
Profiles all columns and fields across raw, interim evidence, weak labels,
and human ground truth datasets to classify their ML roles and leakage risks.

Outputs:
    reports/phase_7/feature_source_audit.csv
    reports/phase_7/feature_source_audit.md
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import yaml
import pandas as pd
from typing import Dict, Any, List


def classify_field_role(field_name: str, source: str) -> tuple[str, str]:
    """Classify feature role and leakage risk."""
    fn = field_name.lower()
    
    # Target columns
    if any(fn.startswith(f"d0{i}_label") for i in range(1, 8)) or fn == "overall_label":
        return "TARGET", "HIGH_TARGET_LEAKAGE"
    if any(fn == f"d0{i}_target" for i in range(1, 8)) or fn == "overall_target":
        return "TARGET", "HIGH_TARGET_LEAKAGE"

    # Rule / Generation confidence and provenance
    if any(fn.startswith(f"d0{i}_confidence") for i in range(1, 8)) or fn == "overall_confidence":
        return "RULE_DERIVED", "PROVENANCE_LEAKAGE"
    if any(fn.startswith(f"d0{i}_evidence") for i in range(1, 8)) or fn == "overall_reason":
        return "RULE_DERIVED", "PROVENANCE_LEAKAGE"
    if fn in ("rule_ids", "rule_engine_version", "evidence_version", "extraction_version", "quality_tier", "label_type", "provenance", "generation_timestamp", "audit", "reviewed_by_human", "reviewer_id", "finalized_timestamp", "human_confirmed"):
        return "PROVENANCE", "PROVENANCE_LEAKAGE"

    # Identifiers
    if fn in ("product_id", "uniq_id", "pid", "product_url", "duplicate_group_id", "product_family_id"):
        return "IDENTIFIER", "SPLIT_GROUPING_KEY"

    # Evidence Derived Observable Inputs
    if any(k in fn for k in ["_token_count", "contains_", "has_"]):
        return "EVIDENCE_DERIVED_INPUT", "NONE"

    # Independent Observable Inputs
    if fn in ("product_name", "title", "description", "brand", "retail_price", "discounted_price", "category", "root_category", "subcategory", "category_hierarchy", "image_urls", "image_count", "price_tier", "specifications", "extra_spec_values"):
        return "INDEPENDENT_INPUT", "NONE"

    if any(k in fn for k in ["length", "count", "amount", "percentage"]):
        return "INDEPENDENT_INPUT", "NONE"

    return "METADATA", "CAUTION"


def audit_dataset_fields(filepath: str, source_name: str, max_samples: int = 500) -> List[Dict[str, Any]]:
    """Audit fields in a CSV or JSONL dataset."""
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found for auditing.")
        return []

    print(f"Auditing {source_name} ({filepath})...")
    records = []
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, low_memory=False, nrows=max_samples)
        for col in df.columns:
            missing_rate = round(float(df[col].isna().mean() * 100), 2)
            unique_count = int(df[col].nunique(dropna=True))
            sample_vals = [str(x) for x in df[col].dropna().unique()[:3]]
            role, risk = classify_field_role(col, source_name)
            records.append({
                "field_name": col,
                "source_dataset": source_name,
                "data_type": str(df[col].dtype),
                "description": f"Field '{col}' from {source_name}",
                "missing_rate_pct": missing_rate,
                "unique_count": unique_count,
                "example_values": "; ".join(sample_vals)[:80],
                "feature_role": role,
                "leakage_risk": risk
            })
    elif filepath.endswith(".jsonl"):
        raw_items = []
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    raw_items.append(json.loads(line))
                if i >= max_samples:
                    break
        df = pd.json_normalize(raw_items)
        for col in df.columns:
            missing_rate = round(float(df[col].isna().mean() * 100), 2)
            try:
                unique_count = int(df[col].astype(str).nunique(dropna=True))
                sample_vals = [str(x) for x in df[col].dropna().astype(str).unique()[:3]]
            except Exception:
                unique_count = -1
                sample_vals = []
            role, risk = classify_field_role(col, source_name)
            records.append({
                "field_name": col,
                "source_dataset": source_name,
                "data_type": str(df[col].dtype),
                "description": f"Field '{col}' from {source_name}",
                "missing_rate_pct": missing_rate,
                "unique_count": unique_count,
                "example_values": "; ".join(sample_vals)[:80],
                "feature_role": role,
                "leakage_risk": risk
            })

    return records


def run_feature_source_audit(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    RAW_CSV = "data/raw/flipkart/flipkart_com-ecommerce_sample.csv"
    EVIDENCE_JSONL = "data/interim/product_evidence.jsonl"
    WEAK_JSONL = "data/processed/flipkart_weak_labels_v1.jsonl"
    PILOT_JSONL = "data/processed/pilot_ground_truth_75_v1.jsonl"

    all_audits = []
    all_audits.extend(audit_dataset_fields(RAW_CSV, "FLIPKART_RAW_CSV"))
    all_audits.extend(audit_dataset_fields(EVIDENCE_JSONL, "PRODUCT_EVIDENCE_STREAM"))
    all_audits.extend(audit_dataset_fields(WEAK_JSONL, "WEAK_LABEL_DATASET_V1"))
    all_audits.extend(audit_dataset_fields(PILOT_JSONL, "HUMAN_GROUND_TRUTH_PILOT"))

    audit_df = pd.DataFrame(all_audits)
    csv_out = os.path.join(output_dir, "feature_source_audit.csv")
    audit_df.to_csv(csv_out, index=False)
    print(f"Saved feature source audit CSV to {csv_out}")

    md_out = os.path.join(output_dir, "feature_source_audit.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Phase 7: Feature Source Audit & Classification Report\n\n")
        f.write(f"**Total Fields Audited:** {len(all_audits)} across 4 datasets  \n\n")

        f.write("## 1. Feature Role Classification Summary\n\n")
        role_counts = audit_df["feature_role"].value_counts()
        f.write("| Feature Role | Field Count | Leakage Risk Description |\n")
        f.write("| :--- | :--- | :--- |\n")
        role_desc = {
            "INDEPENDENT_INPUT": "Clean observable attributes (text length, price, category, image counts). SAFE for ML.",
            "EVIDENCE_DERIVED_INPUT": "Token/pattern count features derived from raw inputs. SAFE for ML.",
            "RULE_DERIVED": "Rule engine scores and confidence. EXCLUDED from feature matrix X.",
            "TARGET": "Ground truth or weak label target columns (D01-D07, overall). EXCLUDED from X.",
            "PROVENANCE": "Auditing and pipeline version metadata. EXCLUDED from X.",
            "IDENTIFIER": "Unique IDs and group clustering keys. Used for splitting only.",
            "METADATA": "Auxiliary descriptors.",
            "EXCLUDED": "Explicitly prohibited features."
        }
        for role, count in role_counts.items():
            f.write(f"| **`{role}`** | {count} | {role_desc.get(role, '')} |\n")
        f.write("\n")

        f.write("## 2. Comprehensive Field Inventory\n\n")
        f.write("| Source | Field Name | Data Type | Missing % | Feature Role | Leakage Risk |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for idx, row in audit_df.iterrows():
            risk_badge = f"**`{row['leakage_risk']}`**" if "LEAKAGE" in row["leakage_risk"] else f"`{row['leakage_risk']}`"
            f.write(f"| {row['source_dataset']} | `{row['field_name']}` | {row['data_type']} | {row['missing_rate_pct']}% | `{row['feature_role']}` | {risk_badge} |\n")
        f.write("\n")

    print(f"Saved feature source audit Markdown to {md_out}")
    return audit_df


if __name__ == "__main__":
    run_feature_source_audit("reports/phase_7")
