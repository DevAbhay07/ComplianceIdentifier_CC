"""
audit_flipkart_dataset.py
==========================
Comprehensive data audit and profiling tool for the Flipkart 20K dataset.
Assesses data quality, field completeness, specification richness, category
distributions, image availability, and Legal Metrology relevance.

Inputs:
    data/raw/flipkart/flipkart_com-ecommerce_sample.csv
    configs/weak_label_rules.yaml

Outputs:
    reports/phase_6/flipkart_dataset_audit.json
    reports/phase_6/flipkart_dataset_audit.md
"""

import os
import sys
import json
import re
import yaml
import pandas as pd
import numpy as np
from collections import Counter


def parse_root_category(cat_tree_str: str) -> str:
    """Extract primary root category from product_category_tree string."""
    if pd.isna(cat_tree_str) or not str(cat_tree_str).strip():
        return "Unknown"
    s = str(cat_tree_str).strip()
    # Matches '["Category >> Subcategory ..."]'
    match = re.search(r'\["([^">]+)', s)
    if match:
        root = match.group(1).strip()
        return root
    return "Unknown"


def parse_category_hierarchy(cat_tree_str: str) -> list:
    """Extract full list of category levels."""
    if pd.isna(cat_tree_str) or not str(cat_tree_str).strip():
        return []
    s = str(cat_tree_str).strip()
    # Strip brackets and outer quotes
    cleaned = re.sub(r'^\["|"]$', '', s)
    # Split on >>
    parts = [p.strip().replace('"', '') for p in cleaned.split(">>") if p.strip()]
    return parts


def count_image_urls(image_val) -> int:
    """Count valid image URLs in image string/list."""
    if pd.isna(image_val) or not str(image_val).strip():
        return 0
    s = str(image_val).strip()
    urls = re.findall(r'https?://[^\s",\]]+', s)
    return len(urls)


def check_spec_richness(spec_val) -> dict:
    """Estimate spec count and parseability."""
    if pd.isna(spec_val) or not str(spec_val).strip():
        return {"has_spec": False, "key_count": 0, "is_ruby_hash": False}
    s = str(spec_val).strip()
    keys = re.findall(r'"key"\s*=>\s*"([^"]+)"', s)
    if not keys:
        keys = re.findall(r'"key"\s*:\s*"([^"]+)"', s)
    return {
        "has_spec": len(s) > 10,
        "key_count": len(keys),
        "is_ruby_hash": "=>" in s
    }


def classify_category_relevance(root_cat: str, rules_config: dict) -> str:
    """Classify category into Legal Metrology relevance tiers."""
    cat_rel = rules_config.get("category_relevance", {})
    for tier, cat_list in cat_rel.items():
        if root_cat in cat_list:
            return tier
        for item in cat_list:
            if item.lower() in root_cat.lower() or root_cat.lower() in item.lower():
                return tier
    return "UNKNOWN"


def run_audit(csv_path: str, config_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    total_rows = len(df)
    columns = list(df.columns)
    print(f"Loaded {total_rows} rows, {len(columns)} columns.")

    with open(config_path, "r", encoding="utf-8") as f:
        rules_config = yaml.safe_load(f)

    # 1. Row & URL Duplication
    exact_duplicate_rows = int(df.duplicated().sum())
    unique_product_urls = int(df["product_url"].nunique(dropna=False))
    duplicate_product_urls = total_rows - unique_product_urls
    unique_uniq_ids = int(df["uniq_id"].nunique(dropna=False))
    unique_pids = int(df["pid"].nunique(dropna=True)) if "pid" in df.columns else 0

    # 2. Missingness per field
    missingness = {}
    for col in columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2)
        missingness[col] = {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "present_count": total_rows - missing_count
        }

    # 3. Category Distribution & Relevance
    df["root_category"] = df["product_category_tree"].apply(parse_root_category)
    root_cat_counts = df["root_category"].value_counts().to_dict()

    relevance_counts = Counter()
    category_relevance_map = {}
    for cat, count in root_cat_counts.items():
        rel = classify_category_relevance(cat, rules_config)
        relevance_counts[rel] += count
        category_relevance_map[cat] = {
            "count": count,
            "relevance": rel
        }

    # 4. Description & Specification Richness
    df["desc_len"] = df["description"].fillna("").apply(lambda x: len(str(x)))
    desc_stats = {
        "empty_or_zero_len": int((df["desc_len"] == 0).sum()),
        "under_50_chars": int((df["desc_len"] < 50).sum()),
        "mean_length": round(float(df["desc_len"].mean()), 1),
        "median_length": float(df["desc_len"].median()),
        "max_length": int(df["desc_len"].max())
    }

    spec_profiles = df["product_specifications"].apply(check_spec_richness)
    spec_key_counts = [p["key_count"] for p in spec_profiles]
    spec_stats = {
        "records_with_specs": int(sum(1 for p in spec_profiles if p["has_spec"])),
        "records_with_zero_keys": int(sum(1 for p in spec_profiles if p["key_count"] == 0)),
        "mean_spec_keys": round(float(np.mean(spec_key_counts)), 1),
        "median_spec_keys": float(np.median(spec_key_counts)),
        "max_spec_keys": int(max(spec_key_counts) if spec_key_counts else 0),
        "ruby_hash_format_count": int(sum(1 for p in spec_profiles if p["is_ruby_hash"]))
    }

    # 5. Image URL Availability
    image_counts = df["image"].apply(count_image_urls)
    image_stats = {
        "records_with_at_least_1_image": int((image_counts > 0).sum()),
        "records_with_zero_images": int((image_counts == 0).sum()),
        "mean_images_per_product": round(float(image_counts.mean()), 2),
        "max_images_per_product": int(image_counts.max())
    }

    # 6. Price Completeness
    price_stats = {
        "has_retail_price": int(df["retail_price"].notna().sum()),
        "has_discounted_price": int(df["discounted_price"].notna().sum()),
        "both_prices_present": int((df["retail_price"].notna() & df["discounted_price"].notna()).sum()),
        "neither_price_present": int((df["retail_price"].isna() & df["discounted_price"].isna()).sum())
    }

    # 7. Brand Completeness
    brand_stats = {
        "has_brand": int(df["brand"].notna().sum()),
        "missing_brand": int(df["brand"].isna().sum()),
        "unique_brands": int(df["brand"].nunique(dropna=True))
    }

    # Summary Dict
    audit_summary = {
        "dataset_file": csv_path,
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "exact_duplicate_rows": exact_duplicate_rows,
        "unique_product_urls": unique_product_urls,
        "duplicate_product_urls": duplicate_product_urls,
        "unique_uniq_ids": unique_uniq_ids,
        "unique_pids": unique_pids,
        "missingness": missingness,
        "category_relevance_summary": dict(relevance_counts),
        "top_20_root_categories": dict(list(root_cat_counts.items())[:20]),
        "description_statistics": desc_stats,
        "specification_statistics": spec_stats,
        "image_statistics": image_stats,
        "price_statistics": price_stats,
        "brand_statistics": brand_stats
    }

    json_out = os.path.join(output_dir, "flipkart_dataset_audit.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved JSON audit to {json_out}")

    # Generate Markdown Report
    md_out = os.path.join(output_dir, "flipkart_dataset_audit.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Flipkart Dataset Audit Report (Phase 6A)\n\n")
        f.write(f"**Source File:** `{csv_path}`  \n")
        f.write(f"**Total Records:** {total_rows:,}  \n")
        f.write(f"**Total Columns:** {len(columns)}  \n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Exact Duplicate Rows:** {exact_duplicate_rows}\n")
        f.write(f"- **Unique Product URLs:** {unique_product_urls:,} ({duplicate_product_urls} duplicate URLs)\n")
        f.write(f"- **Unique Uniq IDs:** {unique_uniq_ids:,}\n")
        f.write(f"- **Unique Flipkart PIDs:** {unique_pids:,}\n")
        f.write(f"- **Records with Image URLs:** {image_stats['records_with_at_least_1_image']:,} ({round(image_stats['records_with_at_least_1_image']/total_rows*100, 1)}%)\n")
        f.write(f"- **Records with Structured Specs:** {spec_stats['records_with_specs']:,} ({round(spec_stats['records_with_specs']/total_rows*100, 1)}%)\n")
        f.write(f"- **Records with Retail Price:** {price_stats['has_retail_price']:,} ({round(price_stats['has_retail_price']/total_rows*100, 1)}%)\n\n")

        f.write("## 2. Field Missingness Profile\n\n")
        f.write("| Column Name | Missing Count | Missing % | Available Count |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for col, m in missingness.items():
            f.write(f"| `{col}` | {m['missing_count']:,} | {m['missing_pct']}% | {m['present_count']:,} |\n")
        f.write("\n")

        f.write("## 3. Legal Metrology Category Relevance\n\n")
        f.write("| Relevance Tier | Record Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for tier, count in relevance_counts.most_common():
            pct = round((count / total_rows) * 100, 2)
            f.write(f"| **`{tier}`** | {count:,} | {pct}% |\n")
        f.write("\n")

        f.write("### Top 20 Root Categories\n\n")
        f.write("| Root Category | Count | Relevance Group |\n")
        f.write("| :--- | :--- | :--- |\n")
        for cat, info in list(category_relevance_map.items())[:20]:
            f.write(f"| {cat} | {info['count']:,} | `{info['relevance']}` |\n")
        f.write("\n")

        f.write("## 4. Evidence Richness Statistics\n\n")
        f.write("### Descriptions\n")
        f.write(f"- Mean Length: {desc_stats['mean_length']} characters\n")
        f.write(f"- Median Length: {desc_stats['median_length']} characters\n")
        f.write(f"- Empty Descriptions: {desc_stats['empty_or_zero_len']:,} ({round(desc_stats['empty_or_zero_len']/total_rows*100, 2)}%)\n\n")

        f.write("### Specifications\n")
        f.write(f"- Records with Specs: {spec_stats['records_with_specs']:,} ({round(spec_stats['records_with_specs']/total_rows*100, 2)}%)\n")
        f.write(f"- Mean Key-Value Pairs: {spec_stats['mean_spec_keys']}\n")
        f.write(f"- Ruby Hash Format Count: {spec_stats['ruby_hash_format_count']:,}\n\n")

        f.write("### Product Images\n")
        f.write(f"- Records with >=1 Image URL: {image_stats['records_with_at_least_1_image']:,} ({round(image_stats['records_with_at_least_1_image']/total_rows*100, 2)}%)\n")
        f.write(f"- Mean Images per Product: {image_stats['mean_images_per_product']}\n")

    print(f"Saved Markdown audit to {md_out}")
    return audit_summary


if __name__ == "__main__":
    CSV_PATH = "data/raw/flipkart/flipkart_com-ecommerce_sample.csv"
    CONFIG_PATH = "configs/weak_label_rules.yaml"
    OUTPUT_DIR = "reports/phase_6"
    run_audit(CSV_PATH, CONFIG_PATH, OUTPUT_DIR)
