"""
build_automated_candidate_pool.py
==================================
Diversity-aware automated candidate pool selection for Phase 6 weak labeling.
Samples diverse, high-quality candidate products across Legal Metrology
relevance tiers, categories, subcategories, price brackets, and evidence richness.

Inputs:
    data/raw/flipkart/flipkart_com-ecommerce_sample.csv
    data/processed/pilot_ground_truth_75_v1.jsonl (for exclusion)
    configs/weak_label_rules.yaml

Output:
    data/interim/automated_candidate_pool.csv
"""

import os
import sys
import json
import re
import yaml
import pandas as pd
import numpy as np


def parse_root_category(cat_tree_str: str) -> str:
    if pd.isna(cat_tree_str) or not str(cat_tree_str).strip():
        return "Unknown"
    s = str(cat_tree_str).strip()
    match = re.search(r'\["([^">]+)', s)
    return match.group(1).strip() if match else "Unknown"


def parse_subcategory(cat_tree_str: str) -> str:
    if pd.isna(cat_tree_str) or not str(cat_tree_str).strip():
        return "Unknown"
    s = str(cat_tree_str).strip()
    cleaned = re.sub(r'^\["|"]$', '', s)
    parts = [p.strip().replace('"', '') for p in cleaned.split(">>") if p.strip()]
    return parts[1] if len(parts) > 1 else (parts[0] if parts else "Unknown")


def count_image_urls(image_val) -> int:
    if pd.isna(image_val) or not str(image_val).strip():
        return 0
    urls = re.findall(r'https?://[^\s",\]]+', str(image_val))
    return len(urls)


def count_spec_keys(spec_val) -> int:
    if pd.isna(spec_val) or not str(spec_val).strip():
        return 0
    s = str(spec_val).strip()
    keys = re.findall(r'"key"\s*=>\s*"([^"]+)"', s)
    if not keys:
        keys = re.findall(r'"key"\s*:\s*"([^"]+)"', s)
    return len(keys)


def assign_price_tier(price) -> str:
    if pd.isna(price) or price <= 0:
        return "UNKNOWN"
    if price < 300:
        return "BUDGET"
    elif price <= 1000:
        return "MID_RANGE"
    elif price <= 3000:
        return "PREMIUM"
    else:
        return "LUXURY"


def load_excluded_pilot_ids(pilot_path: str) -> set:
    excluded = set()
    if os.path.exists(pilot_path):
        with open(pilot_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    pid = rec.get("product_id") or rec.get("uniq_id")
                    if pid:
                        excluded.add(str(pid).strip())
    return excluded


def build_candidate_pool(
    raw_csv_path: str,
    rules_config_path: str,
    pilot_ground_truth_path: str,
    output_csv_path: str,
    target_pool_size: int = 1500
):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, encoding="utf-8-sig", low_memory=False)

    with open(rules_config_path, "r", encoding="utf-8") as f:
        rules_config = yaml.safe_load(f)

    excluded_ids = load_excluded_pilot_ids(pilot_ground_truth_path)
    print(f"Loaded {len(excluded_ids)} pilot ground truth product IDs for exclusion.")

    # Data hygiene filters
    df = df[df["uniq_id"].notna()].copy()
    df["uniq_id"] = df["uniq_id"].astype(str).str.strip()
    df = df[~df["uniq_id"].isin(excluded_ids)].copy()
    df = df[df["product_name"].notna() & (df["product_name"].str.strip() != "")].copy()
    df = df[df["retail_price"].notna() & (pd.to_numeric(df["retail_price"], errors='coerce') > 0)].copy()

    # Feature extraction for filtering & diversity
    df["retail_price"] = pd.to_numeric(df["retail_price"], errors='coerce')
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors='coerce')
    df["root_category"] = df["product_category_tree"].apply(parse_root_category)
    df["subcategory"] = df["product_category_tree"].apply(parse_subcategory)
    df["image_count"] = df["image"].apply(count_image_urls)
    df["spec_count"] = df["product_specifications"].apply(count_spec_keys)
    df["desc_len"] = df["description"].fillna("").apply(lambda x: len(str(x).strip()))
    df["price_tier"] = df["retail_price"].apply(assign_price_tier)

    # Classify relevance
    cat_relevance = rules_config.get("category_relevance", {})
    def get_relevance(cat):
        for tier, cat_list in cat_relevance.items():
            if cat in cat_list:
                return tier
            for item in cat_list:
                if item.lower() in cat.lower() or cat.lower() in item.lower():
                    return tier
        return "UNKNOWN"

    df["relevance_group"] = df["root_category"].apply(get_relevance)

    # Quality criteria:
    # 1. Must have at least 1 image URL
    # 2. Must have description >= 20 chars
    # 3. Must have spec_count >= 2
    qual_mask = (df["image_count"] >= 1) & (df["desc_len"] >= 20) & (df["spec_count"] >= 2)
    qualified_df = df[qual_mask].copy()
    print(f"Qualified candidates after basic quality filtering: {len(qualified_df)} records.")

    # Diversity-Aware Stratified Sampling
    # Quotas by Relevance Tier
    # HIGH_RELEVANCE: 45% of target
    # MEDIUM_RELEVANCE: 40% of target
    # LOW_RELEVANCE: 10% of target
    # UNKNOWN / OTHER: 5% of target
    
    tier_quotas = {
        "HIGH_RELEVANCE": int(target_pool_size * 0.45),
        "MEDIUM_RELEVANCE": int(target_pool_size * 0.40),
        "LOW_RELEVANCE": int(target_pool_size * 0.10),
        "UNKNOWN": int(target_pool_size * 0.05)
    }

    selected_dfs = []
    np.random.seed(42)

    for tier, quota in tier_quotas.items():
        tier_pool = qualified_df[qualified_df["relevance_group"] == tier].copy()
        if len(tier_pool) == 0:
            continue
        
        # Sub-sample across root_category and subcategory
        cats = tier_pool["root_category"].unique()
        per_cat_quota = max(1, quota // len(cats)) if len(cats) > 0 else quota
        
        tier_selected = []
        for cat in cats:
            cat_pool = tier_pool[tier_pool["root_category"] == cat]
            # Deduplicate closely matching titles within same category
            cat_pool = cat_pool.drop_duplicates(subset=["product_name"], keep="first")
            sample_size = min(len(cat_pool), per_cat_quota)
            if sample_size > 0:
                tier_selected.append(cat_pool.sample(n=sample_size, random_state=42))
        
        if tier_selected:
            tier_df = pd.concat(tier_selected, ignore_index=True)
            # If we need more to meet quota, fill from remaining
            if len(tier_df) < quota and len(tier_pool) > len(tier_df):
                remaining = tier_pool[~tier_pool["uniq_id"].isin(tier_df["uniq_id"])]
                needed = min(len(remaining), quota - len(tier_df))
                if needed > 0:
                    tier_df = pd.concat([tier_df, remaining.sample(n=needed, random_state=42)], ignore_index=True)
            selected_dfs.append(tier_df)

    final_pool = pd.concat(selected_dfs, ignore_index=True).drop_duplicates(subset=["uniq_id"])
    print(f"Final sampled candidate pool size: {len(final_pool)} products.")

    # Annotate selection metadata
    final_pool["candidate_selection_reason"] = (
        "Qualified on image availability, spec richness (>=2 keys), "
        "description length (>=20 chars), valid price, and category relevance quota."
    )
    final_pool["selection_group"] = final_pool["relevance_group"]
    final_pool["has_description"] = final_pool["desc_len"] > 0
    final_pool["has_specifications"] = final_pool["spec_count"] > 0
    final_pool["has_image"] = final_pool["image_count"] > 0
    final_pool["specification_count"] = final_pool["spec_count"]
    final_pool["description_length"] = final_pool["desc_len"]

    # Select and save canonical output columns
    out_cols = [
        "uniq_id",
        "product_url",
        "product_name",
        "root_category",
        "subcategory",
        "brand",
        "retail_price",
        "discounted_price",
        "price_tier",
        "has_description",
        "description_length",
        "has_specifications",
        "specification_count",
        "has_image",
        "image_count",
        "relevance_group",
        "selection_group",
        "candidate_selection_reason"
    ]
    
    # Save CSV
    final_pool[out_cols].to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved candidate pool to {output_csv_path}")

    return final_pool


if __name__ == "__main__":
    RAW_CSV = "data/raw/flipkart/flipkart_com-ecommerce_sample.csv"
    RULES_YAML = "configs/weak_label_rules.yaml"
    PILOT_JSONL = "data/processed/pilot_ground_truth_75_v1.jsonl"
    OUTPUT_CSV = "data/interim/automated_candidate_pool.csv"

    build_candidate_pool(RAW_CSV, RULES_YAML, PILOT_JSONL, OUTPUT_CSV)
