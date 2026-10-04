"""
create_pilot_sample.py
========================
Creates a stratified pilot sample of ~75 products for the annotation pilot.

Sampling strategy:
  - Proportional representation across 3 priority categories
  - Stratified to include products with/without brand, with/without specs,
    with/without price, and varying description richness
  - Fixed random seed for reproducibility

Input:  data/interim/parsed_annotation_candidates.jsonl
Output: annotations/pilot_manifest.csv

Usage:
    python src/data/create_pilot_sample.py
"""

import json
import pandas as pd
import os
import sys

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
INPUT_PATH = "data/interim/parsed_annotation_candidates.jsonl"
OUTPUT_PATH = "annotations/pilot_manifest.csv"
RANDOM_SEED = 42
TARGET_SAMPLE_SIZE = 75  # within the 50-100 range


def load_parsed_candidates(path: str) -> pd.DataFrame:
    """Load the JSONL parsed candidates into a DataFrame."""
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def create_stratification_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary stratification features for diverse sampling."""
    df = df.copy()

    # Has brand?
    df["has_brand"] = df["brand"].apply(lambda x: bool(x and str(x).strip()))

    # Has price?
    df["has_price"] = df["retail_price"].notna()

    # Description richness (short vs long)
    df["desc_length"] = df["description"].fillna("").str.len()
    median_desc = df["desc_length"].median()
    df["desc_rich"] = df["desc_length"] > median_desc

    # Spec richness
    df["spec_rich"] = df["spec_key_count"] >= 5

    # Create a combined stratification key
    df["strat_key"] = (
        df["top_category"]
        + "|brand:"
        + df["has_brand"].astype(str)
        + "|spec:"
        + df["spec_rich"].astype(str)
    )

    return df


def stratified_sample(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    """
    Stratified sample that ensures representation of:
    - All 3 categories
    - Products WITH and WITHOUT brand
    - Products WITH and WITHOUT rich specifications
    """
    # Target per-category allocation (roughly proportional)
    cat_counts = df["top_category"].value_counts()
    total = len(df)
    cat_targets = {}
    for cat, count in cat_counts.items():
        proportion = count / total
        cat_targets[cat] = max(10, round(proportion * target_size))

    # Adjust to hit target
    while sum(cat_targets.values()) > target_size:
        largest = max(cat_targets, key=cat_targets.get)
        cat_targets[largest] -= 1
    while sum(cat_targets.values()) < target_size:
        smallest = min(cat_targets, key=cat_targets.get)
        cat_targets[smallest] += 1

    print(f"  Per-category targets:")
    for cat, target in cat_targets.items():
        print(f"    {cat}: {target} (from {cat_counts[cat]} available)")

    # Sample within each category using stratification
    samples = []
    for cat, target in cat_targets.items():
        cat_df = df[df["top_category"] == cat].copy()

        if len(cat_df) <= target:
            # Take all if fewer than target
            samples.append(cat_df)
            continue

        # Within this category, ensure diversity
        # Split into 4 strata: brand+spec, brand+nospec, nobrand+spec, nobrand+nospec
        strata_dfs = []
        for has_brand in [True, False]:
            for spec_rich in [True, False]:
                mask = (cat_df["has_brand"] == has_brand) & (cat_df["spec_rich"] == spec_rich)
                stratum = cat_df[mask]
                if len(stratum) > 0:
                    strata_dfs.append(stratum)

        # Allocate samples across strata (proportional, minimum 1 per non-empty stratum)
        if strata_dfs:
            per_stratum = max(1, target // len(strata_dfs))
            remainder = target - per_stratum * len(strata_dfs)

            cat_samples = []
            for i, stratum in enumerate(strata_dfs):
                n = per_stratum + (1 if i < remainder else 0)
                n = min(n, len(stratum))
                sampled = stratum.sample(n=n, random_state=seed)
                cat_samples.append(sampled)

            cat_sample = pd.concat(cat_samples)

            # If we haven't reached target, fill from remaining
            if len(cat_sample) < target:
                already_selected = set(cat_sample["product_id"])
                remaining = cat_df[~cat_df["product_id"].isin(already_selected)]
                extra_needed = target - len(cat_sample)
                extra = remaining.sample(n=min(extra_needed, len(remaining)), random_state=seed)
                cat_sample = pd.concat([cat_sample, extra])

            samples.append(cat_sample.head(target))
        else:
            samples.append(cat_df.sample(n=min(target, len(cat_df)), random_state=seed))

    return pd.concat(samples).reset_index(drop=True)


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input not found at {INPUT_PATH}")
        print("Run parse_specifications.py first.")
        sys.exit(1)

    df = load_parsed_candidates(INPUT_PATH)
    print(f"Loaded {len(df)} parsed candidates")
    print()

    # ------------------------------------------------------------------
    # Add stratification features
    # ------------------------------------------------------------------
    df = create_stratification_features(df)

    print(f"=== STRATIFICATION SUMMARY ===")
    print(f"  Categories: {df['top_category'].nunique()}")
    print(f"  Has brand: {df['has_brand'].sum()} ({df['has_brand'].mean() * 100:.1f}%)")
    print(f"  Has price: {df['has_price'].sum()} ({df['has_price'].mean() * 100:.1f}%)")
    print(f"  Rich description: {df['desc_rich'].sum()} ({df['desc_rich'].mean() * 100:.1f}%)")
    print(f"  Rich specs (≥5 keys): {df['spec_rich'].sum()} ({df['spec_rich'].mean() * 100:.1f}%)")
    print()

    # ------------------------------------------------------------------
    # Create stratified sample
    # ------------------------------------------------------------------
    print(f"=== CREATING PILOT SAMPLE ===")
    print(f"  Target size: {TARGET_SAMPLE_SIZE}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()

    pilot = stratified_sample(df, TARGET_SAMPLE_SIZE, RANDOM_SEED)

    # ------------------------------------------------------------------
    # Verify sample diversity
    # ------------------------------------------------------------------
    print()
    print(f"=== PILOT SAMPLE SUMMARY ===")
    print(f"  Total selected: {len(pilot)}")
    print()

    print(f"  Category distribution:")
    for cat, count in pilot["top_category"].value_counts().items():
        print(f"    {cat}: {count}")

    print()
    print(f"  Diversity metrics:")
    print(f"    With brand: {pilot['has_brand'].sum()} ({pilot['has_brand'].mean() * 100:.1f}%)")
    print(f"    Without brand: {(~pilot['has_brand']).sum()} ({(~pilot['has_brand']).mean() * 100:.1f}%)")
    print(f"    Rich specs: {pilot['spec_rich'].sum()} ({pilot['spec_rich'].mean() * 100:.1f}%)")
    print(f"    Sparse specs: {(~pilot['spec_rich']).sum()} ({(~pilot['spec_rich']).mean() * 100:.1f}%)")
    print(f"    Has price: {pilot['has_price'].sum()}")
    print(f"    Missing price: {(~pilot['has_price']).sum()}")
    print(f"    Unique product IDs: {pilot['product_id'].nunique()}")

    # ------------------------------------------------------------------
    # Save pilot manifest
    # ------------------------------------------------------------------
    manifest_cols = [
        "product_id",
        "product_name",
        "top_category",
        "brand",
        "retail_price",
        "has_brand",
        "spec_key_count",
        "spec_rich",
        "desc_length",
    ]
    manifest = pilot[manifest_cols].copy()
    manifest["pilot_index"] = range(1, len(manifest) + 1)
    manifest["annotation_status"] = "NOT_STARTED"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"=== SAVED ===")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Rows: {len(manifest)}")
    print(f"  Columns: {list(manifest.columns)}")


if __name__ == "__main__":
    main()
