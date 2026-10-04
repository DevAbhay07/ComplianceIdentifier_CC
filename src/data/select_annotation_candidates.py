"""
select_annotation_candidates.py
================================
Filters the raw Flipkart dataset to products relevant to Legal Metrology
(Packaged Commodities) Rules, 2011 compliance annotation.

Produces: data/interim/annotation_candidates.csv

Usage:
    python src/data/select_annotation_candidates.py
"""

import pandas as pd
import os
import sys

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
RAW_DATA_PATH = "data/raw/flipkart/flipkart_com-ecommerce_sample.csv"
OUTPUT_PATH = "data/interim/annotation_candidates.csv"

# Priority categories for annotation (clearly applicable packaged commodities)
PRIORITY_CATEGORIES = [
    "Beauty and Personal Care",
    "Baby Care",
    "Kitchen & Dining",
]


def get_top_category(tree: str) -> str:
    """Extract the top-level category from a Flipkart category tree string."""
    if pd.isna(tree):
        return "UNKNOWN"
    tree = str(tree).strip()
    if tree.startswith("["):
        tree = tree.strip("[]\"")
    parts = tree.split(" >> ")
    top = parts[0].strip('"') if parts else "UNKNOWN"
    return top


def main():
    # ------------------------------------------------------------------
    # 1. Load raw dataset
    # ------------------------------------------------------------------
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Raw data not found at {RAW_DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    print(f"Total products loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()

    # ------------------------------------------------------------------
    # 2. Parse top-level category
    # ------------------------------------------------------------------
    df["top_category"] = df["product_category_tree"].apply(get_top_category)

    # ------------------------------------------------------------------
    # 3. Show all category counts for context
    # ------------------------------------------------------------------
    print("=== ALL TOP-LEVEL CATEGORIES ===")
    cat_counts = df["top_category"].value_counts()
    for cat, count in cat_counts.head(20).items():
        marker = "  ★" if cat in PRIORITY_CATEGORIES else ""
        print(f"  {cat}: {count}{marker}")
    print(f"  ... ({len(cat_counts)} unique categories total)")
    print()

    # ------------------------------------------------------------------
    # 4. Filter to priority categories
    # ------------------------------------------------------------------
    candidates = df[df["top_category"].isin(PRIORITY_CATEGORIES)].copy()
    print(f"=== FILTERED TO PRIORITY CATEGORIES ===")
    print(f"Candidates: {len(candidates)}")
    print()

    # ------------------------------------------------------------------
    # 5. Per-category breakdown
    # ------------------------------------------------------------------
    print("=== CANDIDATES PER CATEGORY ===")
    for cat in PRIORITY_CATEGORIES:
        count = len(candidates[candidates["top_category"] == cat])
        print(f"  {cat}: {count}")
    print()

    # ------------------------------------------------------------------
    # 6. Data quality summary for candidates
    # ------------------------------------------------------------------
    print("=== DATA QUALITY SUMMARY (candidates only) ===")

    quality_checks = {
        "product_name": "Product name",
        "description": "Description",
        "product_specifications": "Product specifications",
        "brand": "Brand",
        "retail_price": "Retail price",
        "image": "Image URLs",
    }

    for col, label in quality_checks.items():
        null_count = candidates[col].isna().sum()
        pct = null_count / len(candidates) * 100 if len(candidates) > 0 else 0
        print(f"  {label}: {null_count} missing ({pct:.1f}%)")

    # Check for very short descriptions (potentially empty/useless)
    desc_lengths = candidates["description"].fillna("").str.len()
    short_desc = (desc_lengths < 50).sum()
    print(f"  Descriptions < 50 chars: {short_desc} ({short_desc / len(candidates) * 100:.1f}%)")

    # Check for empty specs
    spec_empty = candidates["product_specifications"].fillna("").str.len() < 10
    print(f"  Specifications < 10 chars: {spec_empty.sum()} ({spec_empty.sum() / len(candidates) * 100:.1f}%)")
    print()

    # ------------------------------------------------------------------
    # 7. Drop rows with null uniq_id (unusable)
    # ------------------------------------------------------------------
    before = len(candidates)
    candidates = candidates.dropna(subset=["uniq_id"])
    dropped = before - len(candidates)
    if dropped > 0:
        print(f"Dropped {dropped} rows with null uniq_id")
        print()

    # ------------------------------------------------------------------
    # 8. Save candidates
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    candidates.to_csv(OUTPUT_PATH, index=False)
    print(f"=== SAVED ===")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Rows: {len(candidates)}")
    print(f"  Columns: {list(candidates.columns)}")


if __name__ == "__main__":
    main()
