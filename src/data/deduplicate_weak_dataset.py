"""
deduplicate_weak_dataset.py
============================
Identifies duplicate listings, near-identical titles, shared image URLs,
and clusters products into product families to prevent leakage in future ML splits.

Assigns:
- duplicate_group_id
- product_family_id
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import re
import hashlib
from collections import defaultdict
from typing import List, Dict, Any


def normalize_title_for_clustering(title: str) -> str:
    """Normalize title for fuzzy / family grouping."""
    if not title:
        return "unknown"
    # Lowercase, remove punctuation, collapse whitespace
    t = re.sub(r'[^\w\s]', '', title.lower())
    words = [w for w in t.split() if len(w) > 1 and not w.isdigit()]
    return " ".join(words[:4]) if words else "unknown"


def generate_family_id(brand: str, category: str, norm_title: str) -> str:
    """Generate a deterministic product family ID."""
    b = brand.strip().lower() if brand else "nobrand"
    c = category.strip().lower() if category else "nocat"
    raw_key = f"{b}|{c}|{norm_title}"
    return "fam_" + hashlib.md5(raw_key.encode()).hexdigest()[:12]


def deduplicate_and_assign_families(
    input_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Assign duplicate_group_id and product_family_id to all records."""
    # 1. Exact Duplicate URL / Title map
    url_groups = defaultdict(list)
    title_groups = defaultdict(list)

    for idx, rec in enumerate(input_records):
        url = rec.get("product_url", "").strip()
        title = rec.get("product_name", "").strip().lower()
        if url:
            url_groups[url].append(idx)
        if title:
            title_groups[title].append(idx)

    # 2. Assign Groups & Families
    processed_records = []
    seen_urls = set()

    for idx, rec in enumerate(input_records):
        url = rec.get("product_url", "").strip()
        title = rec.get("product_name", "").strip()
        brand = rec.get("brand", "")
        cat = rec.get("category", "")

        norm_title = normalize_title_for_clustering(title)
        family_id = generate_family_id(brand, cat, norm_title)

        # Duplicate group by exact title or URL
        dup_key = url if url else title.lower()
        dup_group_id = "dup_" + hashlib.md5(dup_key.encode()).hexdigest()[:12]

        enriched_rec = dict(rec)
        enriched_rec["duplicate_group_id"] = dup_group_id
        enriched_rec["product_family_id"] = family_id
        enriched_rec["normalized_family_title"] = norm_title

        processed_records.append(enriched_rec)

    return processed_records
