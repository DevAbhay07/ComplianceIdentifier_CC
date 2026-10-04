"""
extract_product_evidence.py
============================
Extracts, normalizes, and packages multi-source evidence for all candidate products.
Produces a canonical structured evidence stream for automated legal declaration detection.

Inputs:
    data/interim/automated_candidate_pool.csv
    data/raw/flipkart/flipkart_com-ecommerce_sample.csv

Outputs:
    data/interim/product_evidence.jsonl
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import re
import unicodedata
from datetime import datetime, timezone
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.data.image_evidence import parse_image_urls, inspect_product_images


def normalize_text(raw_text: Any) -> str:
    """Normalize whitespace and Unicode characters while preserving semantics."""
    if pd.isna(raw_text) or not str(raw_text).strip():
        return ""
    text = str(raw_text)
    # Unicode decomposition/normalization
    text = unicodedata.normalize("NFKD", text)
    # Replace non-breaking spaces and unusual whitespace
    text = re.sub(r'[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]', ' ', text)
    # Replace smart quotes with standard quotes
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_ruby_hash_specs(raw_spec: Any) -> Tuple[Dict[str, str], List[str], str]:
    """
    Parse Ruby-hash formatted specification string into structured key-value pairs.
    Returns (specs_dict, extra_values_list, status_string).
    """
    if pd.isna(raw_spec) or not str(raw_spec).strip():
        return {}, [], "EMPTY"

    raw = str(raw_spec).strip()
    specs = {}
    extra_values = []

    try:
        converted = raw.replace("=>", ":")
        converted = re.sub(r'\bnil\b', 'null', converted)
        parsed = json.loads(converted)

        spec_data = parsed.get("product_specification", parsed)
        if isinstance(spec_data, dict):
            spec_data = [spec_data]

        if isinstance(spec_data, list):
            for item in spec_data:
                if isinstance(item, dict):
                    k = normalize_text(item.get("key"))
                    v = normalize_text(item.get("value"))
                    if k and v:
                        specs[k] = v
                    elif v:
                        extra_values.append(v)
            return specs, extra_values, "SUCCESS"
        else:
            return {}, [], "UNEXPECTED_SPEC_TYPE"

    except Exception:
        # Fallback to robust regex pattern extraction
        try:
            kv_pairs = re.findall(r'"key"\s*=>\s*"([^"]+)"\s*,\s*"value"\s*=>\s*"([^"]+)"', raw)
            for k, v in kv_pairs:
                specs[normalize_text(k)] = normalize_text(v)
            
            val_only = re.findall(r'\{\s*"value"\s*=>\s*"([^"]+)"\s*\}', raw)
            for v in val_only:
                extra_values.append(normalize_text(v))
            
            if specs or extra_values:
                return specs, extra_values, "REGEX_FALLBACK"
            return {}, [], "PARSE_FAILED"
        except Exception as e:
            return {}, [], f"FAILED: {str(e)[:50]}"


def extract_evidence_record(row: pd.Series, check_images: bool = False) -> Dict[str, Any]:
    """Build a comprehensive evidence record for a single product."""
    product_id = str(row.get("uniq_id", "")).strip()
    raw_title = str(row.get("product_name", "") if pd.notna(row.get("product_name")) else "")
    raw_desc = str(row.get("description", "") if pd.notna(row.get("description")) else "")
    raw_specs = row.get("product_specifications")
    raw_cat_tree = str(row.get("product_category_tree", "") if pd.notna(row.get("product_category_tree")) else "")
    raw_brand = str(row.get("brand", "") if pd.notna(row.get("brand")) else "")

    # Normalization
    title_norm = normalize_text(raw_title)
    desc_norm = normalize_text(raw_desc)
    brand_norm = normalize_text(raw_brand)

    # Specs parsing
    specs_dict, extra_vals, spec_status = parse_ruby_hash_specs(raw_specs)

    # Category hierarchy
    cat_cleaned = re.sub(r'^\["|"]$', '', raw_cat_tree.strip())
    cat_hierarchy = [normalize_text(p.replace('"', '')) for p in cat_cleaned.split(">>") if p.strip()]
    root_category = cat_hierarchy[0] if cat_hierarchy else "Unknown"
    leaf_category = cat_hierarchy[-1] if cat_hierarchy else "Unknown"

    # Price parsing
    retail_price = None
    discounted_price = None
    try:
        if pd.notna(row.get("retail_price")):
            retail_price = float(row.get("retail_price"))
    except Exception:
        pass

    try:
        if pd.notna(row.get("discounted_price")):
            discounted_price = float(row.get("discounted_price"))
    except Exception:
        pass

    # Image URLs & inspection
    img_urls = parse_image_urls(row.get("image"))
    img_inspection = inspect_product_images(img_urls, attempt_download=check_images)

    # Evidence sources tracking
    sources = []
    if title_norm:
        sources.append("PRODUCT_TITLE")
    if desc_norm:
        sources.append("PRODUCT_DESCRIPTION")
    if specs_dict or extra_vals:
        sources.append("PRODUCT_SPECIFICATIONS")
    if brand_norm:
        sources.append("BRAND_FIELD")
    if retail_price is not None:
        sources.append("PRICE_FIELD")
    if img_urls:
        sources.append("IMAGE_METADATA")

    # Status determination
    if len(sources) >= 4 and specs_dict:
        status = "SUCCESS"
    elif len(sources) >= 2:
        status = "PARTIAL"
    else:
        status = "FAILED"

    return {
        "product_id": product_id,
        "product_url": str(row.get("product_url", "") if pd.notna(row.get("product_url")) else ""),
        "pid": str(row.get("pid", "") if pd.notna(row.get("pid")) else ""),
        "title": title_norm,
        "raw_title": raw_title,
        "description": desc_norm,
        "raw_description": raw_desc,
        "brand": brand_norm,
        "retail_price": retail_price,
        "discounted_price": discounted_price,
        "category_hierarchy": cat_hierarchy,
        "root_category": root_category,
        "leaf_category": leaf_category,
        "specifications": specs_dict,
        "extra_spec_values": extra_vals,
        "spec_parse_status": spec_status,
        "image_urls": img_urls,
        "accessible_images": img_inspection["accessible_images"],
        "image_metadata": img_inspection["image_metadata"],
        "image_evidence_status": img_inspection["image_evidence_status"],
        "evidence_sources": sources,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "extraction_status": status,
        "extraction_version": "v1"
    }


def run_evidence_extraction(
    candidates_csv_path: str,
    raw_csv_path: str,
    output_jsonl_path: str,
    check_images: bool = False
):
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    print(f"Reading candidate pool from {candidates_csv_path}...")
    candidates_df = pd.read_csv(candidates_csv_path, encoding="utf-8-sig")
    candidate_ids = set(candidates_df["uniq_id"].astype(str).str.strip())
    print(f"Loaded {len(candidate_ids)} candidate IDs.")

    print(f"Reading raw dataset from {raw_csv_path} to retrieve complete text & specs...")
    raw_df = pd.read_csv(raw_csv_path, encoding="utf-8-sig", low_memory=False)
    raw_df = raw_df[raw_df["uniq_id"].notna()].copy()
    raw_df["uniq_id"] = raw_df["uniq_id"].astype(str).str.strip()

    # Join or filter candidates
    matched_df = raw_df[raw_df["uniq_id"].isin(candidate_ids)].copy()
    print(f"Matched {len(matched_df)} candidate rows in raw data.")

    records = []
    success_count = 0
    partial_count = 0
    failed_count = 0

    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for idx, row in matched_df.iterrows():
            record = extract_evidence_record(row, check_images=check_images)
            records.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if record["extraction_status"] == "SUCCESS":
                success_count += 1
            elif record["extraction_status"] == "PARTIAL":
                partial_count += 1
            else:
                failed_count += 1

    print(f"Extraction complete:")
    print(f"  Total records extracted: {len(records)}")
    print(f"  SUCCESS: {success_count} ({round(success_count/len(records)*100, 1)}%)")
    print(f"  PARTIAL: {partial_count} ({round(partial_count/len(records)*100, 1)}%)")
    print(f"  FAILED: {failed_count} ({round(failed_count/len(records)*100, 1)}%)")
    print(f"Saved product evidence stream to {output_jsonl_path}")

    return records


if __name__ == "__main__":
    CANDIDATES_CSV = "data/interim/automated_candidate_pool.csv"
    RAW_CSV = "data/raw/flipkart/flipkart_com-ecommerce_sample.csv"
    OUTPUT_JSONL = "data/interim/product_evidence.jsonl"

    run_evidence_extraction(CANDIDATES_CSV, RAW_CSV, OUTPUT_JSONL)
