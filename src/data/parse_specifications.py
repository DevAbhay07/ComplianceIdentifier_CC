"""
parse_specifications.py
========================
Parses the Ruby-hash-like product_specifications column from the
Flipkart dataset into structured JSON key-value pairs.

Input:  data/interim/annotation_candidates.csv
Output: data/interim/parsed_annotation_candidates.jsonl

Usage:
    python src/data/parse_specifications.py
"""

import pandas as pd
import json
import re
import os
import sys


def parse_ruby_hash_specs(raw_spec: str) -> dict:
    """
    Parse the Ruby-hash-like specification format into a Python dict.

    The format is like:
      {"product_specification"=>[{"key"=>"Brand", "value"=>"ABC"}, ...]}

    Some records have a single dict instead of a list:
      {"product_specification"=>{"key"=>"Material", "value"=>"Glass"}}

    Some entries have value-only items (no key):
      {"value"=>"Some extra text"}

    Returns:
        {
            "specs": {"Brand": "ABC", "Material": "Glass", ...},
            "extra_values": ["Some extra text", ...],
            "parse_success": True/False,
            "parse_error": null or error message,
            "raw_spec_length": int
        }
    """
    result = {
        "specs": {},
        "extra_values": [],
        "parse_success": False,
        "parse_error": None,
        "raw_spec_length": 0,
    }

    if pd.isna(raw_spec) or not str(raw_spec).strip():
        result["parse_error"] = "empty_or_null"
        return result

    raw = str(raw_spec).strip()
    result["raw_spec_length"] = len(raw)

    try:
        # Strategy: Convert Ruby hash syntax to JSON
        # Replace =>  with :
        # Replace nil with null
        converted = raw.replace("=>", ":")
        converted = re.sub(r'\bnil\b', 'null', converted)

        # Try parsing as JSON
        parsed = json.loads(converted)

        # Extract the specification list
        spec_data = parsed.get("product_specification", parsed)

        # Handle single dict vs list
        if isinstance(spec_data, dict):
            spec_data = [spec_data]

        if isinstance(spec_data, list):
            for item in spec_data:
                if isinstance(item, dict):
                    key = item.get("key", "").strip() if item.get("key") else None
                    value = item.get("value", "").strip() if item.get("value") else ""

                    if key:
                        result["specs"][key] = value
                    elif value:
                        # Value-only entry (no key)
                        result["extra_values"].append(value)

            result["parse_success"] = True
        else:
            result["parse_error"] = f"unexpected_spec_type:{type(spec_data).__name__}"

    except (json.JSONDecodeError, ValueError) as e:
        result["parse_error"] = f"json_parse_error:{str(e)[:100]}"
    except Exception as e:
        result["parse_error"] = f"unexpected_error:{str(e)[:100]}"

    return result


def main():
    INPUT_PATH = "data/interim/annotation_candidates.csv"
    OUTPUT_PATH = "data/interim/parsed_annotation_candidates.jsonl"

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input not found at {INPUT_PATH}")
        print("Run select_annotation_candidates.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"Loaded {len(df)} annotation candidates")

    # ------------------------------------------------------------------
    # Parse specifications for each row
    # ------------------------------------------------------------------
    parse_results = []
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        parsed = parse_ruby_hash_specs(row.get("product_specifications"))

        # Build the output record
        record = {
            "product_id": row["uniq_id"],
            "product_name": row.get("product_name", ""),
            "top_category": row.get("top_category", ""),
            "brand": row.get("brand", "") if pd.notna(row.get("brand")) else "",
            "retail_price": float(row["retail_price"]) if pd.notna(row.get("retail_price")) else None,
            "discounted_price": float(row["discounted_price"]) if pd.notna(row.get("discounted_price")) else None,
            "description": row.get("description", "") if pd.notna(row.get("description")) else "",
            "image_urls": row.get("image", "") if pd.notna(row.get("image")) else "",
            "product_category_tree": row.get("product_category_tree", ""),
            "parsed_specs": parsed["specs"],
            "extra_spec_values": parsed["extra_values"],
            "spec_parse_success": parsed["parse_success"],
            "spec_parse_error": parsed["parse_error"],
            "spec_key_count": len(parsed["specs"]),
        }

        parse_results.append(record)

        if parsed["parse_success"]:
            success_count += 1
        else:
            fail_count += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(f"=== SPECIFICATION PARSING SUMMARY ===")
    print(f"  Successfully parsed: {success_count} ({success_count / len(df) * 100:.1f}%)")
    print(f"  Failed to parse:     {fail_count} ({fail_count / len(df) * 100:.1f}%)")

    # Error breakdown
    error_types = {}
    for r in parse_results:
        err = r.get("spec_parse_error")
        if err:
            err_type = err.split(":")[0]
            error_types[err_type] = error_types.get(err_type, 0) + 1

    if error_types:
        print(f"  Error types:")
        for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {et}: {count}")

    # Spec key count distribution
    key_counts = [r["spec_key_count"] for r in parse_results]
    print()
    print(f"  Spec keys per product:")
    print(f"    Min: {min(key_counts)}")
    print(f"    Max: {max(key_counts)}")
    print(f"    Mean: {sum(key_counts) / len(key_counts):.1f}")
    print(f"    Products with 0 keys: {sum(1 for k in key_counts if k == 0)}")

    # Most common spec keys
    all_keys = {}
    for r in parse_results:
        for k in r["parsed_specs"]:
            all_keys[k] = all_keys.get(k, 0) + 1

    print()
    print(f"  Most common specification keys (top 20):")
    for k, count in sorted(all_keys.items(), key=lambda x: -x[1])[:20]:
        print(f"    {k}: {count}")

    # ------------------------------------------------------------------
    # Save as JSONL
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for record in parse_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print(f"=== SAVED ===")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Records: {len(parse_results)}")
    print(f"  Format: JSONL (one JSON object per line)")


if __name__ == "__main__":
    main()
