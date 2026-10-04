"""
generate_annotation_draft.py
==============================
Deterministic draft annotation engine for NitiMark compliance labeling.

Generates DRAFT recommendations for declarations D01-D07 based on
available listing evidence. These are NOT ground truth — they are
pre-filled suggestions for the human annotator to review and modify.

CRITICAL: Machine-generated labels MUST NEVER automatically become
ground truth. The human annotator must explicitly review and save.

Usage:
    # As a module (imported by annotation_tool.py)
    from src.data.generate_annotation_draft import generate_draft

    draft = generate_draft(product_record)

    # As a standalone test
    python src/data/generate_annotation_draft.py
"""

import json
import re
import os
import sys
from datetime import datetime, timezone


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
VALID_LABELS = ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"]
OVERALL_LABELS = ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"]

# Patterns for quantity detection in product names and descriptions
QUANTITY_PATTERNS = [
    r'\b\d+(?:\.\d+)?\s*(?:ml|l|litre|liter|litres|liters)\b',
    r'\b\d+(?:\.\d+)?\s*(?:g|gm|gms|gram|grams|kg|kgs)\b',
    r'\b\d+(?:\.\d+)?\s*(?:oz|fl\.?\s*oz)\b',
    r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|inch|inches)\b',
    r'\bpack\s*(?:of\s*)?\d+\b',
    r'\bset\s*(?:of\s*)?\d+\b',
    r'\b\d+\s*(?:pc|pcs|piece|pieces|count|units?|nos?)\b',
]

# Keywords for consumer care detection
CONSUMER_CARE_KEYWORDS = [
    "customer care", "consumer care", "helpline", "toll free",
    "toll-free", "complaint", "grievance", "contact us",
    "support@", "care@", "help@",
    r'\b1800\s*\d+', r'\b1860\s*\d+',  # Indian toll-free prefixes
    r'\+91[\s-]*\d{10}',  # Indian phone
]

# Keywords for country of origin detection
ORIGIN_SPEC_KEYWORDS = [
    "country of origin", "country_of_origin", "countryoforigin",
    "made in", "manufactured in", "imported from", "origin",
    "imported by", "importer",
]

# Keywords for manufacturer/address detection
MANUFACTURER_KEYWORDS = [
    "manufacturer", "manufactured by", "mfg by", "mfg.",
    "packer", "packed by", "packing",
    "importer", "imported by",
    "marketed by", "marketer",
]

ADDRESS_KEYWORDS = [
    "address", "pin code", "pincode", "pin:", "zip",
    "road", "street", "lane", "nagar", "colony",
    "dist.", "district", "state", "india",
    "mumbai", "delhi", "bangalore", "bengaluru", "chennai",
    "hyderabad", "kolkata", "pune", "ahmedabad", "jaipur",
]

# Common generic commodity type words
GENERIC_COMMODITY_WORDS = [
    "cream", "lotion", "gel", "serum", "oil", "shampoo", "conditioner",
    "soap", "wash", "cleanser", "moisturizer", "moisturiser",
    "spray", "deodorant", "perfume", "fragrance", "cologne",
    "lipstick", "lip balm", "lip gloss", "mascara", "eyeliner",
    "foundation", "concealer", "powder", "blush", "highlighter",
    "nail polish", "nail paint", "remover",
    "diaper", "wipe", "bottle", "feeder", "pacifier", "teether",
    "romper", "onesie", "bib", "blanket",
    "pan", "pot", "kadhai", "tawa", "cooker", "kettle",
    "plate", "bowl", "glass", "mug", "cup", "jar", "container",
    "spoon", "fork", "knife", "ladle", "spatula", "tong",
    "mixer", "grinder", "blender", "juicer", "toaster",
    "box", "set", "kit", "combo",
    "brush", "comb", "sponge", "puff",
    "mask", "scrub", "pack", "balm", "ointment",
    "drops", "solution", "tonic", "essence",
    "top", "dress", "shirt", "shorts", "pants",
]


def _search_text(text: str, keywords: list, use_regex: bool = False) -> list:
    """Search text for keywords. Returns list of matches found."""
    if not text:
        return []
    text_lower = text.lower()
    matches = []
    for kw in keywords:
        if use_regex:
            found = re.findall(kw, text_lower, re.IGNORECASE)
            if found:
                matches.extend(found)
        else:
            if kw.lower() in text_lower:
                matches.append(kw)
    return matches


def _extract_quantities_from_text(text: str) -> list:
    """Find quantity mentions in text using regex patterns."""
    if not text:
        return []
    matches = []
    for pattern in QUANTITY_PATTERNS:
        found = re.findall(pattern, text, re.IGNORECASE)
        matches.extend(found)
    return matches


# -------------------------------------------------------------------------
# Per-Declaration Draft Generators
# -------------------------------------------------------------------------

def draft_d01(product: dict) -> dict:
    """D01: Manufacturer/Packer/Importer Name & Address."""
    brand = str(product.get("brand", "")).strip()
    specs = product.get("parsed_specs", {})
    desc = str(product.get("description", ""))
    name = str(product.get("product_name", ""))
    extras = product.get("extra_spec_values", [])
    all_text = f"{desc} {' '.join(str(v) for v in extras)}"

    # Check specs for manufacturer keys
    mfg_spec_keys = [k for k in specs
                     if any(w in k.lower() for w in ["manufacturer", "packer", "importer", "marketed"])]
    mfg_spec_values = {k: specs[k] for k in mfg_spec_keys}

    # Check for address-like info
    address_in_specs = [k for k in specs if any(w in k.lower() for w in ["address"])]
    address_in_desc = _search_text(desc, ADDRESS_KEYWORDS)
    mfg_in_desc = _search_text(desc, MANUFACTURER_KEYWORDS)

    has_brand = bool(brand)
    has_mfg_spec = bool(mfg_spec_keys)
    has_address_hint = bool(address_in_specs or address_in_desc)
    has_mfg_in_desc = bool(mfg_in_desc)

    sources = []
    evidence_parts = []

    if has_brand:
        sources.append("BRAND_FIELD")
        evidence_parts.append(f'Brand "{brand}" present')

    if has_mfg_spec:
        sources.append("SPECIFICATION")
        for k, v in mfg_spec_values.items():
            evidence_parts.append(f'Spec "{k}": "{v}"')

    if has_mfg_in_desc:
        sources.append("DESCRIPTION")
        evidence_parts.append(f"Manufacturer keywords in description: {mfg_in_desc}")

    # Decision logic
    if (has_mfg_spec or has_mfg_in_desc) and has_address_hint:
        label = "PRESENT"
        reasoning = "Manufacturer/packer info AND address-like information found."
    elif has_brand or has_mfg_spec or has_mfg_in_desc:
        label = "PARTIAL"
        reasoning = "Brand or manufacturer name found, but complete address is likely missing from listing data."
        if not sources:
            sources.append("BRAND_FIELD" if has_brand else "NONE")
    else:
        label = "ABSENT"
        reasoning = "No brand, manufacturer, packer, or importer information found."
        sources.append("NONE")
        evidence_parts.append("No manufacturer/brand information in listing")

    return {
        "label": label,
        "evidence_sources": sources,
        "evidence_text": "; ".join(evidence_parts) if evidence_parts else "",
        "reasoning": reasoning,
    }


def draft_d02(product: dict) -> dict:
    """D02: Common/Generic Name of Commodity."""
    name = str(product.get("product_name", "")).strip()
    category_tree = str(product.get("product_category_tree", "")).strip()
    desc = str(product.get("description", ""))
    specs = product.get("parsed_specs", {})

    sources = []
    evidence_parts = []

    # Check product name for generic commodity words
    name_lower = name.lower()
    name_matches = [w for w in GENERIC_COMMODITY_WORDS if w in name_lower]

    # Check type in specs
    type_from_spec = specs.get("Type", "") or specs.get("type", "")

    # Check category tree
    cat_parts = category_tree.replace('"', '').split(" >> ") if category_tree else []
    # The deeper category levels often contain the generic name
    generic_from_cat = cat_parts[-1].strip() if len(cat_parts) > 1 else ""

    if name_matches:
        sources.append("LISTING_TITLE")
        evidence_parts.append(f'Generic name in title: "{", ".join(name_matches)}"')

    if type_from_spec:
        sources.append("SPECIFICATION")
        evidence_parts.append(f'Type from specs: "{type_from_spec}"')

    if generic_from_cat:
        evidence_parts.append(f'Category: "{generic_from_cat}"')

    # Decision
    if name_matches or type_from_spec:
        label = "PRESENT"
        reasoning = "Clear generic/common commodity name identified."
    elif generic_from_cat or name:
        label = "PRESENT"
        reasoning = "Product name provides identifiable commodity type."
        if not sources:
            sources.append("LISTING_TITLE")
            evidence_parts.append(f'Product name: "{name[:80]}"')
    else:
        label = "ABSENT"
        reasoning = "No meaningful generic commodity name found."
        sources.append("NONE")

    return {
        "label": label,
        "evidence_sources": sources,
        "evidence_text": "; ".join(evidence_parts) if evidence_parts else "",
        "reasoning": reasoning,
    }


def draft_d03(product: dict) -> dict:
    """D03: Net Quantity (Weight/Volume/Number)."""
    name = str(product.get("product_name", ""))
    specs = product.get("parsed_specs", {})
    desc = str(product.get("description", ""))

    sources = []
    evidence_parts = []

    # Check specs for quantity-related keys
    qty_spec_keys = [k for k in specs if any(
        w in k.lower() for w in
        ["weight", "volume", "quantity", "capacity", "pack of",
         "number of contents", "net quantity", "size", "dimension"]
    )]
    qty_spec_values = {k: specs[k] for k in qty_spec_keys}

    # Check product name for quantity patterns
    name_quantities = _extract_quantities_from_text(name)

    # Check description
    desc_quantities = _extract_quantities_from_text(desc[:500])

    if qty_spec_keys:
        sources.append("SPECIFICATION")
        for k, v in qty_spec_values.items():
            evidence_parts.append(f'Spec "{k}": "{v}"')

    if name_quantities:
        sources.append("LISTING_TITLE")
        evidence_parts.append(f"Quantity in title: {', '.join(name_quantities[:3])}")

    if desc_quantities and not name_quantities:
        sources.append("DESCRIPTION")
        evidence_parts.append(f"Quantity in description: {', '.join(desc_quantities[:3])}")

    # Decision
    has_spec_qty = bool(qty_spec_keys)
    has_text_qty = bool(name_quantities or desc_quantities)

    if has_spec_qty or has_text_qty:
        label = "PRESENT"
        reasoning = "Net quantity information found in listing."
    else:
        label = "ABSENT"
        reasoning = "No weight, volume, pack count, or quantity information found."
        sources.append("NONE")
        evidence_parts.append("No quantity data found in title, specs, or description")

    return {
        "label": label,
        "evidence_sources": sources,
        "evidence_text": "; ".join(evidence_parts) if evidence_parts else "",
        "reasoning": reasoning,
    }


def draft_d04(product: dict) -> dict:
    """D04: Month & Year of Manufacture/Packing/Import.

    Almost never available in e-commerce listings.
    """
    specs = product.get("parsed_specs", {})
    desc = str(product.get("description", ""))

    date_spec_keys = [k for k in specs if any(
        w in k.lower() for w in
        ["date", "manufacture", "manufacturing", "expiry",
         "best before", "use by", "shelf life", "packing date"]
    )]

    date_in_desc = _search_text(desc, [
        "date of manufacture", "mfg date", "manufacturing date",
        "date of packing", "packed on", "expiry date", "best before",
    ])

    sources = []
    evidence_parts = []

    if date_spec_keys:
        sources.append("SPECIFICATION")
        for k in date_spec_keys:
            evidence_parts.append(f'Spec "{k}": "{specs[k]}"')
        label = "PRESENT"
        reasoning = "Date-related information found in specifications."
    elif date_in_desc:
        sources.append("DESCRIPTION")
        evidence_parts.append(f"Date reference in description: {date_in_desc}")
        label = "PARTIAL"
        reasoning = "Date reference found in description but may not be complete month/year."
    else:
        label = "CANNOT_DETERMINE"
        reasoning = "No manufacture, packing, or import date available in listing evidence. This is expected for e-commerce listings."
        sources.append("NONE")
        evidence_parts.append("No date information in listing (expected)")

    return {
        "label": label,
        "evidence_sources": sources,
        "evidence_text": "; ".join(evidence_parts) if evidence_parts else "",
        "reasoning": reasoning,
    }


def draft_d05(product: dict) -> dict:
    """D05: MRP (Maximum Retail Price).

    Per annotation policy: treat the structured retail_price field
    as the dataset's available MRP/price evidence.
    """
    retail_price = product.get("retail_price")
    discounted_price = product.get("discounted_price")

    sources = []
    evidence_parts = []

    if retail_price is not None:
        label = "PRESENT"
        sources.append("PRICE_FIELD")
        evidence_parts.append(f"Retail price: ₹{retail_price}")
        if discounted_price is not None and discounted_price != retail_price:
            evidence_parts.append(f"Discounted price: ₹{discounted_price}")
        reasoning = "Retail price available in listing data."
        # Add the policy note as required
        evidence_parts.append(
            "Note: Based on annotation policy for current dataset — "
            "retail_price field treated as available MRP evidence, "
            "not an independent legal determination."
        )
    else:
        label = "ABSENT"
        sources.append("NONE")
        evidence_parts.append("No retail price available")
        reasoning = "No retail price field available."

    return {
        "label": label,
        "evidence_sources": sources,
        "evidence_text": "; ".join(evidence_parts) if evidence_parts else "",
        "reasoning": reasoning,
    }


def draft_d06(product: dict) -> dict:
    """D06: Consumer Care Contact Details."""
    specs = product.get("parsed_specs", {})
    desc = str(product.get("description", ""))
    extras = product.get("extra_spec_values", [])
    all_text = f"{desc} {' '.join(str(v) for v in extras)}"

    sources = []
    evidence_parts = []

    # Check specs
    care_spec_keys = [k for k in specs if any(
        w in k.lower() for w in
        ["customer care", "consumer care", "helpline", "toll free",
         "contact", "support", "complaint", "grievance", "email"]
    )]

    # Check description with keywords
    care_text_matches = _search_text(all_text, CONSUMER_CARE_KEYWORDS[:8])
    # Check with regex patterns (phone numbers)
    care_phone_matches = _search_text(all_text, CONSUMER_CARE_KEYWORDS[8:], use_regex=True)

    if care_spec_keys:
        sources.append("SPECIFICATION")
        for k in care_spec_keys:
            evidence_parts.append(f'Spec "{k}": "{specs[k]}"')

    if care_text_matches:
        sources.append("DESCRIPTION")
        evidence_parts.append(f"Consumer care keywords: {care_text_matches[:3]}")

    if care_phone_matches:
        if "DESCRIPTION" not in sources:
            sources.append("DESCRIPTION")
        evidence_parts.append(f"Phone pattern found: {care_phone_matches[:2]}")

    # Decision
    if care_spec_keys or care_text_matches or care_phone_matches:
        if care_spec_keys:
            label = "PRESENT"
            reasoning = "Consumer care information found in specifications."
        else:
            label = "PARTIAL"
            reasoning = "Some consumer care reference found but may be incomplete."
    else:
        label = "ABSENT"
        reasoning = "No consumer care, helpline, or complaint contact details found in listing."
        sources.append("NONE")
        evidence_parts.append("No consumer care details in listing (common for e-commerce)")

    return {
        "label": label,
        "evidence_sources": sources,
        "evidence_text": "; ".join(evidence_parts) if evidence_parts else "",
        "reasoning": reasoning,
    }


def draft_d07(product: dict) -> dict:
    """D07: Country of Origin."""
    specs = product.get("parsed_specs", {})
    desc = str(product.get("description", ""))

    sources = []
    evidence_parts = []

    # Check specs for country of origin
    origin_spec_keys = [k for k in specs if any(
        w in k.lower() for w in ORIGIN_SPEC_KEYWORDS
    )]
    origin_values = {k: specs[k] for k in origin_spec_keys}

    # Check description
    origin_in_desc = _search_text(desc, [
        "country of origin", "made in", "manufactured in",
        "imported from", "product of",
    ])

    if origin_spec_keys:
        sources.append("SPECIFICATION")
        for k, v in origin_values.items():
            evidence_parts.append(f'Spec "{k}": "{v}"')

    if origin_in_desc:
        sources.append("DESCRIPTION")
        evidence_parts.append(f"Origin reference in description: {origin_in_desc}")

    # Decision
    if origin_spec_keys:
        label = "PRESENT"
        reasoning = f"Country of origin explicitly stated: {list(origin_values.values())}"
    elif origin_in_desc:
        label = "PRESENT"
        reasoning = "Country/origin information found in description."
    else:
        label = "CANNOT_DETERMINE"
        reasoning = (
            "Country of origin not stated in listing. "
            "Cannot determine from available evidence whether product is domestic or imported."
        )
        sources.append("NONE")
        evidence_parts.append("No country of origin information in listing")

    return {
        "label": label,
        "evidence_sources": sources,
        "evidence_text": "; ".join(evidence_parts) if evidence_parts else "",
        "reasoning": reasoning,
    }


# -------------------------------------------------------------------------
# Overall Assessment
# -------------------------------------------------------------------------

def draft_overall(declarations: dict) -> dict:
    """Generate draft overall assessment from per-declaration drafts."""
    labels = [d["label"] for d in declarations.values()]

    has_absent = "ABSENT" in labels
    has_partial = "PARTIAL" in labels
    has_cannot = "CANNOT_DETERMINE" in labels
    all_present_or_na = all(l in ("PRESENT", "NOT_APPLICABLE") for l in labels)

    if all_present_or_na:
        overall_label = "COMPLIANT"
        reasoning = "All applicable declarations have evidence present."
    elif has_absent:
        overall_label = "NON_COMPLIANT"
        absent_decls = [did for did, d in declarations.items() if d["label"] == "ABSENT"]
        reasoning = f"Declaration(s) {', '.join(absent_decls)} are ABSENT."
    else:
        overall_label = "UNCERTAIN"
        uncertain_decls = [
            did for did, d in declarations.items()
            if d["label"] in ("PARTIAL", "CANNOT_DETERMINE")
        ]
        reasoning = f"Declaration(s) {', '.join(uncertain_decls)} are uncertain."

    # Confidence based on evidence quality
    present_count = labels.count("PRESENT")
    cannot_count = labels.count("CANNOT_DETERMINE")

    if cannot_count >= 3:
        confidence = "LOW"
    elif cannot_count >= 1 or has_partial:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    return {
        "label": overall_label,
        "notes": f"[MACHINE DRAFT] {reasoning}",
        "confidence": confidence,
        "reasoning": reasoning,
    }


# -------------------------------------------------------------------------
# Main Draft Generator
# -------------------------------------------------------------------------

def generate_draft(product: dict) -> dict:
    """
    Generate a complete draft annotation for a product.

    Args:
        product: Parsed product record (from parsed_annotation_candidates.jsonl)

    Returns:
        {
            "draft_generated": True,
            "draft_timestamp": "ISO timestamp",
            "declarations": {
                "D01": {"label": ..., "evidence_sources": [...], "evidence_text": ..., "reasoning": ...},
                ...
            },
            "overall_assessment": {"label": ..., "notes": ..., "confidence": ..., "reasoning": ...}
        }
    """
    declarations = {
        "D01": draft_d01(product),
        "D02": draft_d02(product),
        "D03": draft_d03(product),
        "D04": draft_d04(product),
        "D05": draft_d05(product),
        "D06": draft_d06(product),
        "D07": draft_d07(product),
    }

    overall = draft_overall(declarations)

    return {
        "draft_generated": True,
        "draft_timestamp": datetime.now(timezone.utc).isoformat(),
        "declarations": declarations,
        "overall_assessment": overall,
    }


# -------------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------------

def main():
    """Test the draft generator on pilot products."""
    PILOT_DATA_PATH = "data/interim/parsed_annotation_candidates.jsonl"
    MANIFEST_PATH = "annotations/pilot_manifest.csv"

    if not os.path.exists(PILOT_DATA_PATH):
        print(f"ERROR: {PILOT_DATA_PATH} not found")
        sys.exit(1)

    import pandas as pd

    manifest = pd.read_csv(MANIFEST_PATH)
    pilot_ids = set(manifest["product_id"].astype(str).tolist())

    products = []
    with open(PILOT_DATA_PATH) as f:
        for line in f:
            record = json.loads(line)
            if str(record.get("product_id", "")) in pilot_ids:
                products.append(record)

    # Test on first 5 products
    test_products = products[:5]
    print(f"Testing draft generator on {len(test_products)} products\n")

    for i, product in enumerate(test_products):
        print(f"{'='*70}")
        print(f"Product {i+1}: {product.get('product_name', 'N/A')[:60]}")
        print(f"  Brand: {product.get('brand', 'N/A')}")
        print(f"  Price: {product.get('retail_price', 'N/A')}")
        print(f"  Specs: {product.get('spec_key_count', 0)} keys")
        print()

        draft = generate_draft(product)

        for decl_id, decl in draft["declarations"].items():
            print(f"  {decl_id}: {decl['label']}")
            print(f"    Sources: {decl['evidence_sources']}")
            print(f"    Reasoning: {decl['reasoning']}")
            if decl['evidence_text']:
                print(f"    Evidence: {decl['evidence_text'][:100]}")
            print()

        overall = draft["overall_assessment"]
        print(f"  OVERALL: {overall['label']} (confidence: {overall['confidence']})")
        print(f"    Reasoning: {overall['reasoning']}")
        print()

    print(f"Draft generation test complete. All {len(test_products)} products processed successfully.")


if __name__ == "__main__":
    main()
