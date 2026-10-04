"""
detect_declaration_evidence.py
===============================
Independent legal declaration evidence detector for declarations D01 through D07.
Processes structured multi-source product evidence and applies deterministic legal
patterns to classify evidence nature into:
- EXPLICIT_EVIDENCE
- WEAK_EVIDENCE
- NO_EVIDENCE
- CONFLICTING_EVIDENCE

Inputs:
    Evidence records from extract_product_evidence.py
    configs/weak_label_rules.yaml

Outputs:
    Structured per-declaration evidence evaluation dictionary.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import re
from typing import Dict, Any, List, Tuple


# Regex patterns for quantity units
QUANTITY_UNIT_PATTERNS = [
    r'\b\d+(?:\.\d+)?\s*(?:ml|l|litre|liter|litres|liters)\b',
    r'\b\d+(?:\.\d+)?\s*(?:g|gm|gms|gram|grams|kg|kgs)\b',
    r'\b\d+(?:\.\d+)?\s*(?:oz|fl\.?\s*oz)\b',
    r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|meter|meters|inch|inches)\b',
    r'\bpack\s*(?:of\s*)?\d+\b',
    r'\bset\s*(?:of\s*)?\d+\b',
    r'\b\d+\s*(?:pc|pcs|piece|pieces|count|units?|nos?)\b',
    r'\b\d+\s*x\s*\d+\s*(?:cm|mm|m|inch)?\b',
]

# Generic commodity keywords across diverse retail categories
GENERIC_COMMODITY_TERMS = {
    "cream", "lotion", "gel", "serum", "oil", "shampoo", "conditioner", "soap",
    "wash", "cleanser", "moisturizer", "moisturiser", "spray", "deodorant", "perfume",
    "fragrance", "cologne", "lipstick", "lip balm", "mascara", "eyeliner", "foundation",
    "powder", "blush", "nail polish", "diaper", "wipe", "bottle", "pacifier", "romper",
    "pan", "pot", "cooker", "kettle", "plate", "bowl", "glass", "mug", "cup", "spoon",
    "fork", "knife", "mixer", "grinder", "blender", "toaster", "box", "set", "kit",
    "combo", "brush", "comb", "mask", "scrub", "top", "dress", "shirt", "shorts",
    "t-shirt", "trousers", "jeans", "shoes", "sandals", "slippers", "curtain", "sheet",
    "towel", "pillow", "cushion", "cover", "mat", "carpet", "toy", "game", "doll",
    "pen", "pencil", "notebook", "diary", "wallet", "belt", "bag", "backpack", "watch"
}

# Country names for conflict & origin detection
KNOWN_COUNTRIES = {
    "india", "china", "usa", "united states", "uk", "united kingdom", "germany",
    "japan", "korea", "france", "italy", "thailand", "vietnam", "taiwan", "malaysia",
    "indonesia", "bangladesh", "sri lanka", "switzerland", "uae", "spain"
}


def search_keywords(text: str, keywords: List[str], use_regex: bool = False) -> List[str]:
    """Search for keyword occurrences in lowercased text."""
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


def detect_d01_evidence(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """D01: Manufacturer/Packer/Importer Name & Complete Address."""
    brand = record.get("brand", "").strip()
    specs = record.get("specifications", {})
    desc = record.get("description", "")
    d01_cfg = rules.get("declarations", {}).get("D01", {})

    spec_keywords = d01_cfg.get("spec_keywords", ["manufacturer", "packer", "importer", "marketed by"])
    address_keywords = d01_cfg.get("address_keywords", ["address", "pin code", "road", "street", "city"])

    # Look for manufacturer in specs
    mfg_specs = {k: v for k, v in specs.items() if any(w in k.lower() for w in spec_keywords)}
    addr_specs = {k: v for k, v in specs.items() if any(w in k.lower() for w in address_keywords)}

    # Look in description
    mfg_in_desc = search_keywords(desc, spec_keywords)
    addr_in_desc = search_keywords(desc, address_keywords)

    sources = []
    evidence_text = []
    rule_ids = []

    if mfg_specs:
        sources.append("PRODUCT_SPECIFICATIONS")
        for k, v in mfg_specs.items():
            evidence_text.append(f'Spec "{k}": "{v}"')
        rule_ids.append("D01_SPEC_MFG_FOUND")

    if addr_specs:
        if "PRODUCT_SPECIFICATIONS" not in sources:
            sources.append("PRODUCT_SPECIFICATIONS")
        for k, v in addr_specs.items():
            evidence_text.append(f'Spec Address "{k}": "{v}"')
        rule_ids.append("D01_SPEC_ADDR_FOUND")

    if mfg_in_desc:
        sources.append("PRODUCT_DESCRIPTION")
        evidence_text.append(f"Description mfg keywords: {mfg_in_desc[:2]}")
        rule_ids.append("D01_DESC_MFG_FOUND")

    if addr_in_desc:
        if "PRODUCT_DESCRIPTION" not in sources:
            sources.append("PRODUCT_DESCRIPTION")
        evidence_text.append(f"Description address keywords: {addr_in_desc[:3]}")
        rule_ids.append("D01_DESC_ADDR_FOUND")

    if brand:
        sources.append("BRAND_FIELD")
        evidence_text.append(f'Brand "{brand}" present')
        rule_ids.append("D01_BRAND_FIELD_FOUND")

    has_mfg = bool(mfg_specs or mfg_in_desc)
    has_addr = bool(addr_specs or addr_in_desc)

    if (has_mfg or brand) and has_addr:
        return {
            "declaration": "D01",
            "label": "PRESENT",
            "confidence": d01_cfg.get("confidence_policy", {}).get("mfg_and_address", 0.90),
            "evidence_nature": "EXPLICIT_EVIDENCE",
            "evidence_sources": sources,
            "evidence_text": evidence_text,
            "rule_ids": rule_ids or ["D01_EXPLICIT_MFG_ADDRESS"]
        }
    elif has_mfg or brand:
        return {
            "declaration": "D01",
            "label": "PARTIAL",
            "confidence": d01_cfg.get("confidence_policy", {}).get("mfg_only_or_brand", 0.70),
            "evidence_nature": "WEAK_EVIDENCE",
            "evidence_sources": sources,
            "evidence_text": evidence_text,
            "rule_ids": rule_ids or ["D01_PARTIAL_BRAND_OR_MFG"]
        }
    else:
        return {
            "declaration": "D01",
            "label": "ABSENT",
            "confidence": d01_cfg.get("confidence_policy", {}).get("no_mfg_info", 0.85),
            "evidence_nature": "NO_EVIDENCE",
            "evidence_sources": ["NO_EVIDENCE"],
            "evidence_text": ["No manufacturer, packer, importer or brand identified in listing evidence"],
            "rule_ids": ["D01_NO_MFG_EVIDENCE"]
        }


def detect_d02_evidence(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """D02: Common / Generic Commodity Name."""
    title = record.get("title", "")
    cat_hierarchy = record.get("category_hierarchy", [])
    specs = record.get("specifications", {})
    d02_cfg = rules.get("declarations", {}).get("D02", {})

    title_words = set(re.findall(r'\b\w+\b', title.lower()))
    generic_in_title = list(title_words.intersection(GENERIC_COMMODITY_TERMS))

    type_in_spec = ""
    for k, v in specs.items():
        if k.lower() in ("type", "product type", "ideal for", "generic name"):
            type_in_spec = f"{k}: {v}"
            break

    sources = []
    evidence_text = []
    rule_ids = []

    if generic_in_title:
        sources.append("PRODUCT_TITLE")
        evidence_text.append(f'Generic terms in title: "{", ".join(generic_in_title[:3])}"')
        rule_ids.append("D02_GENERIC_IN_TITLE")

    if type_in_spec:
        sources.append("PRODUCT_SPECIFICATIONS")
        evidence_text.append(f'Spec "{type_in_spec}"')
        rule_ids.append("D02_SPEC_TYPE_FOUND")

    if cat_hierarchy:
        leaf = cat_hierarchy[-1]
        evidence_text.append(f'Category tree: "{leaf}"')
        if "PRODUCT_TITLE" not in sources:
            sources.append("PRODUCT_CATEGORY")
        rule_ids.append("D02_CATEGORY_LEAF")

    if generic_in_title or type_in_spec:
        return {
            "declaration": "D02",
            "label": "PRESENT",
            "confidence": d02_cfg.get("confidence_policy", {}).get("explicit_generic_in_title", 0.95),
            "evidence_nature": "EXPLICIT_EVIDENCE",
            "evidence_sources": sources or ["PRODUCT_TITLE"],
            "evidence_text": evidence_text,
            "rule_ids": rule_ids or ["D02_EXPLICIT_COMMODITY_NAME"]
        }
    elif title:
        return {
            "declaration": "D02",
            "label": "PRESENT",
            "confidence": d02_cfg.get("confidence_policy", {}).get("category_leaf_name", 0.80),
            "evidence_nature": "WEAK_EVIDENCE",
            "evidence_sources": ["PRODUCT_TITLE"],
            "evidence_text": [f'Product title provides commodity indication: "{title[:60]}"'],
            "rule_ids": ["D02_TITLE_IMPLIED_NAME"]
        }
    else:
        return {
            "declaration": "D02",
            "label": "ABSENT",
            "confidence": 0.85,
            "evidence_nature": "NO_EVIDENCE",
            "evidence_sources": ["NO_EVIDENCE"],
            "evidence_text": ["Empty product title and no commodity identification"],
            "rule_ids": ["D02_NO_COMMODITY_NAME"]
        }


def detect_d03_evidence(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """D03: Net Quantity."""
    title = record.get("title", "")
    desc = record.get("description", "")
    specs = record.get("specifications", {})
    d03_cfg = rules.get("declarations", {}).get("D03", {})

    qty_spec_keys = [k for k in specs if any(w in k.lower() for w in [
        "net quantity", "net weight", "weight", "volume", "capacity", "pack of",
        "number of contents", "size", "quantity"
    ])]

    title_quantities = []
    for pat in QUANTITY_UNIT_PATTERNS:
        matches = re.findall(pat, title, re.IGNORECASE)
        if matches:
            title_quantities.extend(matches)

    desc_quantities = []
    for pat in QUANTITY_UNIT_PATTERNS:
        matches = re.findall(pat, desc[:500], re.IGNORECASE)
        if matches:
            desc_quantities.extend(matches)

    sources = []
    evidence_text = []
    rule_ids = []

    if qty_spec_keys:
        sources.append("PRODUCT_SPECIFICATIONS")
        for k in qty_spec_keys[:3]:
            evidence_text.append(f'Spec "{k}": "{specs[k]}"')
        rule_ids.append("D03_SPEC_QUANTITY_FOUND")

    if title_quantities:
        sources.append("PRODUCT_TITLE")
        evidence_text.append(f'Title quantity: "{", ".join(title_quantities[:2])}"')
        rule_ids.append("D03_TITLE_QUANTITY_FOUND")

    if desc_quantities and not title_quantities:
        sources.append("PRODUCT_DESCRIPTION")
        evidence_text.append(f'Description quantity: "{", ".join(desc_quantities[:2])}"')
        rule_ids.append("D03_DESC_QUANTITY_FOUND")

    if qty_spec_keys or title_quantities or desc_quantities:
        return {
            "declaration": "D03",
            "label": "PRESENT",
            "confidence": d03_cfg.get("confidence_policy", {}).get("spec_with_standard_unit", 0.95),
            "evidence_nature": "EXPLICIT_EVIDENCE",
            "evidence_sources": sources,
            "evidence_text": evidence_text,
            "rule_ids": rule_ids or ["D03_EXPLICIT_NET_QUANTITY"]
        }
    else:
        return {
            "declaration": "D03",
            "label": "ABSENT",
            "confidence": d03_cfg.get("confidence_policy", {}).get("no_quantity_found", 0.85),
            "evidence_nature": "NO_EVIDENCE",
            "evidence_sources": ["NO_EVIDENCE"],
            "evidence_text": ["No weight, volume, or pack quantity found in title, specs, or description"],
            "rule_ids": ["D03_NO_QUANTITY_EVIDENCE"]
        }


def detect_d04_evidence(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """D04: Month & Year of Manufacture/Packing/Import."""
    specs = record.get("specifications", {})
    desc = record.get("description", "")
    d04_cfg = rules.get("declarations", {}).get("D04", {})

    date_spec_keys = [k for k in specs if any(w in k.lower() for w in [
        "date of manufacture", "mfg date", "manufacturing date", "packing date",
        "packed on", "expiry date", "best before", "shelf life"
    ])]

    date_in_desc = search_keywords(desc, [
        "date of manufacture", "mfg date", "manufacturing date",
        "date of packing", "packed on", "expiry date", "best before"
    ])

    if date_spec_keys:
        evidence_text = [f'Spec "{k}": "{specs[k]}"' for k in date_spec_keys]
        return {
            "declaration": "D04",
            "label": "PRESENT",
            "confidence": d04_cfg.get("confidence_policy", {}).get("spec_explicit_date", 0.85),
            "evidence_nature": "EXPLICIT_EVIDENCE",
            "evidence_sources": ["PRODUCT_SPECIFICATIONS"],
            "evidence_text": evidence_text,
            "rule_ids": ["D04_SPEC_DATE_FOUND"]
        }
    elif date_in_desc:
        return {
            "declaration": "D04",
            "label": "PARTIAL",
            "confidence": d04_cfg.get("confidence_policy", {}).get("desc_mention_date", 0.60),
            "evidence_nature": "WEAK_EVIDENCE",
            "evidence_sources": ["PRODUCT_DESCRIPTION"],
            "evidence_text": [f"Description date mention: {date_in_desc}"],
            "rule_ids": ["D04_DESC_DATE_FOUND"]
        }
    else:
        return {
            "declaration": "D04",
            "label": "CANNOT_DETERMINE",
            "confidence": d04_cfg.get("confidence_policy", {}).get("missing_in_listing", 0.90),
            "evidence_nature": "NO_EVIDENCE",
            "evidence_sources": ["NO_EVIDENCE"],
            "evidence_text": ["Month and year of manufacture/packing not stated in listing (standard for e-commerce)"],
            "rule_ids": ["D04_NO_DATE_IN_LISTING"]
        }


def detect_d05_evidence(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """D05: Maximum Retail Price (MRP)."""
    retail_price = record.get("retail_price")
    discounted_price = record.get("discounted_price")
    d05_cfg = rules.get("declarations", {}).get("D05", {})

    if retail_price is not None and retail_price > 0:
        evidence = [f"Retail price: ₹{retail_price:.2f}"]
        if discounted_price is not None:
            evidence.append(f"Discounted price: ₹{discounted_price:.2f}")
        evidence.append("Note: Standard listing price field evaluated as MRP per NitiMark annotation policy")
        return {
            "declaration": "D05",
            "label": "PRESENT",
            "confidence": d05_cfg.get("confidence_policy", {}).get("structured_retail_price", 0.95),
            "evidence_nature": "EXPLICIT_EVIDENCE",
            "evidence_sources": ["PRICE_FIELD"],
            "evidence_text": evidence,
            "rule_ids": ["D05_PRICE_FIELD_PRESENT"]
        }
    else:
        return {
            "declaration": "D05",
            "label": "ABSENT",
            "confidence": d05_cfg.get("confidence_policy", {}).get("no_price_info", 0.90),
            "evidence_nature": "NO_EVIDENCE",
            "evidence_sources": ["NO_EVIDENCE"],
            "evidence_text": ["No retail price or MRP information present"],
            "rule_ids": ["D05_NO_PRICE_FOUND"]
        }


def detect_d06_evidence(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """D06: Consumer Care / Complaint Contact Details."""
    specs = record.get("specifications", {})
    desc = record.get("description", "")
    d06_cfg = rules.get("declarations", {}).get("D06", {})

    care_spec_keys = [k for k in specs if any(w in k.lower() for w in [
        "customer care", "consumer care", "helpline", "toll free", "contact",
        "support", "grievance", "email"
    ])]

    care_text_matches = search_keywords(desc, [
        "customer care", "consumer care", "helpline", "toll free", "toll-free",
        "support@", "care@", "help@"
    ])

    phone_matches = re.findall(r'\b1800\s*\d{6,7}\b|\+91[\s-]*\d{10}\b', desc)

    if care_spec_keys:
        evidence = [f'Spec "{k}": "{specs[k]}"' for k in care_spec_keys]
        return {
            "declaration": "D06",
            "label": "PRESENT",
            "confidence": d06_cfg.get("confidence_policy", {}).get("spec_explicit_contact", 0.90),
            "evidence_nature": "EXPLICIT_EVIDENCE",
            "evidence_sources": ["PRODUCT_SPECIFICATIONS"],
            "evidence_text": evidence,
            "rule_ids": ["D06_SPEC_CARE_FOUND"]
        }
    elif care_text_matches or phone_matches:
        evidence = []
        if care_text_matches:
            evidence.append(f"Care terms: {care_text_matches}")
        if phone_matches:
            evidence.append(f"Helpline phone: {phone_matches}")
        return {
            "declaration": "D06",
            "label": "PARTIAL",
            "confidence": d06_cfg.get("confidence_policy", {}).get("phone_or_email_in_desc", 0.75),
            "evidence_nature": "WEAK_EVIDENCE",
            "evidence_sources": ["PRODUCT_DESCRIPTION"],
            "evidence_text": evidence,
            "rule_ids": ["D06_DESC_CARE_FOUND"]
        }
    else:
        return {
            "declaration": "D06",
            "label": "ABSENT",
            "confidence": d06_cfg.get("confidence_policy", {}).get("no_contact_info", 0.90),
            "evidence_nature": "NO_EVIDENCE",
            "evidence_sources": ["NO_EVIDENCE"],
            "evidence_text": ["No consumer care contact, toll-free number, or grievance email in listing"],
            "rule_ids": ["D06_NO_CARE_FOUND"]
        }


def detect_d07_evidence(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """D07: Country of Origin."""
    specs = record.get("specifications", {})
    desc = record.get("description", "")
    d07_cfg = rules.get("declarations", {}).get("D07", {})

    origin_spec_keys = [k for k in specs if any(w in k.lower() for w in [
        "country of origin", "country_of_origin", "countryoforigin", "origin", "made in"
    ])]

    spec_origin_vals = [str(specs[k]).strip() for k in origin_spec_keys]
    
    desc_origin_matches = search_keywords(desc, [
        "country of origin", "made in india", "made in china", "imported from", "product of"
    ])

    # Conflict check: Look for multiple different countries declared
    found_countries = set()
    all_origin_text = " ".join(spec_origin_vals + desc_origin_matches).lower()
    for c in KNOWN_COUNTRIES:
        if c in all_origin_text:
            found_countries.add(c)

    if len(found_countries) > 1:
        return {
            "declaration": "D07",
            "label": "CANNOT_DETERMINE",
            "confidence": 0.20,
            "evidence_nature": "CONFLICTING_EVIDENCE",
            "evidence_sources": ["PRODUCT_SPECIFICATIONS", "PRODUCT_DESCRIPTION"],
            "evidence_text": [f"Conflicting country mentions found: {list(found_countries)}"],
            "rule_ids": ["D07_CONFLICTING_ORIGIN_MENTIONS"]
        }

    if origin_spec_keys:
        evidence = [f'Spec "{k}": "{specs[k]}"' for k in origin_spec_keys]
        return {
            "declaration": "D07",
            "label": "PRESENT",
            "confidence": d07_cfg.get("confidence_policy", {}).get("spec_explicit_origin", 0.90),
            "evidence_nature": "EXPLICIT_EVIDENCE",
            "evidence_sources": ["PRODUCT_SPECIFICATIONS"],
            "evidence_text": evidence,
            "rule_ids": ["D07_SPEC_ORIGIN_FOUND"]
        }
    elif desc_origin_matches:
        return {
            "declaration": "D07",
            "label": "PRESENT",
            "confidence": d07_cfg.get("confidence_policy", {}).get("desc_explicit_origin", 0.75),
            "evidence_nature": "WEAK_EVIDENCE",
            "evidence_sources": ["PRODUCT_DESCRIPTION"],
            "evidence_text": [f"Origin reference: {desc_origin_matches}"],
            "rule_ids": ["D07_DESC_ORIGIN_FOUND"]
        }
    else:
        return {
            "declaration": "D07",
            "label": "CANNOT_DETERMINE",
            "confidence": d07_cfg.get("confidence_policy", {}).get("missing_in_listing", 0.85),
            "evidence_nature": "NO_EVIDENCE",
            "evidence_sources": ["NO_EVIDENCE"],
            "evidence_text": ["Country of origin not stated in listing; domestic vs imported status cannot be determined"],
            "rule_ids": ["D07_NO_ORIGIN_IN_LISTING"]
        }


def detect_all_declaration_evidence(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Detect evidence for all declarations D01 through D07."""
    return {
        "D01": detect_d01_evidence(record, rules),
        "D02": detect_d02_evidence(record, rules),
        "D03": detect_d03_evidence(record, rules),
        "D04": detect_d04_evidence(record, rules),
        "D05": detect_d05_evidence(record, rules),
        "D06": detect_d06_evidence(record, rules),
        "D07": detect_d07_evidence(record, rules),
    }
