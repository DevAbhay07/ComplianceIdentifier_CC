"""
normalization.py
================
Canonical value normalization and conflict detection for Legal Metrology declarations (D01-D07).
Ensures that equivalent representations (e.g. '250 g', '250g', '250 G', '250 gm', '0.25 kg')
normalize to the same canonical representation, and distinguishes corroborating observations
from true irreconcilable conflicts.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Union


# =========================================================================
# 1. QUANTITY & UNIT NORMALIZATION (D03)
# =========================================================================

# Standard unit aliases mapping to (canonical_unit, category, conversion_factor_to_base)
# Base units:
#   WEIGHT -> grams (g)
#   VOLUME -> milliliters (ml)
#   COUNT  -> count (units)
UNIT_TABLE = {
    # Weight units (base = g)
    "g": ("g", "WEIGHT", 1.0),
    "gm": ("g", "WEIGHT", 1.0),
    "gms": ("g", "WEIGHT", 1.0),
    "gram": ("g", "WEIGHT", 1.0),
    "grams": ("g", "WEIGHT", 1.0),
    "g.": ("g", "WEIGHT", 1.0),
    "kg": ("kg", "WEIGHT", 1000.0),
    "kgs": ("kg", "WEIGHT", 1000.0),
    "kilo": ("kg", "WEIGHT", 1000.0),
    "kilogram": ("kg", "WEIGHT", 1000.0),
    "kilograms": ("kg", "WEIGHT", 1000.0),
    "k.g.": ("kg", "WEIGHT", 1000.0),
    "kg.": ("kg", "WEIGHT", 1000.0),
    "mg": ("mg", "WEIGHT", 0.001),
    "mgs": ("mg", "WEIGHT", 0.001),
    "milligram": ("mg", "WEIGHT", 0.001),
    "milligrams": ("mg", "WEIGHT", 0.001),
    "m.g.": ("mg", "WEIGHT", 0.001),
    "oz": ("oz", "WEIGHT", 28.3495),
    "ounce": ("oz", "WEIGHT", 28.3495),
    "ounces": ("oz", "WEIGHT", 28.3495),

    # Volume units (base = ml)
    "ml": ("ml", "VOLUME", 1.0),
    "m.l.": ("ml", "VOLUME", 1.0),
    "mls": ("ml", "VOLUME", 1.0),
    "millilitre": ("ml", "VOLUME", 1.0),
    "milliliter": ("ml", "VOLUME", 1.0),
    "millilitres": ("ml", "VOLUME", 1.0),
    "milliliters": ("ml", "VOLUME", 1.0),
    "l": ("l", "VOLUME", 1000.0),
    "l.": ("l", "VOLUME", 1000.0),
    "ltr": ("l", "VOLUME", 1000.0),
    "litre": ("l", "VOLUME", 1000.0),
    "liter": ("l", "VOLUME", 1000.0),
    "litres": ("l", "VOLUME", 1000.0),
    "liters": ("l", "VOLUME", 1000.0),
    "cl": ("cl", "VOLUME", 10.0),
    "fl oz": ("fl oz", "VOLUME", 29.5735),
    "fl. oz.": ("fl oz", "VOLUME", 29.5735),

    # Count / Numerical units (base = count)
    "pcs": ("pcs", "COUNT", 1.0),
    "pc": ("pcs", "COUNT", 1.0),
    "piece": ("pcs", "COUNT", 1.0),
    "pieces": ("pcs", "COUNT", 1.0),
    "units": ("units", "COUNT", 1.0),
    "unit": ("units", "COUNT", 1.0),
    "u": ("units", "COUNT", 1.0),
    "n": ("units", "COUNT", 1.0),
    "nos": ("units", "COUNT", 1.0),
    "no": ("units", "COUNT", 1.0),
    "tablets": ("tablets", "COUNT", 1.0),
    "tablet": ("tablets", "COUNT", 1.0),
    "tabs": ("tablets", "COUNT", 1.0),
    "capsules": ("capsules", "COUNT", 1.0),
    "capsule": ("capsules", "COUNT", 1.0),
    "caps": ("capsules", "COUNT", 1.0),
    "wipes": ("wipes", "COUNT", 1.0),
    "wipe": ("wipes", "COUNT", 1.0),
    "bags": ("bags", "COUNT", 1.0),
    "bag": ("bags", "COUNT", 1.0),
    "tea bags": ("tea bags", "COUNT", 1.0),
    "sachets": ("sachets", "COUNT", 1.0),
    "sachet": ("sachets", "COUNT", 1.0),
    "pack": ("pack", "COUNT", 1.0),
    "packs": ("pack", "COUNT", 1.0),
    "set": ("set", "COUNT", 1.0),
    "sets": ("set", "COUNT", 1.0),
    "count": ("count", "COUNT", 1.0)
}


def normalize_quantity_value(
    quantity: Union[str, float, int],
    unit: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalizes a numerical quantity and unit into a canonical representation.
    Handles unit equivalence (e.g. 0.25 kg -> 250 g, 0.15 L -> 150 ml, 250 gm -> 250 g).
    """
    # Parse string if passed as combined (e.g. "250 g" or "0.25kg")
    if isinstance(quantity, str):
        q_str = quantity.strip().lower()
        m = re.match(r'^([\d.,]+)\s*([a-zA-Z\s.]+)$', q_str)
        if m:
            qty_num_str, u_str = m.groups()
            try:
                qty_num = float(qty_num_str.replace(",", ""))
            except ValueError:
                qty_num = 0.0
            unit_raw = u_str.strip()
        else:
            try:
                qty_num = float(q_str.replace(",", ""))
                unit_raw = (unit or "").strip().lower()
            except ValueError:
                qty_num = 0.0
                unit_raw = (unit or "").strip().lower()
    else:
        qty_num = float(quantity) if quantity is not None else 0.0
        unit_raw = (unit or "").strip().lower()

    # Look up unit in mapping table
    unit_clean = unit_raw.rstrip(".")
    unit_info = UNIT_TABLE.get(unit_clean) or UNIT_TABLE.get(unit_raw)
    
    if unit_info:
        norm_unit, category, factor = unit_info
        base_value = round(qty_num * factor, 4)
    else:
        norm_unit = unit_clean or "units"
        category = "COUNT"
        base_value = round(qty_num, 4)

    # Compute human-readable canonical display and canonical key
    if category == "WEIGHT":
        # Standardize on base grams for comparison
        canonical_key = f"{base_value:g} g"
        # Display preference: if base_value >= 1000 and base_value % 1000 == 0 -> kg, else g
        if base_value >= 1000.0 and base_value % 1000.0 == 0:
            canonical_display = f"{base_value / 1000.0:g} kg"
        else:
            canonical_display = f"{base_value:g} g"
    elif category == "VOLUME":
        # Standardize on base ml for comparison
        canonical_key = f"{base_value:g} ml"
        if base_value >= 1000.0 and base_value % 1000.0 == 0:
            canonical_display = f"{base_value / 1000.0:g} L"
        else:
            canonical_display = f"{base_value:g} ml"
    else:
        canonical_key = f"{base_value:g} {norm_unit}"
        canonical_display = f"{base_value:g} {norm_unit}"

    return {
        "quantity": qty_num,
        "unit": norm_unit,
        "unit_category": category,
        "base_value": base_value,
        "canonical_key": canonical_key,
        "canonical_display": canonical_display,
        "formatted": canonical_display
    }


def are_quantities_conflicting(q1: Dict[str, Any], q2: Dict[str, Any]) -> bool:
    """
    Returns True if two normalized quantity dictionaries are genuinely conflicting.
    Returns False if they represent the same physical quantity (corroborating).
    """
    cat1 = q1.get("unit_category")
    cat2 = q2.get("unit_category")
    b1 = q1.get("base_value")
    b2 = q2.get("base_value")

    if b1 is None or b2 is None or b1 <= 0 or b2 <= 0:
        return False

    if cat1 != cat2:
        return True

    # Check relative difference (< 0.5% tolerance for floating conversion precision)
    diff = abs(b1 - b2)
    max_val = max(b1, b2, 1.0)
    if diff / max_val > 0.005 and diff > 0.05:
        return True

    return False


# =========================================================================
# 2. PRICE & MRP NORMALIZATION (D05)
# =========================================================================

def normalize_price_value(
    price: Union[str, float, int],
    tax_inclusive: bool = True
) -> Dict[str, Any]:
    """
    Normalizes a price value into standard INR currency float and formatted string.
    """
    if isinstance(price, str):
        p_clean = re.sub(r'[^\d.]', '', price.replace(",", ""))
        try:
            mrp_float = round(float(p_clean), 2)
        except ValueError:
            mrp_float = 0.0
    else:
        mrp_float = round(float(price), 2) if price is not None else 0.0

    canonical_key = f"₹{mrp_float:.2f}"
    return {
        "mrp": mrp_float,
        "currency": "INR",
        "tax_inclusive": tax_inclusive,
        "canonical_key": canonical_key,
        "formatted": canonical_key
    }


def are_prices_conflicting(p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
    """
    Returns True if two MRP values have a material conflict (> ₹1.0 difference).
    """
    m1 = p1.get("mrp")
    m2 = p2.get("mrp")
    if m1 is None or m2 is None or m1 <= 0 or m2 <= 0:
        return False
    return abs(m1 - m2) > 1.0


# =========================================================================
# 3. DATE OF MANUFACTURE / PACKING NORMALIZATION (D04)
# =========================================================================

def normalize_date_value(
    month: Union[str, int],
    year: Union[str, int],
    raw_date: str = ""
) -> Dict[str, Any]:
    """
    Normalizes month and year into canonical month (1-12) and 4-digit year.
    """
    try:
        m_int = int(month)
        if not (1 <= m_int <= 12):
            m_int = 1
    except (ValueError, TypeError):
        m_int = 1

    try:
        y_int = int(year)
        if y_int < 100:
            y_int = 2000 + y_int
    except (ValueError, TypeError):
        y_int = 2026

    canonical_key = f"{m_int:02d}/{y_int}"
    return {
        "month": m_int,
        "year": y_int,
        "raw_date": raw_date or canonical_key,
        "format_valid": True,
        "canonical_key": canonical_key,
        "formatted": canonical_key
    }


def are_dates_conflicting(d1: Dict[str, Any], d2: Dict[str, Any]) -> bool:
    """
    Returns True if two dates have different months or years.
    """
    m1, y1 = d1.get("month"), d1.get("year")
    m2, y2 = d2.get("month"), d2.get("year")
    if not m1 or not y1 or not m2 or not y2:
        return False
    return (m1 != m2) or (y1 != y2)


# =========================================================================
# 4. COUNTRY OF ORIGIN NORMALIZATION (D07)
# =========================================================================

COUNTRY_ALIASES = {
    "india": "india",
    "ind": "india",
    "republic of india": "india",
    "made in india": "india",
    "product of india": "india",
    "china": "china",
    "prc": "china",
    "p.r.c.": "china",
    "made in china": "china",
    "usa": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "germany": "germany",
    "japan": "japan",
    "uk": "uk",
    "united kingdom": "uk",
    "france": "france",
    "italy": "italy",
    "vietnam": "vietnam",
    "thailand": "thailand",
    "taiwan": "taiwan",
    "indonesia": "indonesia",
    "malaysia": "malaysia",
    "korea": "south korea",
    "south korea": "south korea"
}


def normalize_country_value(country_name: str) -> Dict[str, Any]:
    """
    Normalizes a country of origin string into canonical country name and domestic flag.
    """
    raw = (country_name or "").strip()
    clean = re.sub(r'[^a-zA-Z\s]', '', raw).lower().strip()
    canonical = COUNTRY_ALIASES.get(clean, clean)
    display = canonical.title() if canonical else "Unknown"

    return {
        "country": display,
        "canonical_country": canonical,
        "is_domestic": ("india" in canonical),
        "canonical_key": canonical,
        "formatted": display
    }


def are_countries_conflicting(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
    """
    Returns True if two country candidates specify completely different countries.
    """
    k1 = c1.get("canonical_country") or c1.get("canonical_key", "").lower()
    k2 = c2.get("canonical_country") or c2.get("canonical_key", "").lower()
    if not k1 or not k2 or k1 == "unknown" or k2 == "unknown":
        return False
    return k1 != k2


# =========================================================================
# 5. MANUFACTURER / PACKER / IMPORTER / MARKETER NORMALIZATION (D01)
# =========================================================================

CORPORATE_STOP_WORDS = {
    "ltd", "pvt", "limited", "private", "co", "company", "mfg", "mfd", "packed", "by",
    "india", "the", "and", "&", "of", "in", "solutions", "corp", "corporation", "inc",
    "incorporated", "enterprises", "products", "industries", "holdings", "group", "llp"
}

CORPORATE_SYNONYMS = {
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "corporation": "corp",
    "incorporated": "inc"
}


def extract_core_company_tokens(name: str) -> set:
    """
    Extracts normalized core identifying tokens of a business entity,
    filtering out generic legal entity and geographic stop words.
    """
    clean = re.sub(r'[^\w\s]', ' ', (name or "").lower())
    tokens = set(clean.split())
    syn_tokens = set(CORPORATE_SYNONYMS.get(t, t) for t in tokens)
    return syn_tokens - CORPORATE_STOP_WORDS


def normalize_manufacturer_value(
    entity_name: str = "",
    address_markers: Optional[List[str]] = None,
    pincode: str = "",
    is_packer: bool = False,
    is_importer: bool = False,
    is_marketer: bool = False,
    role_entities: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Normalizes manufacturer, marketer, packer, or importer details,
    preserving distinct business roles while keeping backward compatibility.
    """
    clean_name = re.sub(r'\s+', ' ', (entity_name or "")).strip()
    core_tokens = extract_core_company_tokens(clean_name)
    
    # Build structured role entities
    roles = {}
    entities = []

    if role_entities:
        for r_name, r_data in role_entities.items():
            r_clean = r_name.upper()
            ent_name = r_data.get("name") or r_data.get("entity_name") or ""
            ent_addr = r_data.get("address_markers") or r_data.get("address") or []
            if isinstance(ent_addr, str):
                ent_addr = [ent_addr] if ent_addr else []
            ent_pin = str(r_data.get("pincode") or "").strip()
            ent_obj = {
                "role": r_clean,
                "name": ent_name,
                "address_markers": ent_addr,
                "pincode": ent_pin,
                "canonical_tokens": list(extract_core_company_tokens(ent_name))
            }
            roles[r_clean.lower()] = ent_obj
            entities.append(ent_obj)
            if r_clean == "MARKETER":
                is_marketer = True
            elif r_clean == "PACKER":
                is_packer = True
            elif r_clean == "IMPORTER":
                is_importer = True
    else:
        # Default single entity
        primary_role = "MARKETER" if is_marketer else ("PACKER" if is_packer else ("IMPORTER" if is_importer else "MANUFACTURER"))
        if clean_name:
            ent_obj = {
                "role": primary_role,
                "name": clean_name,
                "address_markers": address_markers or [],
                "pincode": str(pincode or "").strip(),
                "canonical_tokens": list(core_tokens)
            }
            roles[primary_role.lower()] = ent_obj
            entities.append(ent_obj)

    # Format human-readable multi-role display
    formatted_parts = []
    if "manufacturer" in roles and roles["manufacturer"]["name"]:
        m = roles["manufacturer"]
        formatted_parts.append(f"Mfg: {m['name']}" + (f" (PIN: {m['pincode']})" if m['pincode'] else ""))
    if "marketer" in roles and roles["marketer"]["name"]:
        mk = roles["marketer"]
        formatted_parts.append(f"Mkt: {mk['name']}" + (f" (PIN: {mk['pincode']})" if mk['pincode'] else ""))
    if "packer" in roles and roles["packer"]["name"]:
        p = roles["packer"]
        formatted_parts.append(f"Packer: {p['name']}" + (f" (PIN: {p['pincode']})" if p['pincode'] else ""))
    if "importer" in roles and roles["importer"]["name"]:
        imp = roles["importer"]
        formatted_parts.append(f"Importer: {imp['name']}" + (f" (PIN: {imp['pincode']})" if imp['pincode'] else ""))

    if formatted_parts:
        formatted_str = " | ".join(formatted_parts)
    elif clean_name:
        formatted_str = f"{clean_name} (PIN: {pincode})" if pincode else clean_name
    else:
        formatted_str = "Detected"

    # Primary entity name fallback
    primary_name = clean_name
    if not primary_name and entities:
        primary_name = entities[0]["name"]

    return {
        "entity_name": primary_name,
        "address_markers": address_markers or [],
        "pincode": str(pincode or "").strip(),
        "is_packer": is_packer,
        "is_importer": is_importer,
        "is_marketer": is_marketer,
        "roles": roles,
        "entities": entities,
        "canonical_key": primary_name.lower(),
        "canonical_tokens": list(core_tokens),
        "formatted": formatted_str
    }


def are_manufacturers_conflicting(m1: Dict[str, Any], m2: Dict[str, Any]) -> bool:
    """
    Checks if two manufacturer evidence candidates specify irreconcilable entities.
    
    CRITICAL RULE:
    Having different entities for different business ROLES (e.g. Manufacturer A + Marketer B)
    is legitimate under Rule 6 and is NOT a conflict.
    
    A conflict occurs ONLY when there is contradictory evidence for the SAME ROLE
    without token overlap or explanatory context.
    """
    roles1 = m1.get("roles") or {}
    roles2 = m2.get("roles") or {}

    # If both have structured roles, compare role-by-role
    if roles1 or roles2:
        common_roles = set(roles1.keys()) & set(roles2.keys())
        if not common_roles:
            # Different roles across images (e.g. Image 1 has marketer, Image 2 has manufacturer) -> NO CONFLICT
            return False

        # If common roles exist, check for conflict within any shared role
        for r in common_roles:
            ent1 = roles1[r]
            ent2 = roles2[r]
            name1 = ent1.get("name", "")
            name2 = ent2.get("name", "")
            pin1 = str(ent1.get("pincode", "")).strip()
            pin2 = str(ent2.get("pincode", "")).strip()

            tok1 = extract_core_company_tokens(name1)
            tok2 = extract_core_company_tokens(name2)

            if not tok1 or not tok2:
                continue

            # If both have core tokens and token overlap is zero on the SAME role
            if not (tok1 & tok2):
                # Substring check
                if name1.lower() not in name2.lower() and name2.lower() not in name1.lower():
                    return True

        return False

    # Fallback for flat legacy representations without explicit roles dict
    is_mkt1 = m1.get("is_marketer", False)
    is_mkt2 = m2.get("is_marketer", False)
    is_pck1 = m1.get("is_packer", False)
    is_pck2 = m2.get("is_packer", False)
    is_imp1 = m1.get("is_importer", False)
    is_imp2 = m2.get("is_importer", False)

    # If one is marketer and other is pure manufacturer -> different roles -> not a conflict
    if (is_mkt1 != is_mkt2) or (is_pck1 != is_pck2) or (is_imp1 != is_imp2):
        return False

    name1 = m1.get("entity_name", "")
    name2 = m2.get("entity_name", "")
    tok1 = extract_core_company_tokens(name1)
    tok2 = extract_core_company_tokens(name2)
    pin1 = str(m1.get("pincode", "")).strip()
    pin2 = str(m2.get("pincode", "")).strip()

    if not tok1 or not tok2:
        return False

    if pin1 and pin2 and pin1 != pin2 and not (tok1 & tok2):
        return True

    if not (tok1 & tok2):
        if name1.lower() not in name2.lower() and name2.lower() not in name1.lower():
            return True

    return False


# =========================================================================
# 6. CONSUMER CARE DETAILS NORMALIZATION (D06)
# =========================================================================

def normalize_consumer_care_value(
    email: str = "",
    phone: str = "",
    address: str = ""
) -> Dict[str, Any]:
    """
    Normalizes consumer care phone number, email address, and grievance office.
    """
    clean_email = (email or "").strip().lower()
    clean_phone = re.sub(r'[^\d+]', '', str(phone or "")).strip()
    toll_free = bool("1800" in clean_phone)

    canonical_key = f"{clean_email}|{clean_phone}"
    return {
        "email": clean_email,
        "phone": clean_phone,
        "has_toll_free": toll_free,
        "address": address.strip(),
        "canonical_key": canonical_key,
        "formatted": f"Phone: {clean_phone}, Email: {clean_email}".strip(", ")
    }


def are_consumer_care_conflicting(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
    """
    Checks if two consumer care candidates have conflicting details.
    """
    e1, e2 = c1.get("email"), c2.get("email")
    p1, p2 = c1.get("phone"), c2.get("phone")

    # If both have emails and they differ -> conflict
    if e1 and e2 and e1 != e2:
        return True

    # If both have 10-digit phones and they differ -> conflict
    if p1 and p2 and len(p1) >= 8 and len(p2) >= 8 and p1 != p2:
        return True

    return False


# =========================================================================
# 7. GENERIC / COMMON NAME NORMALIZATION (D02)
# =========================================================================

def normalize_generic_name_value(common_name: str) -> Dict[str, Any]:
    """
    Normalizes common/generic product commodity name.
    """
    clean_name = re.sub(r'\s+', ' ', (common_name or "")).strip()
    return {
        "common_name": clean_name,
        "name_length": len(clean_name),
        "canonical_key": clean_name.lower(),
        "formatted": clean_name
    }


def are_generic_names_conflicting(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
    """
    Checks if generic commodity names are completely contradictory.
    """
    k1 = n1.get("canonical_key", "")
    k2 = n2.get("canonical_key", "")
    if not k1 or not k2:
        return False
    # If one is substring of other or token overlap exists -> not conflict
    if k1 in k2 or k2 in k1:
        return False
    tok1 = set(re.findall(r'[a-zA-Z0-9]+', k1))
    tok2 = set(re.findall(r'[a-zA-Z0-9]+', k2))
    if tok1 & tok2:
        return False
    return True
