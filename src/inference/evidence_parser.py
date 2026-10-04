"""
evidence_parser.py
==================
Extracts structured Legal Metrology declaration evidence (D01-D07) from raw OCR text
and textual product descriptions. Does NOT assign ML target labels or legal compliance verdicts.
"""

import re
from typing import Dict, Any, Optional

# Regex extractors for evidence spans
MFG_REGEX = re.compile(
    r'(?:manufactured by|mfd by|packed by|imported by|marketed by|mfg by|mfd\.? by)[\s:]*([^\n\r.]+)',
    re.IGNORECASE
)
ADDRESS_REGEX = re.compile(
    r'(?:plot no|street|road|mumbai|delhi|bengaluru|bangalore|chennai|kolkata|hyderabad|pune|ahmedabad|indore|gujarat|maharashtra|karnataka|tamil nadu|haryana|pin[\s:-]*\d{6}|\b\d{6}\b)',
    re.IGNORECASE
)
NET_QTY_REGEX = re.compile(
    r'(?:net\s*(?:qty|quantity|wt|weight|vol|volume|content)?[\s:]*)?(\b\d+(?:\.\d+)?\s*(?:ml|l|litre|liter|litres|g|gm|kg|oz|pcs|pc|units|capsules|tablets|pack|set|wipes|bags|count)\b)',
    re.IGNORECASE
)
DATE_REGEX = re.compile(
    r'(?:mfg|mfd|packed|use by|best before|expiry|exp|pkd|manufacturing|mfg\s*dt|mfd\s*dt)[\s.:]*(?:date|dt[\s.:]*)?[\s.:]*(\d{1,2}[/-]\d{2,4}|\d{2}[/-]\d{2}[/-]\d{2,4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\b|\d{1,2}\s*months)',
    re.IGNORECASE
)
MRP_REGEX = re.compile(
    r'(?:mrp|max(?:imum)?\s*retail\s*price|price)[\s.:]*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d{2})?)',
    re.IGNORECASE
)
CARE_REGEX = re.compile(
    r'(?:customer\s*care|consumer\s*care|helpline|support|feedback)[\s.:]*(?:no\.?|phone|email|contact)?[\s.:]*([^\n\r,]+)|(\b1800\s*[-]?\d{3,4}\s*[-]?\d{3,4}\b|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
    re.IGNORECASE
)
ORIGIN_REGEX = re.compile(
    r'(?:country of origin|made in|origin|country)[\s.:]*([a-zA-Z\s]+)',
    re.IGNORECASE
)


def parse_product_evidence(
    raw_ocr_text: str = "",
    title: str = "",
    description: str = "",
    specifications: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Parse OCR text and product metadata into a structured declaration evidence dictionary.
    """
    specs = specifications or {}
    spec_text = " ".join([f"{k}: {v}" for k, v in specs.items()])
    combined_raw = f"{title} {description} {spec_text} {raw_ocr_text}".strip()

    evidence = {}

    # D01: Manufacturer / Packer / Importer Name & Address
    d01_mfg = MFG_REGEX.findall(combined_raw)
    d01_addr = ADDRESS_REGEX.findall(combined_raw)
    d01_spans = []
    if d01_mfg:
        d01_spans.append(f"Entity: {d01_mfg[0].strip()}")
    if d01_addr:
        d01_spans.append(f"Address Markers: {', '.join(d01_addr[:3])}")
    
    evidence["D01"] = {
        "declaration_name": "Name & Address of Manufacturer / Packer / Importer",
        "evidence_detected": len(d01_spans) > 0,
        "evidence_text": " | ".join(d01_spans) if d01_spans else "No manufacturer name/address markers detected",
        "source": "OCR_AND_METADATA" if (raw_ocr_text and (title or description)) else ("OCR" if raw_ocr_text else "METADATA")
    }

    # D02: Generic / Common Product Name
    d02_present = bool(title.strip() or len(raw_ocr_text.strip()) > 5)
    evidence["D02"] = {
        "declaration_name": "Generic or Common Name of Commodity",
        "evidence_detected": d02_present,
        "evidence_text": title.strip() if title.strip() else raw_ocr_text.split("\n")[0][:80],
        "source": "METADATA_TITLE" if title.strip() else "OCR_HEADER"
    }

    # D03: Net Quantity
    d03_matches = NET_QTY_REGEX.findall(combined_raw)
    evidence["D03"] = {
        "declaration_name": "Net Quantity in Standard Units",
        "evidence_detected": len(d03_matches) > 0,
        "evidence_text": ", ".join(d03_matches[:3]) if d03_matches else "No net quantity units found",
        "source": "COMBINED_TEXT"
    }

    # D04: Date of Manufacture / Packing / Expiry
    d04_matches = DATE_REGEX.findall(combined_raw)
    evidence["D04"] = {
        "declaration_name": "Date of Manufacture / Packing / Expiry",
        "evidence_detected": len(d04_matches) > 0,
        "evidence_text": ", ".join(d04_matches[:2]) if d04_matches else "No date markers detected",
        "source": "COMBINED_TEXT"
    }

    # D05: Maximum Retail Price (MRP)
    d05_matches = MRP_REGEX.findall(combined_raw)
    evidence["D05"] = {
        "declaration_name": "Maximum Retail Price (MRP) & Tax Inclusivity",
        "evidence_detected": len(d05_matches) > 0,
        "evidence_text": f"MRP Rs. {d05_matches[0]}" if d05_matches else "No MRP text detected",
        "source": "COMBINED_TEXT"
    }

    # D06: Consumer Care Details
    d06_matches = [m[0] or m[1] for m in CARE_REGEX.findall(combined_raw) if m[0] or m[1]]
    evidence["D06"] = {
        "declaration_name": "Consumer Care Details (Phone / Email / Address)",
        "evidence_detected": len(d06_matches) > 0,
        "evidence_text": ", ".join(d06_matches[:2]) if d06_matches else "No customer care contacts detected",
        "source": "COMBINED_TEXT"
    }

    # D07: Country of Origin
    d07_matches = ORIGIN_REGEX.findall(combined_raw)
    evidence["D07"] = {
        "declaration_name": "Country of Origin",
        "evidence_detected": len(d07_matches) > 0,
        "evidence_text": d07_matches[0].strip() if d07_matches else "No country of origin statement detected",
        "source": "COMBINED_TEXT"
    }

    # Quality Diagnostics
    detected_count = sum(1 for d in evidence.values() if d["evidence_detected"])
    completeness = round(detected_count / 7.0, 2)

    diagnostics = {
        "extracted_text_length": len(combined_raw),
        "detected_declarations_count": detected_count,
        "completeness_score": completeness,
        "has_ocr_input": bool(raw_ocr_text.strip()),
        "has_text_metadata": bool(title.strip() or description.strip())
    }

    return {
        "declarations": evidence,
        "diagnostics": diagnostics
    }
