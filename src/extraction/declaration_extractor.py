"""
declaration_extractor.py
========================
Extracts structured Legal Metrology statutory declarations (D01-D07) from OCR text,
metadata, and optional YOLO visual packaging detections.
Preserves actual normalized values, dates, numbers, units, and bounding boxes.

CRITICAL ARCHITECTURAL RULE:
YOLO non-detection (YOLO=0) is NOT legal absence.
If OCR/Vision identifies valid declaration text, the declaration is PRESENT.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from src.extraction.declaration_schema import (
    DeclarationEvidence,
    DeclarationStatus,
    EvidenceSource,
    DECLARATION_DEFINITIONS
)
from src.extraction.normalization import (
    normalize_quantity_value,
    normalize_price_value,
    normalize_date_value,
    normalize_country_value,
    normalize_manufacturer_value,
    normalize_consumer_care_value,
    normalize_generic_name_value
)

# Robust regex extractors with group capture for actual values
ROLE_PATTERNS = [
    ("MARKETER", re.compile(r'(?:marketed\s*by|mkt\s*by|marketed\s*and\s*distributed\s*by|marketer)[\s:]*([^\n\r;]+)', re.IGNORECASE)),
    ("MANUFACTURER", re.compile(r'(?:manufactured\s*by|mfd\s*by|mfg\s*by|mfd\.?\s*by|manufacturer)[\s:]*([^\n\r;]+)', re.IGNORECASE)),
    ("PACKER", re.compile(r'(?:packed\s*by|pkd\s*by|pkg\s*by|packer)[\s:]*([^\n\r;]+)', re.IGNORECASE)),
    ("IMPORTER", re.compile(r'(?:imported\s*by|imp\s*by|importer)[\s:]*([^\n\r;]+)', re.IGNORECASE))
]

MFG_REGEX = re.compile(
    r'(?:manufactured by|mfd by|packed by|imported by|marketed by|mfg by|mfd\.? by|marketer|packer)[\s:]*([^\n\r;.]+)',
    re.IGNORECASE
)
ADDRESS_REGEX = re.compile(
    r'(?:plot no|street|road|mumbai|delhi|bengaluru|bangalore|chennai|kolkata|hyderabad|pune|ahmedabad|indore|noida|okhla|gujarat|maharashtra|karnataka|tamil nadu|haryana|uttarakhand|panchkula|haridwar|makali|pin[\s:-]*\d{6}|\b\d{6}\b)',
    re.IGNORECASE
)
PINCODE_REGEX = re.compile(r'\b(\d{6})\b')

# Explicit statutory Net Quantity regex (prioritized)
EXPLICIT_NET_QTY_REGEX = re.compile(
    r'(?:net\s*(?:qty|quantity|wt|weight|vol|volume|contents?|mass)|(?:net\s*contents?)|(?:pack\s*size)|(?:quantity|qty|net\s*wt)[\s:]+)[\s.:]*(\d+(?:\.\d+)?)[ \t]*([a-zA-Z.]+)',
    re.IGNORECASE
)

# General standalone quantity regex (fallback)
GENERAL_QTY_REGEX = re.compile(
    r'(\b\d+(?:\.\d+)?)[ \t]*(ml|l|litre|liter|litres|g|gm|gms|grams?|kg|kgs|kilo(?:gram)?s?|oz|pcs|pc|units?|capsules?|tablets?|pack|set|wipes?|bags?|sachets?|count)\b',
    re.IGNORECASE
)

# Distractor prefixes to exclude from incidental fallback matches (e.g. nutritional tables, serving size)
DISTRACTOR_PREFIX = re.compile(
    r'(?:serving\s*size|per\s*serving|per\s*100\s*g|per\s*100\s*ml|energy|protein|carbohydrates?|fat|sugar|sodium|calcium|potassium|caffeine)[\s:]*$',
    re.IGNORECASE
)

DATE_REGEX = re.compile(
    r'(?:(?:date|mfg|mfd|packed|use by|best before|expiry|exp|pkd|manufacturing|mfg\s*dt|mfd\s*dt)[\s.:]*(?:date|dt[\s.:]*)?[\s.:]*)?(\b0?[1-9]|1[0-2])[/-](\d{4}|\d{2})\b',
    re.IGNORECASE
)

DATE_MONTH_YEAR_REGEX = re.compile(
    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})\b',
    re.IGNORECASE
)

# Explicit statutory MRP regex (prioritized over incidental prices)
EXPLICIT_MRP_REGEX = re.compile(
    r'(?:(?:mrp|m\.r\.p\.|max(?:imum)?\s*retail\s*price)[\s.:]*(?:\(?\s*incl\.?(?:usive)?\s*(?:of)?\s*(?:all)?\s*taxes\s*\)?)?[\s.:]*(?:rs\.?|inr|₹)?)\s*([\d,]+(?:\.\d{1,2})?)',
    re.IGNORECASE
)

# General currency fallback regex
GENERAL_PRICE_REGEX = re.compile(
    r'(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)',
    re.IGNORECASE
)

# Distractor prefixes to exclude from price matching (e.g. unit sale price, per g, per serving)
PRICE_DISTRACTOR_PREFIX = re.compile(
    r'(?:unit\s*sale\s*price|usp|per\s*g|per\s*gm|per\s*ml|per\s*piece|per\s*unit|per\s*serving)[\s.:/]*$',
    re.IGNORECASE
)

TAX_INCLUSIVE_REGEX = re.compile(
    r'(?:incl(?:usive)?\s*(?:of)?\s*(?:all)?\s*taxes|inclusive of taxes|incl of taxes|incl\. taxes|incl\. of all taxes)',
    re.IGNORECASE
)

EMAIL_REGEX = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', re.IGNORECASE)
PHONE_REGEX = re.compile(r'(\b1800\s*[-]?\d{2,4}\s*[-]?\d{2,4}\b|\+\d{1,3}[-\s]?\d{3,4}[-\s]?\d{6,7}|\b0\d{2,4}[-\s]?\d{6,8}\b|\b\d{10}\b)', re.IGNORECASE)

CARE_HEADER_REGEX = re.compile(
    r'(?:customer\s*care|consumer\s*care|helpline|support|feedback|complaints|consumer\s*cell)[\s.:]*(?:no\.?|phone|email|contact)?[\s.:]*([^\n\r]+)',
    re.IGNORECASE
)

ORIGIN_REGEX = re.compile(
    r'(?:country of origin|made in|origin|country)[\s.:]*([a-zA-Z\s]+)',
    re.IGNORECASE
)

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}


class DeclarationExtractor:
    """
    Extracts structured D01-D07 declarations preserving actual values,
    normalizations, optical qualities, and spatial bounding boxes.
    """

    def extract_from_text_and_vision(
        self,
        raw_ocr_text: str = "",
        title: str = "",
        description: str = "",
        specifications: Optional[Dict[str, Any]] = None,
        visual_regions: Optional[List[Dict[str, Any]]] = None,
        image_name: str = "image_01",
        ocr_confidence: float = 0.90
    ) -> Dict[str, DeclarationEvidence]:
        """
        Parses OCR text, metadata, and YOLO bounding boxes into structured D01-D07 DeclarationEvidence objects.
        """
        specs = specifications or {}
        spec_text = " ".join([f"{k}: {v}" for k, v in specs.items()])
        combined_text = f"{title}\n{description}\n{spec_text}\n{raw_ocr_text}".strip()
        
        visual_boxes = visual_regions or []
        yolo_classes_present = {reg.get("class", reg.get("class_name", "")).lower(): reg for reg in visual_boxes}

        declarations: Dict[str, DeclarationEvidence] = {}

        # -------------------------------------------------------------
        # D01: Manufacturer / Packer / Importer Name & Address
        # -------------------------------------------------------------
        role_entities: Dict[str, Dict[str, Any]] = {}
        for role_name, pat in ROLE_PATTERNS:
            for m in pat.finditer(combined_text):
                ent_name = m.group(1).strip()
                if len(ent_name) >= 3:
                    # Look ahead up to 250 chars for associated address/pincode
                    snippet = combined_text[m.end():min(len(combined_text), m.end() + 250)]
                    pins = PINCODE_REGEX.findall(snippet)
                    addrs = ADDRESS_REGEX.findall(snippet)
                    role_entities[role_name] = {
                        "name": ent_name,
                        "address_markers": addrs[:3] if addrs else [],
                        "pincode": pins[0] if pins else ""
                    }

        d01_mfg = MFG_REGEX.findall(combined_text)
        d01_addr = ADDRESS_REGEX.findall(combined_text)
        d01_pin = PINCODE_REGEX.findall(combined_text)
        d01_yolo = yolo_classes_present.get("address") or yolo_classes_present.get("brand")

        has_d01_text = bool(role_entities) or len(d01_mfg) > 0 or len(d01_addr) > 0

        if has_d01_text:
            entity_raw = d01_mfg[0].strip() if d01_mfg else ""
            pin_raw = d01_pin[0] if d01_pin else ""
            norm_d01 = normalize_manufacturer_value(
                entity_name=entity_raw,
                address_markers=d01_addr[:3] if d01_addr else [],
                pincode=pin_raw,
                is_packer=any(k in combined_text.lower() for k in ["packed by", "packer"]),
                is_importer=any(k in combined_text.lower() for k in ["imported by", "importer"]),
                is_marketer=any(k in combined_text.lower() for k in ["marketed by", "marketer"]),
                role_entities=role_entities if role_entities else None
            )
            evidence_str = f"Entity: {norm_d01['formatted']}"
            declarations["D01"] = DeclarationEvidence(
                declaration_id="D01",
                declaration_name=DECLARATION_DEFINITIONS["D01"]["name"],
                status=DeclarationStatus.PRESENT.value,
                raw_value=norm_d01["formatted"] if norm_d01["formatted"] != "Detected" else (d01_mfg[0].strip() if d01_mfg else "Manufacturer/Address"),
                normalized_value=norm_d01,
                source=EvidenceSource.COMBINED.value if d01_yolo else EvidenceSource.OCR.value,
                source_image=image_name,
                source_region=d01_yolo.get("class") if d01_yolo else None,
                bounding_box=d01_yolo.get("bbox") if d01_yolo else None,
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.96 if (norm_d01.get("roles") or (norm_d01["entity_name"] and norm_d01["address_markers"])) else 0.88,
                evidence_text=evidence_str
            )
        elif d01_yolo:
            declarations["D01"] = DeclarationEvidence(
                declaration_id="D01",
                declaration_name=DECLARATION_DEFINITIONS["D01"]["name"],
                status=DeclarationStatus.CANNOT_DETERMINE.value,
                raw_value="",
                normalized_value={"has_visual_panel": True},
                source=EvidenceSource.VISUAL_YOLO.value,
                source_image=image_name,
                source_region=d01_yolo.get("class"),
                bounding_box=d01_yolo.get("bbox"),
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.50,
                evidence_text="Visual address/brand panel localized by YOLO, but address text could not be resolved from image."
            )
        else:
            declarations["D01"] = self._create_absent_evidence("D01", image_name, ocr_confidence, raw_ocr_text)

        # -------------------------------------------------------------
        # D02: Generic or Common Name of Commodity
        # -------------------------------------------------------------
        prod_matches = re.findall(r'(?:product|commodity|generic\s*name|item|common\s*name)[\s:]*([^\n\r;.]+)', combined_text, re.IGNORECASE)
        if title.strip():
            d02_text = title.strip()
        elif prod_matches:
            d02_text = prod_matches[0].strip()
        elif raw_ocr_text.strip():
            first_line = raw_ocr_text.strip().split("\n")[0][:80]
            d02_text = first_line if len(first_line) >= 3 else ""
        else:
            d02_text = ""

        d02_yolo = yolo_classes_present.get("details") or yolo_classes_present.get("brand")
        has_d02_text = bool(d02_text and len(d02_text) >= 3)

        if has_d02_text:
            norm_d02 = normalize_generic_name_value(d02_text)
            declarations["D02"] = DeclarationEvidence(
                declaration_id="D02",
                declaration_name=DECLARATION_DEFINITIONS["D02"]["name"],
                status=DeclarationStatus.PRESENT.value,
                raw_value=d02_text,
                normalized_value=norm_d02,
                source=EvidenceSource.COMBINED.value if d02_yolo else (EvidenceSource.OCR.value if raw_ocr_text else EvidenceSource.METADATA.value),
                source_image=image_name,
                source_region=d02_yolo.get("class") if d02_yolo else None,
                bounding_box=d02_yolo.get("bbox") if d02_yolo else None,
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.95,
                evidence_text=f"Product Identity: {d02_text}"
            )
        elif d02_yolo:
            declarations["D02"] = DeclarationEvidence(
                declaration_id="D02",
                declaration_name=DECLARATION_DEFINITIONS["D02"]["name"],
                status=DeclarationStatus.CANNOT_DETERMINE.value,
                raw_value="",
                normalized_value={"has_visual_panel": True},
                source=EvidenceSource.VISUAL_YOLO.value,
                source_image=image_name,
                source_region=d02_yolo.get("class"),
                bounding_box=d02_yolo.get("bbox"),
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.50,
                evidence_text="Visual details/brand region detected by YOLO, but generic commodity name could not be resolved from text."
            )
        else:
            declarations["D02"] = self._create_absent_evidence("D02", image_name, ocr_confidence, raw_ocr_text)

        # -------------------------------------------------------------
        # D03: Net Quantity
        # -------------------------------------------------------------
        d03_yolo = yolo_classes_present.get("weight")
        
        # 1. Prioritize explicit statutory Net Quantity markers
        d03_matches = EXPLICIT_NET_QTY_REGEX.findall(combined_text)
        
        # 2. Fallback to general quantity regex if no explicit match, filtering out distractors (serving size, per 100g)
        if not d03_matches:
            for m in GENERAL_QTY_REGEX.finditer(combined_text):
                preceding = combined_text[max(0, m.start() - 35):m.start()]
                if not DISTRACTOR_PREFIX.search(preceding):
                    d03_matches = [(m.group(1), m.group(2))]
                    break

        # 3. Check metadata / specifications
        if not d03_matches and specifications:
            spec_qty_val = specifications.get("Net Quantity") or specifications.get("Quantity") or specifications.get("Weight") or specifications.get("Net Content")
            if spec_qty_val:
                sm = GENERAL_QTY_REGEX.findall(str(spec_qty_val))
                if sm:
                    d03_matches = sm

        has_d03_text = len(d03_matches) > 0

        if has_d03_text:
            qty_val_str, unit_str = d03_matches[0]
            norm_d03 = normalize_quantity_value(qty_val_str, unit_str)
            declarations["D03"] = DeclarationEvidence(
                declaration_id="D03",
                declaration_name=DECLARATION_DEFINITIONS["D03"]["name"],
                status=DeclarationStatus.PRESENT.value,
                raw_value=f"{qty_val_str} {unit_str}",
                normalized_value=norm_d03,
                source=EvidenceSource.COMBINED.value if d03_yolo else EvidenceSource.OCR.value,
                source_image=image_name,
                source_region=d03_yolo.get("class") if d03_yolo else None,
                bounding_box=d03_yolo.get("bbox") if d03_yolo else None,
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.98,
                evidence_text=f"Net Quantity: {norm_d03['canonical_display']} ({norm_d03['unit_category']})"
            )
        elif d03_yolo:
            declarations["D03"] = DeclarationEvidence(
                declaration_id="D03",
                declaration_name=DECLARATION_DEFINITIONS["D03"]["name"],
                status=DeclarationStatus.CANNOT_DETERMINE.value,
                raw_value="",
                normalized_value={"has_visual_panel": True},
                source=EvidenceSource.VISUAL_YOLO.value,
                source_image=image_name,
                source_region=d03_yolo.get("class"),
                bounding_box=d03_yolo.get("bbox"),
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.50,
                evidence_text="Visual weight/quantity panel detected by YOLO, but numerical net quantity value could not be resolved from OCR."
            )
        else:
            declarations["D03"] = self._create_absent_evidence("D03", image_name, ocr_confidence, raw_ocr_text)

        # -------------------------------------------------------------
        # D04: Date of Manufacture / Packing / Expiry
        # -------------------------------------------------------------
        d04_matches = DATE_REGEX.findall(combined_text)
        d04_month_matches = DATE_MONTH_YEAR_REGEX.findall(combined_text)
        d04_yolo = yolo_classes_present.get("details")
        has_d04_text = len(d04_matches) > 0 or len(d04_month_matches) > 0

        if has_d04_text:
            if d04_matches:
                m_str, y_str = d04_matches[0]
                m_int = int(m_str) if (1 <= int(m_str) <= 12) else 1
                y_int = int(y_str) if len(y_str) == 4 else (2000 + int(y_str) if len(y_str) == 2 else 2026)
                raw_dt = f"{m_str}/{y_str}"
            else:
                m_name, y_str = d04_month_matches[0]
                m_int = MONTH_MAP.get(m_name.lower()[:3], 1)
                y_int = int(y_str)
                raw_dt = f"{m_name} {y_str}"

            norm_d04 = normalize_date_value(m_int, y_int, raw_dt)
            declarations["D04"] = DeclarationEvidence(
                declaration_id="D04",
                declaration_name=DECLARATION_DEFINITIONS["D04"]["name"],
                status=DeclarationStatus.PRESENT.value,
                raw_value=raw_dt,
                normalized_value=norm_d04,
                source=EvidenceSource.COMBINED.value if d04_yolo else EvidenceSource.OCR.value,
                source_image=image_name,
                source_region=d04_yolo.get("class") if d04_yolo else None,
                bounding_box=d04_yolo.get("bbox") if d04_yolo else None,
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.96,
                evidence_text=f"Mfg/Packing Date: {norm_d04['formatted']}"
            )
        elif d04_yolo:
            declarations["D04"] = DeclarationEvidence(
                declaration_id="D04",
                declaration_name=DECLARATION_DEFINITIONS["D04"]["name"],
                status=DeclarationStatus.CANNOT_DETERMINE.value,
                raw_value="",
                normalized_value={"has_visual_panel": True},
                source=EvidenceSource.VISUAL_YOLO.value,
                source_image=image_name,
                source_region=d04_yolo.get("class"),
                bounding_box=d04_yolo.get("bbox"),
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.50,
                evidence_text="Visual details panel detected by YOLO, but date of manufacture/packing could not be resolved from OCR."
            )
        else:
            declarations["D04"] = self._create_absent_evidence("D04", image_name, ocr_confidence, raw_ocr_text)

        # -------------------------------------------------------------
        # D05: Maximum Retail Price (MRP) & Tax Inclusivity
        # -------------------------------------------------------------
        has_tax = bool(TAX_INCLUSIVE_REGEX.search(combined_text)) or bool(yolo_classes_present.get("tax"))
        d05_yolo = yolo_classes_present.get("mrp details") or yolo_classes_present.get("tax")

        # 1. Prioritize explicit statutory MRP regex
        d05_matches = [m.replace(",", "").strip() for m in EXPLICIT_MRP_REGEX.findall(combined_text) if m]

        # 2. Fallback to general currency regex with distractor filtering
        if not d05_matches:
            for m in GENERAL_PRICE_REGEX.finditer(combined_text):
                val_str = m.group(1).replace(",", "").strip()
                try:
                    f_val = float(val_str)
                except ValueError:
                    continue
                if f_val <= 0.0:
                    continue
                preceding = combined_text[max(0, m.start() - 35):m.start()]
                following = combined_text[m.end():min(len(combined_text), m.end() + 15)]
                if PRICE_DISTRACTOR_PREFIX.search(preceding) or re.search(r'^\s*[/]\s*(?:g|gm|ml|piece|unit|serving|kg)', following, re.IGNORECASE):
                    continue
                d05_matches = [val_str]
                break

        has_d05_text = len(d05_matches) > 0

        if has_d05_text:
            mrp_raw = str(d05_matches[0]).replace(",", "")
            norm_d05 = normalize_price_value(mrp_raw, tax_inclusive=has_tax)
            declarations["D05"] = DeclarationEvidence(
                declaration_id="D05",
                declaration_name=DECLARATION_DEFINITIONS["D05"]["name"],
                status=DeclarationStatus.PRESENT.value,
                raw_value=f"MRP Rs. {norm_d05['mrp']:.2f}",
                normalized_value=norm_d05,
                source=EvidenceSource.COMBINED.value if d05_yolo else EvidenceSource.OCR.value,
                source_image=image_name,
                source_region=d05_yolo.get("class") if d05_yolo else None,
                bounding_box=d05_yolo.get("bbox") if d05_yolo else None,
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.98,
                evidence_text=f"MRP: {norm_d05['formatted']} (Tax Inclusivity: {'YES' if has_tax else 'NOT EXPLICIT'})"
            )
        elif d05_yolo:
            declarations["D05"] = DeclarationEvidence(
                declaration_id="D05",
                declaration_name=DECLARATION_DEFINITIONS["D05"]["name"],
                status=DeclarationStatus.CANNOT_DETERMINE.value,
                raw_value="",
                normalized_value={"has_visual_panel": True},
                source=EvidenceSource.VISUAL_YOLO.value,
                source_image=image_name,
                source_region=d05_yolo.get("class"),
                bounding_box=d05_yolo.get("bbox"),
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.50,
                evidence_text="Visual MRP/tax panel detected by YOLO, but numerical MRP could not be resolved from OCR."
            )
        else:
            declarations["D05"] = self._create_absent_evidence("D05", image_name, ocr_confidence, raw_ocr_text)

        # -------------------------------------------------------------
        # D06: Consumer Care Details
        # -------------------------------------------------------------
        d06_emails = EMAIL_REGEX.findall(combined_text)
        d06_phones = PHONE_REGEX.findall(combined_text)
        d06_headers = CARE_HEADER_REGEX.findall(combined_text)
        d06_yolo = yolo_classes_present.get("consumer")
        has_d06_text = len(d06_emails) > 0 or len(d06_phones) > 0 or len(d06_headers) > 0

        if has_d06_text:
            email_val = d06_emails[0] if d06_emails else ""
            phone_val = d06_phones[0] if d06_phones else ""
            addr_val = d06_headers[0].strip() if d06_headers else ""
            norm_d06 = normalize_consumer_care_value(email=email_val, phone=phone_val, address=addr_val)
            declarations["D06"] = DeclarationEvidence(
                declaration_id="D06",
                declaration_name=DECLARATION_DEFINITIONS["D06"]["name"],
                status=DeclarationStatus.PRESENT.value,
                raw_value=norm_d06["formatted"] or "Consumer Care",
                normalized_value=norm_d06,
                source=EvidenceSource.COMBINED.value if d06_yolo else EvidenceSource.OCR.value,
                source_image=image_name,
                source_region=d06_yolo.get("class") if d06_yolo else None,
                bounding_box=d06_yolo.get("bbox") if d06_yolo else None,
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.97 if (d06_emails or d06_phones) else 0.85,
                evidence_text=f"Consumer Care: {norm_d06['formatted']}" if norm_d06['formatted'] else "Consumer Helpline present"
            )
        elif d06_yolo:
            declarations["D06"] = DeclarationEvidence(
                declaration_id="D06",
                declaration_name=DECLARATION_DEFINITIONS["D06"]["name"],
                status=DeclarationStatus.CANNOT_DETERMINE.value,
                raw_value="",
                normalized_value={"has_visual_panel": True},
                source=EvidenceSource.VISUAL_YOLO.value,
                source_image=image_name,
                source_region=d06_yolo.get("class"),
                bounding_box=d06_yolo.get("bbox"),
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.50,
                evidence_text="Visual consumer care region detected by YOLO, but contact details (phone/email) could not be resolved from OCR."
            )
        else:
            declarations["D06"] = self._create_absent_evidence("D06", image_name, ocr_confidence, raw_ocr_text)

        # -------------------------------------------------------------
        # D07: Country of Origin
        # -------------------------------------------------------------
        d07_matches = ORIGIN_REGEX.findall(combined_text)
        d07_yolo = yolo_classes_present.get("origin")
        has_d07_text = len(d07_matches) > 0

        if has_d07_text:
            c_name = d07_matches[0].strip()
            norm_d07 = normalize_country_value(c_name)
            declarations["D07"] = DeclarationEvidence(
                declaration_id="D07",
                declaration_name=DECLARATION_DEFINITIONS["D07"]["name"],
                status=DeclarationStatus.PRESENT.value,
                raw_value=c_name,
                normalized_value=norm_d07,
                source=EvidenceSource.COMBINED.value if d07_yolo else EvidenceSource.OCR.value,
                source_image=image_name,
                source_region=d07_yolo.get("class") if d07_yolo else None,
                bounding_box=d07_yolo.get("bbox") if d07_yolo else None,
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.95,
                evidence_text=f"Country of Origin: {norm_d07['country']}"
            )
        elif d07_yolo:
            declarations["D07"] = DeclarationEvidence(
                declaration_id="D07",
                declaration_name=DECLARATION_DEFINITIONS["D07"]["name"],
                status=DeclarationStatus.CANNOT_DETERMINE.value,
                raw_value="",
                normalized_value={"has_visual_panel": True},
                source=EvidenceSource.VISUAL_YOLO.value,
                source_image=image_name,
                source_region=d07_yolo.get("class"),
                bounding_box=d07_yolo.get("bbox"),
                ocr_confidence=ocr_confidence,
                extraction_confidence=0.50,
                evidence_text="Visual origin/address region detected by YOLO, but country of origin text could not be resolved."
            )
        else:
            declarations["D07"] = self._create_absent_evidence("D07", image_name, ocr_confidence, raw_ocr_text)

        return declarations

    def _create_absent_evidence(
        self,
        declaration_id: str,
        image_name: str,
        ocr_confidence: float,
        raw_ocr_text: str
    ) -> DeclarationEvidence:
        """
        Creates ABSENT or CANNOT_DETERMINE declaration evidence based on OCR quality.
        """
        if raw_ocr_text and len(raw_ocr_text.strip()) >= 20 and ocr_confidence >= 0.50:
            status = DeclarationStatus.ABSENT.value
            ev_text = f"Declaration '{DECLARATION_DEFINITIONS[declaration_id]['name']}' ({declaration_id}) not found in packaging text or visual regions."
            conf = 0.90
        else:
            status = DeclarationStatus.CANNOT_DETERMINE.value
            ev_text = f"Insufficient optical clarity / text to verify declaration '{DECLARATION_DEFINITIONS[declaration_id]['name']}' ({declaration_id})."
            conf = 0.50

        return DeclarationEvidence(
            declaration_id=declaration_id,
            declaration_name=DECLARATION_DEFINITIONS[declaration_id]["name"],
            status=status,
            raw_value="",
            normalized_value={},
            source=EvidenceSource.OCR.value,
            source_image=image_name,
            source_region=None,
            bounding_box=None,
            ocr_confidence=ocr_confidence,
            extraction_confidence=conf,
            evidence_text=ev_text
        )

    def extract_declarations(
        self,
        raw_text: str = "",
        product_id: str = "PROD_EXTRACT",
        source_image: str = "image_01",
        ocr_confidence: float = 0.90,
        visual_regions: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Convenience wrapper returning a structured ProductEvidenceRecord.
        """
        from src.extraction.declaration_schema import ProductEvidenceRecord
        decls = self.extract_from_text_and_vision(
            raw_ocr_text=raw_text,
            visual_regions=visual_regions,
            image_name=source_image,
            ocr_confidence=ocr_confidence
        )
        return ProductEvidenceRecord(
            product_id=product_id,
            image_count=1,
            source_images=[source_image],
            declarations=decls,
            overall_ocr_confidence=ocr_confidence,
            image_quality=ocr_confidence,
            raw_ocr_texts={source_image: raw_text},
            visual_detections={source_image: visual_regions or []}
        )
