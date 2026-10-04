# Declaration Extraction & Actual Value Preservation Report

## Executive Summary
The Declaration Extraction subsystem (`src/extraction/declaration_extractor.py`) implements the core semantic layer of the Legal Metrology compliance architecture.
It converts raw optical packaging text, specifications, and spatial visual regions into a standardized structured representation across statutory declarations **D01 through D07**.

---

## Architectural Principles & Status Semantics

### 1. Separation of Concerns
- **YOLO Visual Localization**: Provides spatial packaging region guidance (bounding boxes identifying where text zones occur).
- **Vision / OCR**: Reads verbatim packaging text from crops and the full image.
- **Declaration Extraction**: Analyzes transcribed text using rule-based parsing and entity normalizers to structure evidence into D01–D07.

### 2. Declaration Schema (D01–D07)
| Declaration ID | Statutory Rule | Declaration Name | Mandatory | Extracted Entities & Values |
|---|---|---|---|---|
| **D01** | Rule 6(1)(a) | Manufacturer / Packer / Importer Name & Address | Yes | `entity_name`, `address_markers`, `pincode`, `is_packer`, `is_importer`, `is_marketer` |
| **D02** | Rule 6(1)(b) | Generic or Common Commodity Name | Yes | `common_name`, `category`, `identity_confidence` |
| **D03** | Rule 6(1)(c) | Net Quantity in Metric Units | Yes | `quantity` (float), `unit` (g, kg, ml, l, pcs, tablets, etc.) |
| **D04** | Rule 6(1)(d) | Month & Year of Manufacture / Packing / Expiry | Yes | `month` (1-12), `year` (YYYY), `expiry_month`, `expiry_year` |
| **D05** | Rule 6(1)(da) | Maximum Retail Price (MRP incl. of taxes) | Yes | `mrp` (float), `tax_inclusive` (bool), `price_currency` |
| **D06** | Rule 6(1)(n) | Consumer Care / Helpline Contacts | Yes | `phone`, `email`, `address`, `toll_free` |
| **D07** | Rule 6(1)(ea) | Country of Origin | Conditional | `country`, `is_domestic` |

---

## Actual Value Preservation

The architecture **never reduces declarations to binary flags immediately**. Full entity details and optical provenance are preserved in each `DeclarationEvidence` container:

```json
{
  "declaration_id": "D05",
  "declaration_name": "Maximum Retail Price (MRP incl. of all taxes)",
  "status": "PRESENT",
  "raw_value": "MRP ₹1063.00 (Incl. of all taxes)",
  "normalized_value": {
    "mrp": 1063.0,
    "currency": "INR",
    "tax_inclusive": true
  },
  "source": "OCR",
  "source_image": "image_01",
  "source_region": "roi_mrp_01",
  "bounding_box": [0.15, 0.72, 0.45, 0.85],
  "ocr_confidence": 0.94,
  "extraction_confidence": 0.95,
  "evidence_text": "MRP extracted: ₹1063.00 (Inclusive of all taxes)"
}
```

---

## Critical Regression: YOLO=0 with Positive OCR (Tata 1mg Creatine)

A central requirement of the Corrective Phase is:
> **YOLO non-detection (YOLO=0) is NOT legal absence.**

When packaging visual detectors fail due to non-standard layouts, reflections, or packaging curves, valid OCR evidence must be preserved.

### Concrete Example:
- **Product**: Tata 1mg Micronized Creatine Monohydrate Powder (250g)
- **YOLO Detections**: 0 bounding boxes detected.
- **Raw OCR Output**:
  ```text
  Marketed by: TATA 1MG Healthcare Solutions Pvt. Ltd., B-12, Okhla Phase II, New Delhi 110020.
  Manufactured by: BACFO Pharmaceuticals (India) Limited, C-46, Phase II, Noida 201305.
  Product: Micronized Creatine Monohydrate Powder
  Net Quantity: 250 g
  Mfg Date: 02/2026 | Expiry: 07/2027
  MRP: Rs. 1063.00 (Inclusive of all taxes)
  Customer Care: care@1mg.com | +91-995-8932200
  Country of Origin: India
  ```
- **Extracted Structured Evidence**:
  - **D01**: `PRESENT` (`entity_name`: "TATA 1MG Healthcare Solutions Pvt. Ltd.")
  - **D02**: `PRESENT` (`common_name`: "Micronized Creatine Monohydrate Powder")
  - **D03**: `PRESENT` (`quantity`: 250.0, `unit`: "g")
  - **D04**: `PRESENT` (`month`: 2, `year`: 2026)
  - **D05**: `PRESENT` (`mrp`: 1063.0, `tax_inclusive`: true)
  - **D06**: `PRESENT` (`email`: "care@1mg.com", `phone`: "+91-995-8932200")
  - **D07**: `PRESENT` (`country`: "India")

**Outcome**: Rather than erroneously marking the product non-compliant due to YOLO failure, the pipeline correctly recovers all seven declarations and accurately predicts `COMPLIANT`.
