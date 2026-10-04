# Model Input Provenance & End-to-End Transformation Trace
> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms  
> **Phase**: Corrective Phase — Declaration-Centric Product Compliance ML  
> **Date**: September 2026  

---

## 1. End-to-End Transformation Architecture

This document provides mandatory mathematical and empirical proof that the product-level ML model operates directly on **semantic declaration representations** extracted from physical packaging, rather than arbitrary text length heuristics or detector presence counts alone.

```
+-----------------------------------------------------------------------------------------------+
| 1. PHYSICAL PACKAGING IMAGE(S)                                                                |
|    - High-resolution photograph (Front / Back / Side panels)                                  |
+-----------------------------------------------+-----------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
| 2. VISUAL LOCALIZATION (YOLOv8)               | 3. OPTICAL CHARACTER RECOGNITION (OCR)        |
|    - 9 Visual Region Classes                  |    - Gemini 3.6 Flash / Mock                  |
|    - Spatial Bounding Boxes [x1, y1, x2, y2]  |    - Full Text String + Optical Clarity Score |
+-----------------------------------------------+-----------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
| 4. DECLARATION EXTRACTION (src/extraction/declaration_extractor.py)                           |
|    - Structured D01-D07 entity extraction preserving actual normalized values                 |
|    - Value Normalization: Quantity float, Unit, Month, Year, MRP float, Tax bool, Emails, etc.|
+-----------------------------------------------------------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
| 5. MULTI-IMAGE EVIDENCE FUSION (src/extraction/evidence_fusion.py)                            |
|    - Cross-view union (front + back + sides merged into unified product record)               |
|    - Conflict detection (e.g. conflicting MRPs flagged and preserved)                         |
+-----------------------------------------------------------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
| 6. DECLARATION FEATURE ENGINEERING (src/features/declaration_feature_extractor.py)            |
|    - Transformation into controlled numerical feature matrices (Sets A through E)             |
+-----------------------------------------------------------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
| 7. PRODUCT-LEVEL ML MODEL (src/models/declaration_product_model.py)                           |
|    - Supervised / Calibrated Classifier (LogisticRegression, RandomForest, XGBoost)          |
|    - Output: Probabilities [COMPLIANT, NON_COMPLIANT, UNCERTAIN]                              |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Concrete Sample Transformation Trace (Tata 1mg Creatine Packaging)

### Step 1: Physical Input
- **Packaging Text**:
  ```text
  TATA 1MG Healthcare Solutions Pvt. Ltd.
  Manufactured By: BACFO Pharmaceuticals (India) Limited, Plot C-46, Phase-2, Noida 201305, Uttar Pradesh.
  Product: Micronized Creatine Monohydrate Powder
  Net Quantity: 250 g
  Mfg Date: 02/2026 | Expiry: 07/2027
  MRP: Rs. 1063.00 (Inclusive of all taxes)
  For feedback: Email care@1mg.com or call +91-995-8932200
  Country of Origin: India
  ```
- **YOLO Localization**: Zero bounding boxes localized ($YOLO = 0$).

### Step 2: Declaration Extraction & Actual Value Normalization
```json
{
  "D01": {
    "status": "PRESENT",
    "raw_value": "BACFO Pharmaceuticals (India) Limited, Plot C-46, Phase-2, Noida 201305",
    "normalized_value": {
      "entity_name": "BACFO Pharmaceuticals (India) Limited",
      "address_markers": ["Noida", "201305", "Uttar Pradesh"],
      "pincode": "201305",
      "is_importer": false,
      "is_packer": false
    },
    "source": "OCR",
    "extraction_confidence": 0.95
  },
  "D02": {
    "status": "PRESENT",
    "raw_value": "Micronized Creatine Monohydrate Powder",
    "normalized_value": {
      "common_name": "Micronized Creatine Monohydrate Powder",
      "name_length": 38
    },
    "source": "OCR",
    "extraction_confidence": 0.95
  },
  "D03": {
    "status": "PRESENT",
    "raw_value": "250 g",
    "normalized_value": {
      "quantity": 250.0,
      "unit": "g",
      "unit_category": "WEIGHT"
    },
    "source": "OCR",
    "extraction_confidence": 0.98
  },
  "D04": {
    "status": "PRESENT",
    "raw_value": "02/2026",
    "normalized_value": {
      "month": 2,
      "year": 2026,
      "raw_date": "02/2026",
      "format_valid": true
    },
    "source": "OCR",
    "extraction_confidence": 0.96
  },
  "D05": {
    "status": "PRESENT",
    "raw_value": "MRP Rs. 1063.00",
    "normalized_value": {
      "mrp": 1063.0,
      "tax_inclusive": true,
      "currency": "INR"
    },
    "source": "OCR",
    "extraction_confidence": 0.98
  },
  "D06": {
    "status": "PRESENT",
    "raw_value": "Phone: +91-995-8932200, Email: care@1mg.com",
    "normalized_value": {
      "email": "care@1mg.com",
      "phone": "+91-995-8932200",
      "has_toll_free": false
    },
    "source": "OCR",
    "extraction_confidence": 0.97
  },
  "D07": {
    "status": "PRESENT",
    "raw_value": "India",
    "normalized_value": {
      "country": "India",
      "is_domestic": true
    },
    "source": "OCR",
    "extraction_confidence": 0.95
  }
}
```

### Step 3: Numerical Feature Extraction (Entering Feature Matrix $X$)
| Feature Column Name | Extracted Value in Feature Matrix $X$ | Derivation Source |
|:---|:---:|:---|
| `d01_present` | $1.0$ | D01 status == PRESENT |
| `d01_name_length` | $37.0$ | `len("BACFO Pharmaceuticals...")` |
| `d01_has_pincode` | $1.0$ | PIN code 201305 detected |
| `d02_present` | $1.0$ | D02 status == PRESENT |
| `d02_name_length` | $38.0$ | `len("Micronized Creatine...")` |
| `d03_present` | $1.0$ | D03 status == PRESENT |
| `d03_quantity_numeric_log` | $\ln(1 + 250) = 5.5255$ | Magnitude of Net Quantity ($250\text{ g}$) |
| `d03_unit_is_weight` | $1.0$ | Unit `g` $\in$ WEIGHT category |
| `d04_present` | $1.0$ | D04 status == PRESENT |
| `d04_manufacture_month` | $2.0$ | Month $02$ (February) |
| `d04_manufacture_year` | $6.0$ | Year $2026 - 2020 = 6$ |
| `d05_present` | $1.0$ | D05 status == PRESENT |
| `d05_mrp_numeric_log` | $\ln(1 + 1063) = 6.9698$ | Retail Price ($₹1063.00$) |
| `d05_tax_inclusive_present` | $1.0$ | "Inclusive of all taxes" detected |
| `d06_present` | $1.0$ | D06 status == PRESENT |
| `d06_phone_present` | $1.0$ | Helpline `+91-995-8932200` |
| `d06_email_present` | $1.0$ | Email `care@1mg.com` |
| `d07_present` | $1.0$ | D07 status == PRESENT |
| `d07_is_domestic` | $1.0$ | Origin country is India |
| `mandatory_declarations_present_count` | $6.0$ | All 6 mandatory declarations present |
| `all_mandatory_present` | $1.0$ | True |
| `yolo_total_detections` | $0.0$ | Supplementary visual feature |

### Step 4: Downstream Model Prediction
- **Feature Vector $x \in \mathbb{R}^{61}$** passed to product-level classifier.
- Model evaluates all semantic declaration signals, yielding:
  - **Verdict**: `COMPLIANT`
  - **Confidence**: $95.0\%$
  - **Uncertainty**: `LOW`
