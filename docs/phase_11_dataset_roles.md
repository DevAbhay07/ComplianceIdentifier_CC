# Packaging-Domain Dataset Schema & Structure Specification

**Project**: NitiMark-ML (SIH25057)  
**Milestone**: Phase 11C — Packaging-Domain Dataset Schema  
**Version**: 1.0.0

---

## 1. Overview
The packaging-domain dataset schema unifies multimodal evidence extracted from physical retail product packages. It bridges visual localization (YOLOv8 bounding boxes), optical character recognition (OCR text transcriptions), structured metadata, statutory declaration states (D01–D07), and the overall product-level compliance assessment.

---

## 2. Formal JSON Schema Definition

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PackagingDomainProductRecord",
  "type": "object",
  "required": [
    "product_id",
    "image_path",
    "source",
    "category",
    "ocr_text",
    "visual_regions",
    "declarations",
    "product_compliance",
    "evidence_quality",
    "ocr_quality",
    "image_quality",
    "label_source",
    "label_confidence",
    "annotation_version"
  ],
  "properties": {
    "product_id": {
      "type": "string",
      "description": "Unique identifier for the product record"
    },
    "image_path": {
      "type": "string",
      "description": "Relative path to the packaging image"
    },
    "source": {
      "type": "string",
      "enum": ["RETAIL_STORE_CAPTURE", "ROBOFLOW_PACKAGING", "FLIPKART_CATALOG", "SYNTHETIC_BENCHMARK"],
      "description": "Provenance origin of the packaging image"
    },
    "category": {
      "type": "string",
      "description": "Standardized commodity category (e.g. Beauty and Personal Care, Food & Beverage, Pet Supplies)"
    },
    "ocr_text": {
      "type": "string",
      "description": "Raw optical character recognition transcription text extracted from the packaging visual"
    },
    "visual_regions": {
      "type": "array",
      "description": "Localized packaging bounding boxes extracted via YOLOv8",
      "items": {
        "type": "object",
        "required": ["class", "bbox", "confidence"],
        "properties": {
          "class": {
            "type": "string",
            "enum": ["brand", "address", "weight", "mrp details", "consumer", "origin", "details", "tax", "barcode"]
          },
          "bbox": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x1, y1, x2, y2] normalized bounding box coordinates (0.0 to 1.0)"
          },
          "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "cropped_ocr_text": {
            "type": "string",
            "description": "Optional OCR transcription extracted specifically within this bounding box ROI"
          }
        }
      }
    },
    "declarations": {
      "type": "object",
      "required": ["D01", "D02", "D03", "D04", "D05", "D06", "D07"],
      "properties": {
        "D01": {
          "type": "string",
          "enum": ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"],
          "description": "Manufacturer / Packer / Importer Name and Address"
        },
        "D02": {
          "type": "string",
          "enum": ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"],
          "description": "Generic / Common Name of Commodity"
        },
        "D03": {
          "type": "string",
          "enum": ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"],
          "description": "Net Quantity in Standard Units (g, kg, ml, l, pcs)"
        },
        "D04": {
          "type": "string",
          "enum": ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"],
          "description": "Month & Year of Manufacture / Packaging / Import"
        },
        "D05": {
          "type": "string",
          "enum": ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"],
          "description": "Maximum Retail Price (MRP) incl. of all taxes & Unit Sale Price"
        },
        "D06": {
          "type": "string",
          "enum": ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"],
          "description": "Consumer Care / Grievance Helpline, Email, Address"
        },
        "D07": {
          "type": "string",
          "enum": ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"],
          "description": "Country of Origin (for imported/domestic goods)"
        }
      }
    },
    "product_compliance": {
      "type": "string",
      "enum": ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"],
      "description": "Overall Legal Metrology product-level compliance assessment"
    },
    "evidence_quality": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Completeness score across applicable statutory declaration signals"
    },
    "ocr_quality": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "OCR transcription quality / confidence score"
    },
    "image_quality": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Image sharpness and resolution score"
    },
    "label_source": {
      "type": "string",
      "enum": ["HUMAN_VERIFIED", "SILVER_RULE_DERIVED", "YOLO_OCR_EXTRACTED", "BENCHMARK_GROUND_TRUTH"],
      "description": "Provenance methodology for product-level label"
    },
    "label_confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "annotation_version": {
      "type": "string",
      "description": "Version string of schema and annotation guidelines"
    }
  }
}
```
