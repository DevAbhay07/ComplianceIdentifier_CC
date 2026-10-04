# Inference Pipeline to Streamlit UI Mapping Specification
> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms  
> **Phase**: Phase 12 — Streamlit Testing App  

---

## 1. End-to-End Information Flow

This document defines the strict, transparent mapping from the frozen Phase 11 multimodal inference backend into the Streamlit presentation layer.

```
+-----------------------------------------------------------------------------------------+
|                                    USER INTERACTION                                     |
|  [ User Uploads Packaging Image / Selects Sample Image ] -> [ Format & Integrity Guard ] |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                  INFERENCE BRIDGE                                       |
|  1. YOLOv8 Visual Packaging Detection  --> 6 Class Bounding Boxes + Confidence          |
|  2. Visual ROI Cropping                --> High-Res Bounding Box Image Cutouts          |
|  3. Vision / OCR Extraction            --> Raw Text + Optical Clarity Score             |
|  4. Statutory Evidence Parser          --> Regex & Structural Entity Extraction (D01-D07)|
|  5. YOLO Spatial Feature Extractor     --> 29 Continuous Visual Features                |
|  6. Multimodal Feature Transformer     --> Set D (D=741) Multimodal Representation      |
|  7. Direct ML Compliance Classifier    --> Random Forest Set D Softmax Probabilities    |
|  8. Evidence-Aware Decision Engine     --> Primary Verdict (Rule 6 Statutory Logic)     |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                  UI PRESENTATION                                        |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | 1. Hero Verdict Card (app/components/result_card.py)                              |  |
|  |    - Source: bridge_result["overall_prediction"] -> COMPLIANT / NON_COMPLIANT /   |  |
|  |                     UNCERTAIN                                                     |  |
|  |    - Metric: bridge_result["confidence"] (0.0% - 100.0%)                          |  |
|  |    - Subtext: bridge_result["primary_reason"]                                     |  |
|  +-----------------------------------------------------------------------------------+  |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | 2. Statutory "Why" Explanation (app/components/why_explanation.py)                |  |
|  |    - Source: bridge_result["decision_basis"] + bridge_result["primary_reason"]    |  |
|  |    - Legal Framework: Rule 6 Legal Metrology (Packaged Commodities) Rules, 2011   |  |
|  +-----------------------------------------------------------------------------------+  |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | 3. Internal Statutory Evidence Table (app/components/evidence_panel.py)           |  |
|  |    - Source: bridge_result["declarations"]["D01"..."D07"]                         |  |
|  |    - Fields: Status Badge (PRESENT/ABSENT/CANNOT_DETERMINE), Extracted Snippet,   |  |
|  |              Confidence Score, Provenance Source (OCR / Visual / Metadata)        |  |
|  +-----------------------------------------------------------------------------------+  |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | 4. YOLOv8 Visual Detection Overlay (app/components/yolo_viewer.py)                |  |
|  |    - Source: bridge_result["yolo"]["detections"]                                  |  |
|  |    - Display: Rendered Image with Bounding Boxes & Confidence Table               |  |
|  +-----------------------------------------------------------------------------------+  |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | 5. Cropped Packaging ROI Gallery (app/components/roi_viewer.py)                   |  |
|  |    - Source: bridge_result["yolo"]["roi_crops"]                                   |  |
|  |    - Display: High-resolution visual crops for each localized declaration zone    |  |
|  +-----------------------------------------------------------------------------------+  |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | 6. Raw OCR & Clarity Viewer (app/components/ocr_viewer.py)                         |  |
|  |    - Source: bridge_result["ocr"]["raw_text"], ["confidence"], ["provider"]       |  |
|  |    - Display: Extracted string, character metrics, optical quality status        |  |
|  +-----------------------------------------------------------------------------------+  |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | 7. Developer Diagnostic Trace (app/components/technical_trace.py) [Checkbox]      |  |
|  |    - Source: bridge_result["yolo"]["features_29"] (29 YOLO Spatial Features)      |  |
|  |    - Source: bridge_result["direct_ml"]["prediction"] & ["probabilities"]         |  |
|  |    - Source: bridge_result["latency_ms"] (Per-Stage Latency Breakdown)            |  |
|  +-----------------------------------------------------------------------------------+  |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | 8. Sanitized Audit Export (app/components/export.py)                               |  |
|  |    - Action: Generates downloaded JSON file with credentials & paths stripped     |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Statutory Declarations (D01–D07) Internal Representation

| Code | Statutory Description | Legal Rule | Key Evidence Patterns | Visual Zone Anchor |
|:---:|:---|:---|:---|:---|
| **D01** | Manufacturer / Packer / Importer | Rule 6(1)(a) | `Mfd by`, `Packed by`, `Imported by`, PIN code | `manufacturer_info` |
| **D02** | Generic or Common Name | Rule 6(1)(b) | Product Title, Packaging Topline | Visual Packaging Header |
| **D03** | Net Quantity in Standard Units | Rule 6(1)(c) | `g`, `kg`, `ml`, `l`, `pcs`, `units` | `net_quantity` |
| **D04** | Date of Manufacture / Expiry | Rule 6(1)(d) | `Mfg Date`, `Best Before`, `Use by`, `MM/YYYY` | `mfg_date` |
| **D05** | Maximum Retail Price (MRP) | Rule 6(1)(da)| `MRP Rs.`, `₹`, `Incl. of all taxes` | `mrp_panel` |
| **D06** | Consumer Care / Complaints | Rule 6(1)(n) | Toll-Free `1800-xxx-xxxx`, `care@domain.com` | `consumer_care` |
| **D07** | Country of Origin | Rule 6(1)(ea)| `Country of Origin: India / Imported` | `country_of_origin` |

---

## 3. Disclaimers and User Transparency

- **Primary System Verdict vs Secondary Direct ML**:
  The UI prominently displays the **Multimodal Evidence-Aware Decision Engine** verdict as the primary statutory answer.
  The Direct ML Classifier is explicitly labeled as a **Secondary Diagnostic Layer** available only when Developer Mode is enabled.
- **Legal Advisory**:
  All UI views and exported artifacts display the mandatory disclaimer:
  *"This application is an automated compliance assistance tool for Legal Metrology (Packaged Commodities) Rules, 2011. It does not constitute a legal certification or guarantee of regulatory immunity."*
