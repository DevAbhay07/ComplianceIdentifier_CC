# Multimodal Model Trace & Technical Feature Lineage
> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms  
> **Phase**: Phase 12 — Streamlit Testing App  

---

## 1. Multimodal Feature Lineage & Taxonomy

The Phase 12 inference engine evaluates packaging inputs across **three distinct evidence representations**:

```
                                [ Real Product Packaging Image ]
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
        [ YOLOv8 Visual Localization ]                          [ OCR / Vision API ]
        - 6 Visual Zone Classes                                 - Raw Text & Quality
        - Spatial Bounding Boxes [x1, y1, x2, y2]               - Tokenizer & NER
                        |                                               |
                        v                                               v
        +-------------------------------+               +-------------------------------+
        |  29 Continuous Visual Features|               |  Rule 6 Parsed Declarations   |
        |  - Class presence counts (6)  |               |  (D01 through D07 status &    |
        |  - Bounding box areas (6)     |               |   confidence scores)          |
        |  - Aspect ratios (6)          |               +---------------+---------------+
        |  - Centroid coordinates (6)   |                               |
        |  - Detection confidence (5)   |                               |
        +---------------+---------------+                               |
                        |                                               |
                        +-----------------------+-----------------------+
                                                |
                                                v
                        +-----------------------------------------------+
                        |  Set D Multimodal Feature Vector (D = 741)    |
                        |  - Set A Base Textual Features (D = 219)      |
                        |  - Set B Domain Context Features (D = 493)    |
                        |  - Set D YOLO Visual Features (D = 29)        |
                        +-----------------------+-----------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
        +-------------------------------+               +-------------------------------+
        |  Direct ML Compliance Model   |               |  Statutory Decision Engine    |
        |  (RandomForest / XGBoost)     |               |  (Evidence-Aware Logic Core)  |
        |  - Secondary Diagnostic       |               |  - Primary System Verdict     |
        |  - Multiclass Softmax Probs   |               |  - Calibrated Confidence      |
        +-------------------------------+               +-------------------------------+
```

---

## 2. 29 YOLOv8 Visual Spatial & Confidence Features

| Index | Feature Name | Dimension Type | Statutory Significance |
|:---:|:---|:---:|:---|
| 0 | `yolo_count_mrp_panel` | Count $\in [0, 5]$ | Frequency of detected retail price markings |
| 1 | `yolo_count_mfg_date` | Count $\in [0, 5]$ | Presence of manufacture/expiry date stamps |
| 2 | `yolo_count_net_quantity` | Count $\in [0, 5]$ | Presence of standard metric volume/weight units |
| 3 | `yolo_count_consumer_care` | Count $\in [0, 5]$ | Toll-free helpline or email visual blocks |
| 4 | `yolo_count_manufacturer_info` | Count $\in [0, 5]$ | Physical entity name/address typography blocks |
| 5 | `yolo_count_country_of_origin` | Count $\in [0, 5]$ | Origin markings on imported/domestic packaging |
| 6–11 | `yolo_area_<class>` | Normalized Float $[0, 1]$ | Visual prominence & legible surface area on packaging |
| 12–17 | `yolo_aspect_ratio_<class>` | Float $[0.1, 10.0]$ | Bounding box geometry (horizontal banner vs box) |
| 18–23 | `yolo_center_y_<class>` | Normalized Float $[0, 1]$ | Vertical hierarchy (e.g. MRP usually bottom/side panel)|
| 24 | `yolo_total_detections` | Integer | Total visual declaration elements detected |
| 25 | `yolo_unique_classes` | Integer $\in [0, 6]$ | Distinct statutory declaration classes localized |
| 26 | `yolo_mean_confidence` | Float $[0, 1]$ | Mean detector confidence across all bounding boxes |
| 27 | `yolo_max_confidence` | Float $[0, 1]$ | Maximum single-detection confidence score |
| 28 | `yolo_total_area_coverage` | Normalized Float $[0, 1]$ | Total packaging surface area occupied by declarations |

---

## 3. Decision Model vs Direct ML Discrepancy Analysis

The application enforces a clear division between the **Statutory Decision Engine** and the **Direct ML Classifier**:

1. **Why the Decision Engine is Primary**:
   - Legal Metrology (Packaged Commodities) Rules, 2011 are **deterministic and conjunctive**: If a mandatory declaration (e.g., MRP or Consumer Care) is demonstrably ABSENT, the commodity is legally non-compliant, regardless of overall catalog text length or high ML prior probabilities.
   - The Decision Engine enforces strict safety constraints ($100\%$ recall on true non-compliant products) while maintaining dual-confirmation visual corroboration.

2. **Why Direct ML is Secondary/Diagnostic**:
   - The Direct ML classifier (`RandomForest_SET_D`) provides a soft probability distribution based on multimodal statistical embeddings. It is exposed in Developer Mode for transparency, model debugging, and domain-shift auditing.
