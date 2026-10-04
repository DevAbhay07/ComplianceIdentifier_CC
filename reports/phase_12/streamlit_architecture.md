# Streamlit Application Architecture & Component Design
> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms  
> **Phase**: Phase 12 — Streamlit Product Compliance Testing Application  

---

## 1. Architectural Philosophy & Scope Boundaries

The Phase 12 Streamlit application (`app/streamlit_app.py`) is engineered as a **pure testing and demonstration interface** for the frozen Phase 11 Multimodal Product Compliance Engine.

```
+-----------------------------------------------------------------------------------+
|                           STREAMLIT UI LAYER (app/)                               |
|                                                                                   |
|  +--------------------+  +----------------------+  +---------------------------+  |
|  | Input Panel        |  | Primary Hero Card    |  | Plain-English "Why"       |  |
|  | Upload / Samples   |  | COMPLIANT/NON/UNCERT |  | Statutory Justification   |  |
|  +---------+----------+  +----------^-----------+  +-------------^-------------+  |
|            |                        |                            |                |
|            +------------------------+----------------------------+                |
|                                     |                                             |
|  +--------------------+  +----------+-----------+  +---------------------------+  |
|  | YOLOv8 Visualizer  |  | ROI Patch Gallery    |  | OCR Text & Clarity        |  |
|  | Bounding Box Box   |  | Cropped Declarations |  | Structured JSON Viewer    |  |
|  +--------------------+  +----------------------+  +---------------------------+  |
|                                                                                   |
|  +--------------------+  +-----------------------------------------------------+  |
|  | Sanitized JSON     |  | Developer Diagnostic Mode (Checkbox)                |  |
|  | Export Button      |  | 29 YOLO Visual Feats | Set D Vector | Direct ML      |  |
|  +--------------------+  +-----------------------------------------------------+  |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                    INFERENCE PIPELINE BRIDGE (app/utils/)                         |
|                                                                                   |
|  +--------------------------+    +--------------------------+                     |
|  | Image Validation Guard   |    | Singleton Engine Bridge  |                     |
|  | Format/Dims/Integrity    |    | Cached Resource Mgmt     |                     |
|  +--------------------------+    +--------------------------+                     |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                   FROZEN STANDALONE ML & DECISION CORE (src/)                     |
|                                                                                   |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  | YOLOv8 Localization|  | OCR / Vision API   |  | Evidence Parser             |  |
|  | 6 Visual Classes   |  | Gemini 3.6 / Mock  |  | Rule-based Entity Extractor |  |
|  +---------+----------+  +----------+---------+  +--------------+--------------+  |
|            |                        |                           |                 |
|            +------------------------+---------------------------+                 |
|                                     |                                             |
|                                     v                                             |
|  +-----------------------------------------------------------------------------+  |
|  | Multimodal Feature Extraction (Set D, D=741 features)                       |  |
|  | - 219 Base Features (Set A)                                                 |  |
|  | - 29 YOLO Visual Spatial / Confidence Features                              |  |
|  | - 493 Text & Context Features                                               |  |
|  +----------------------------------+------------------------------------------+  |
|                                     |                                             |
|                                     v                                             |
|  +----------------------------------+------------------------------------------+  |
|  | Primary Engine: Statutory Evidence-Aware Decision Engine                    |  |
|  | - Rule 6 PCR 2011 Non-Compliance Strict Trigger                            |  |
|  | - Visual Corroboration Guard (Dual Confirmation)                            |  |
|  | - Calibrated Confidence & Ambiguity Penalty                                |  |
|  +-----------------------------------------------------------------------------+  |
|                                     |                                             |
|  +----------------------------------v------------------------------------------+  |
|  | Secondary Engine: Direct ML Classifier (RandomForest / XGBoost)            |  |
|  | - Direct Multiclass Softmax Probabilities (Diagnostic Telemetry Only)      |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### Strict System Boundaries:
1. **Zero Database / Auth Infrastructure**: The application is fully stateless. Uploaded images reside exclusively in memory or temporary execution contexts and are destroyed immediately post-inference.
2. **Zero Model Retraining**: Models (`yolov8_packaging.pt`, `best_product_compliance_model.joblib`, `feature_extractor_v2.joblib`) are strictly read-only and frozen.
3. **Decoupled UI & Inference Core**: All UI rendering logic resides in `app/components/` and interacts with ML models solely via `app/utils/inference_bridge.py`.

---

## 2. Directory Structure & Modular Separation

```
app/
├── __init__.py
├── streamlit_app.py             # Main entry point & session orchestration
├── components/                  # Modular UI Presentation Layer
│   ├── __init__.py
│   ├── result_card.py          # Primary Hero Verdict (COMPLIANT/NON_COMPLIANT/UNCERTAIN)
│   ├── why_explanation.py      # Plain-English statutory justification & legal rationale
│   ├── evidence_panel.py       # D01-D07 internal declaration status table
│   ├── yolo_viewer.py          # Annotated bounding box visualizer
│   ├── roi_viewer.py           # Cropped packaging declaration patch gallery
│   ├── ocr_viewer.py           # Raw OCR text & optical clarity viewer
│   ├── technical_trace.py      # Developer Mode diagnostics (29 YOLO feats, ML probs, latency)
│   └── export.py               # Sanitized JSON audit trail download
└── utils/                       # Bridge & Security Utilities
    ├── __init__.py
    ├── validation.py           # Upload integrity, byte verification, format/dimension guards
    ├── visualizer.py           # Color-coded bounding box rendering with Pillow
    └── inference_bridge.py     # Singleton bridge to frozen Phase 11 inference engine
```

---

## 3. Component Details & Responsibilities

### 3.1 Input & Pre-Validation (`app/utils/validation.py`)
- Enforces strict constraints before image bytes touch inference pipelines:
  - Allowed formats: `PNG`, `JPG`, `JPEG`, `WEBP`.
  - Max file size: $15.0\text{ MB}$.
  - Min dimensions: $100 \times 100\text{ px}$; Max dimensions: $6000 \times 6000\text{ px}$.
  - Integrity check: Verified via `PIL.Image.verify()` to prevent truncated or malformed payloads from crashing workers.

### 3.2 Result Hero Card (`app/components/result_card.py`)
- Renders the single user-facing answer to *"Is this product compliant?"*
- Color taxonomy:
  - `COMPLIANT`: Forest Green (`#166534`), badge `COMPLIANT`, solid green container.
  - `NON_COMPLIANT`: Crimson Red (`#991b1b`), badge `NON_COMPLIANT`, solid red container.
  - `UNCERTAIN`: Amber Gold (`#92400e`), badge `UNCERTAIN`, solid amber container.
- Displays calibrated confidence percentage and primary statutory grounds.

### 3.3 Statutory "Why" Explanation (`app/components/why_explanation.py`)
- Translates ML feature representations and evidence extraction into actionable, legally grounded plain English.
- Links specific missing declarations to Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 6).
- Discloses optical quality and visual corroboration state to avoid false assumptions.

### 3.4 Evidence Panel (`app/components/evidence_panel.py`)
- Collapsible/secondary inspection table for mandatory statutory declarations:
  - `D01`: Manufacturer / Packer / Importer Name & Address
  - `D02`: Generic or Common Name of Commodity
  - `D03`: Net Quantity / Standard Metric Units
  - `D04`: Month & Year of Manufacture / Pre-packing / Expiry
  - `D05`: Maximum Retail Price (MRP incl. of all taxes)
  - `D06`: Consumer Care / Grievance Redressal Contacts
  - `D07`: Country of Origin (for imported commodities)
- Status badges: `PRESENT`, `ABSENT`, `NOT_APPLICABLE`, `UNCERTAIN` alongside extracted snippets.

### 3.5 YOLOv8 Visualizer & ROI Gallery (`app/components/yolo_viewer.py`, `roi_viewer.py`)
- Visual bounding box overlay drawn onto the original image using distinct color coding:
  - `mrp_panel`: Blue (`#3B82F6`)
  - `mfg_date`: Green (`#10B981`)
  - `net_quantity`: Purple (`#8B5CF6`)
  - `consumer_care`: Amber (`#F59E0B`)
  - `manufacturer_info`: Teal (`#06B6D4`)
  - `country_of_origin`: Red (`#EF4444`)
- Dynamic cropped ROI patch display showing high-resolution cutouts of detected declaration zones.

### 3.6 OCR Viewer (`app/components/ocr_viewer.py`)
- Tabbed interface displaying raw extracted OCR text, optical clarity assessment, character count, and structured token representations.

### 3.7 Technical Diagnostic Trace (`app/components/technical_trace.py`)
- Hidden by default; activated via **Developer Mode** toggle in sidebar.
- Diagnostic metrics:
  - Detailed per-stage latency breakdown (Validation, YOLO, ROI, OCR, Feature Pipeline, Direct ML, Decision Engine).
  - 29 YOLOv8 Visual Features table (bounding box counts, mean area, aspect ratios, confidence aggregates).
  - Multimodal Feature Vector dimensions ($D=741$).
  - Direct ML Classifier vs Statutory Decision Engine comparative breakdown with softmax probability distribution.

### 3.8 Sanitized Export (`app/components/export.py`)
- Exports full audit trail as structured JSON.
- Guaranteed redaction: Strips all API keys, local filesystem paths, environment tokens, and unmasked credentials.

---

## 4. State Management & Lifecycle

- Application state is orchestrated using Streamlit's native `st.session_state`:
  - `st.session_state.inference_result`: Stores latest prediction result dictionary.
  - `st.session_state.processed_image`: Stores loaded PIL image object.
  - `st.session_state.last_processed_file`: Checksum/name cache to prevent redundant re-runs on UI interactions.
- Engine persistence: `InferencePipelineBridge.get_instance()` employs a singleton pattern, loading YOLOv8 weights and scikit-learn models once into memory.
