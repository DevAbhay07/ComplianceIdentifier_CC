# Standalone Inference Architecture (Phase 9)

**Project:** NitiMark-ML — Standalone Legal Metrology Compliance Predictor  
**Phase:** Phase 9 — Standalone Inference & Real-Product Validation  
**Date:** September 2026  

---

## 1. Architectural Overview

The Phase 9 inference pipeline wraps the frozen Phase 8 machine learning models and the frozen Phase 7 preprocessing pipeline (`feature_pipeline_v1.joblib`) into a self-contained prediction engine. It operates completely independently of any web framework or external UI.

```
                    ┌───────────────────────────────┐
                    │      USER / CLIENT INPUT      │
                    │ (Image / Text / JSON Payload) │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        IMAGE VALIDATOR        │
                    │  (Format, Size, Corruption)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       VISION / OCR API        │
                    │ (Gemini 2.5 Flash / Mock OCR) │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        EVIDENCE PARSER        │
                    │   (D01-D07 Text Extraction)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   INFERENCE FEATURE ADAPTER   │
                    │ (Normalized Dict -> Raw Feats)│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  models/feature_pipeline_v1   │
                    │ (Fitted StandardScaler, OHE,  │
                    │     TF-IDF Word & Char)       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       SCHEMA VALIDATOR        │
                    │  (Verify 219 & 719 Dimensions)│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     FROZEN PHASE 8 MODELS     │
                    │  (RandomForest & Baselines)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     UNCERTAINTY & TRACE       │
                    │ (Confidence, Margins, Hashes) │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    CLI / JSON OUTPUT RESULT   │
                    └───────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Image Validator (`image_validator.py`)
- **Inputs:** Absolute or relative image file path.
- **Supported Formats:** `JPEG`, `JPG`, `PNG`, `WEBP`.
- **Integrity Checks:** File existence, non-zero size, maximum 25 MB, minimum dimension 50x50 px, and byte corruption detection via `PIL.Image.verify()`.

### 2.2 Vision / OCR Layer (`vision_ocr.py` & `mock_vision_ocr.py`)
- **Provider Abstraction:** `VisionOCRProvider` interface decouples OCR extraction from specific vendors.
- **Provider Implementations:**
  - `GoogleGeminiVisionProvider`: Integrates with `google-genai` using `gemini-2.5-flash`.
  - `MockVisionOCRProvider`: Deterministic local mock provider for automated unit tests and CI/CD without live API keys.
- **Prompting Principle:** Prompts explicitly demand extraction of visible text spans (manufacturer, net quantity, dates, MRP, customer care, country of origin) without requesting legal verdicts.

### 2.3 Evidence Parser (`evidence_parser.py`)
- Extracts regex and keyword matches for declarations D01 through D07.
- Computes diagnostic metadata: `extracted_text_length`, `detected_declarations_count`, and `completeness_score`.

### 2.4 Feature Adapter (`inference_features.py`)
- Transforms raw text and OCR spans into the standard raw feature dictionary expected by `NitiMarkFeaturePipeline`.
- Calls `pipeline.transform()` to produce exact Feature Set A (219 dimensions) and Feature Set B (719 dimensions).

### 2.5 Frozen Model Registry (`model_loader.py`)
- Loads model artifacts from `models/phase_8/` according to `model_freeze_record.md`:
  - `D01`: `RandomForest` on `FEATURE_SET_A`
  - `D02`: `MajorityBaseline` (`PRESENT`)
  - `D03`: `RandomForest` on `FEATURE_SET_A`
  - `D04`: `MajorityBaseline` (`CANNOT_DETERMINE`)
  - `D05`: `MajorityBaseline` (`PRESENT`)
  - `D06`: `RandomForest` on `FEATURE_SET_B`
  - `D07`: `MajorityBaseline` (`CANNOT_DETERMINE`)

### 2.6 Uncertainty Engine (`uncertainty.py`)
- Computes maximum probability confidence, probability margin ($P_{(1)} - P_{(2)}$), and normalized Shannon entropy.
- Assigns decision flag (`PREDICT` vs. `UNCERTAIN`).

### 2.7 Traceability & Audit Logging (`trace.py`)
- Hashes input images and text with SHA-256.
- Measures stage-by-stage latencies (validation, OCR, feature extraction, ML inference).
- Generates reproducible trace logs stored in `data/interim/inference_logs/`.
