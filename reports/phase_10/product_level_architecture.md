# Phase 10 Product-Level Compliance Prediction Architecture

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Architecture Goal**: Transition from disparate declaration-level outputs (D01–D07) to an end-to-end, evidence-aware product compliance decision system.

---

## 1. System Architecture Overview

The Phase 10 system implements a dual-layer prediction architecture where raw multimodal inputs are transcribed into structured declaration evidence, processed by internal learned estimators, and synthesized by an Evidence-Aware Decision Engine into an overall compliance verdict:

```
                          MULTIMODAL PRODUCT INPUT
                   (Packaging Image + Product Metadata)
                                     |
                                     v
                       [IMAGE VALIDATION & METADATA]
                                     |
                                     v
                           [VISION / OCR LAYER]
                 (Google Gemini 3.6 Flash / Local Engine)
                                     |
                                     v
                         [EVIDENCE PARSING LAYER]
                 (Regex entity extraction for D01-D07)
                                     |
                                     v
                       [FEATURE ADAPTATION PIPELINE]
                     (Set A: D=219 | Set B: D=719)
                                     |
                                     v
                      [INTERNAL SIGNALS ESTIMATION]
                (Frozen Phase 8 Multi-Label Model Suite)
                                     |
                                     v
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
  [APPROACH A: DIRECT ML]                         [APPROACH B: EVIDENCE-AWARE DECISION]
(Random Forest / XGBoost on $X$)                  (Statutory Applicability + Confidence)
         |                                                       |
         +---------------------------+---------------------------+
                                     |
                                     v
                      [PRODUCT COMPLIANCE VERDICT]
                +------------------------------------+
                |  COMPLIANT                         |
                |  NON_COMPLIANT                     |
                |  UNCERTAIN (Abstained)             |
                +------------------------------------+
                |  Confidence Score (0.0 to 1.0)     |
                |  Transparent Decision Basis (D01-7)|
                |  Statutory Legal Disclaimer        |
                +------------------------------------+
```

---

## 2. Core Architectural Components

### 1. Vision & OCR Ingestion Layer (`src/inference/vision_ocr.py`)
- Transcribes observable text from packaging containers without prompting LLMs for legal compliance verdicts.
- Employs `gemini-3.6-flash` in online mode and deterministic regex extractors in offline mode.

### 2. Evidence Parser & Feature Adapter (`src/inference/inference_features.py`)
- Extracts structured lexical tokens, numerical dimensions, date patterns, contact info, and MRP tags into aligned feature matrices:
  - **Feature Set A ($D=219$)**: Numerical, structural, and one-hot categorical features.
  - **Feature Set B ($D=719$)**: Set A + 300-dim word TF-IDF + 200-dim char TF-IDF.

### 3. Internal Declaration Signal Evaluator (`src/inference/declaration_predictor.py`)
- Runs the frozen Phase 8 multi-label model suite across D01 through D07.
- Retains internal predicted labels, class probabilities, margins, and prediction entropy.

### 4. Evidence-Aware Compliance Decision Engine (`src/decision/product_compliance_decision.py`)
- Evaluates statutory applicability (e.g. durable goods date exemptions under Legal Metrology Rules).
- Synthesizes positive, negative, and uncertain declaration states into an overall product-level verdict.
- Generates transparent, human-auditable decision traces.
