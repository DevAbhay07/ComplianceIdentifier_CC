# Phase 10 Architectural & ML Improvement Recommendations

> **Project**: NitiMark-ML — Standalone Prediction Layer (SIH25057)  
> **Transition Objective**: Strategic engineering roadmap to upgrade NitiMark-ML from Phase 9.5 standalone benchmark to Phase 10 production-ready multimodal compliance system.

---

## 1. Executive Summary & Root Bottlenecks Identified in Phase 9.5

The Phase 9.5 empirical validation and multi-split benchmarking established two primary findings:
1. **Strong Within-Distribution Generalization**: Random Forest models achieved solid discriminative power on Flipkart weak-label splits ($F1 \ge 0.66$ for D01, D03, D06) and 100% agreement on universal fields (D02, D05).
2. **The Web-to-Packaging Domain Asymmetry**: Baseline single-class models for Date of Manufacture (`D04`) and Country of Origin (`D07`) fail on real physical packaging images because Flipkart catalog text systematically omitted these fields, whereas physical packaging universally includes them.

---

## 2. Priority 1: Physical Packaging Dataset Expansion ($N = 500\text{--}1,000$)

### Problem
Training models purely on e-commerce catalog web metadata creates an inescapable domain transfer ceiling for packaging-specific declarations (`D04` manufacturing date, `D07` country of origin, `D01` physical factory address).

### Recommended Action for Phase 10
- Construct a curated, multi-category packaging dataset of $500\text{--}1,000$ real packaging photographs across:
  - Food & FMCG (mandatory FSSAI + Legal Metrology dates)
  - Cosmetics & Personal Care (batch, manufacturing, expiry, origin)
  - Consumer Electronics & Appliances (importer, MRP, origin stamps)
  - Textile & Apparel (dimensions, fiber composition, wash care)
- Utilize the Phase 4.5B/C human-in-the-loop review tool with multi-model Gemini drafting to rapidly generate double-verified ground truth.

---

## 3. Priority 2: Replace Baseline Majority Predictors with Supervised Multimodal Models

### Problem
In Phase 8, `D02`, `D04`, `D05`, and `D07` were assigned majority baseline models due to extreme single-class imbalance in Flipkart web listings.

### Recommended Action for Phase 10
- Train dedicated Gradient Boosted Decision Trees (LightGBM/XGBoost) and Fine-tuned Multimodal Transformers (e.g., LayoutLMv3 or Donut) for all 7 declarations:
  - **D04 (Date)**: Feature detectors for batch date regexes (`\b(mfg|pkd|exp|best before)\b`), calendar date formats (`MM/YYYY`, `DD/MM/YYYY`), and numeric year spans.
  - **D07 (Origin)**: Feature detectors for ISO country name dictionaries, `"Made in [Country]"`, `"Country of Origin: [Country]"`, and importer declarations.

---

## 4. Priority 3: Multimodal Feature Fusion & Layout OCR Representation

### Problem
Current feature engineering flattens OCR text into bag-of-words and regex counts, discarding spatial bounding box coordinates and font size hierarchies.

### Recommended Action for Phase 10
- Implement **Spatial Layout Tokens**:
  - Encode bounding box coordinates $[x_0, y_0, x_1, y_1]$ and relative area for detected text segments.
  - Legal Metrology declarations often appear in specific packaging zones (e.g., principal display panel vs. back/bottom label).
- **Multilingual Token Normalizers**:
  - Incorporate multilingual OCR transliteration for Hindi, Tamil, Telugu, and regional text representations on Indian consumer commodities.

---

## 5. Priority 4: Statistical Confidence Calibration & Uncertainty Quantification

### Problem
The validation revealed that uncalibrated tree ensembles can produce high confidence ($\ge 0.80$) even when transferring across domain shifts.

### Recommended Action for Phase 10
- Apply **Platt Scaling** or **Isotonic Calibration** to map raw model logits to true empirical class probabilities.
- Enforce a **Confidence-Gated Human Escalation Workflow**:
  - If $\text{Margin}(\hat{y}) < 0.20$ or $\text{Entropy}(P) > 0.65$, automatically flag the product for human auditor review rather than issuing automated compliance flags.

---

## 6. Priority 5: High-Speed On-Premise OCR Integration

### Problem
Live cloud vision API calls introduce $1.2\text{s}\text{--}2.8\text{s}$ latency and external quota dependencies.

### Recommended Action for Phase 10
- Introduce a dual-tier OCR architecture:
  - **Tier 1 (Fast Local OCR)**: Run an embedded OCR engine (PaddleOCR / Tesseract-v5) directly in C++/Python for <150ms local processing.
  - **Tier 2 (Cloud Multimodal Fallback)**: Invoke Gemini Vision only when local OCR confidence score falls below $0.75$.

---

## 7. Roadmap to Main Application Integration (Post-Phase 10)

Once Phase 10 model retraining and packaging evaluation are complete, the standalone inference module (`src/inference/`) can be packaged into a clean microservice API or shared Python wheel for seamless integration into the NitiMark web dashboard and backend compliance engine.
