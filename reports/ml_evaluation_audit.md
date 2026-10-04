# NitiMark-ML — Scientific ML Evaluation Audit & Data Provenance Verification

**Generated**: September 7, 2026  
**Auditor**: Antigravity ML Evaluation Team  
**Repository**: `/home/devabhay07/Desktop/MLmodelV2`  
**Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations  

---

## 1. Executive Summary & Objective

This document provides a comprehensive, rigorous scientific audit of the ML evaluation pipeline, dataset provenance, target formulations, feature representations, and baseline performance for the NitiMark-ML product compliance system.

The objective is to establish **genuine model utility** and **real-world physical packaging performance** using reproducible, evidence-based metrics—without data fabrication, artificial scores, or confusing model predictions with physical ground truth.

---

## 2. System Architecture & Target Definitions

### 2.1 Two-Tier Output Architecture

The NitiMark-ML compliance architecture separates the raw statistical model from the legal compliance decision engine:

1. **Primary ML Classifier Target (Binary)**:
   - Target Space: $\mathcal{Y}_{\text{ML}} = \{\text{COMPLIANT}, \text{NON\_COMPLIANT}\}$
   - Formulation: Supervised probabilistic classification $P(\text{COMPLIANT} \mid \mathbf{x})$.
   - Training: Trained exclusively on binary-labeled records where statutory declaration states are known.

2. **System-Level Legal Decision (3-Class with Abstention)**:
   - Output Space: $\mathcal{Y}_{\text{System}} = \{\text{COMPLIANT}, \text{NON\_COMPLIANT}, \text{UNCERTAIN}\}$
   - Formulation: Confidence-calibrated selective classification with evidence gating.
   - Abstention Triggers:
     - Insufficient optical clarity ($\text{OCR Quality} < 0.60$ and completeness $< 0.35$).
     - No readable packaging text detected (completeness $= 0.0$).
     - Conflicting declaration evidence across multi-image views.
     - Model confidence below calibrated threshold ($\tau_{\text{abstain}} = 0.65$).
     - Top probability margin below minimum separation ($\Delta p < 0.20$).

```
Raw Product Evidence (Image / Multi-Image Views)
                 │
                 ▼
    YOLOv8 + Gemini OCR Extraction
                 │
                 ▼
      D01–D07 Evidence Fusion
                 │
                 ▼
  Declaration Features X ∈ R^d
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Raw ML Prob      Evidence Gating & Quality
   P(C|X), P(NC|X)   (OCR Quality, Conflicts, Missingness)
        └────────┬────────┘
                 ▼
    Confidence Calibration & Thresholding
                 │
                 ▼
Final Verdict: COMPLIANT / NON_COMPLIANT / UNCERTAIN
```

---

## 3. Dataset Provenance & Label Audit

### 3.1 Dataset Inventory

| Dataset Partition | File Path | Total Records | COMPLIANT | NON_COMPLIANT | UNCERTAIN | Provenance Tier | Intended Role |
|---|---|:---:|:---:|:---:|:---:|---|---|
| **Flipkart Catalog** | `data/processed/flipkart_weak_labels_v1.jsonl` | 1,663 | 0 | 1,663 | 0 | `WEAKLY_DERIVED_SILVER` | Pretraining weak supervision |
| **YOLO Physical Packaging** | `data/processed/yolo_gemini_enhanced.jsonl` | 249 | 6 | 243 | 0 | `WEAKLY_DERIVED_SILVER` | Physical packaging spatial layout & regions |
| **Synthetic Physical Packaging** | `data/processed/synthetic_physical_packaging.jsonl` | 204 | 102 | 102 | 0 | `SYNTHETIC_GROUND_TRUTH` | Controlled balanced physical packaging training |
| **Real Gold Benchmark** | `data/processed/real_gold_20.jsonl` | 20 | 17 | 3 | 0 | `REAL_HUMAN_CURATED_BENCHMARK` | Physical multi-view packaging validation |
| **Pilot Human Reference 75** | `data/processed/pilot_ground_truth_75_v1.jsonl` | 75 | 2 | 61 | 12 | `ORIGINAL_PILOT_ANNOTATION` | Immutable benchmark evaluation |

---

### 3.2 Real Gold 20 Provenance Discrepancy Investigation

#### The Discrepancy:
- **Historical Benchmark Note**: Mentioned "Real Packaging 20: 20 NON_COMPLIANT, 0 COMPLIANT".
- **Current `real_gold_20.jsonl`**: Contains 17 COMPLIANT, 3 NON_COMPLIANT.

#### Provenance Trace & Root Cause:
1. **Physical Images & Manual References**:
   - Location: `real_product_validation/images/real_prod_01.png` through `real_prod_20.png`
   - Manual Reference Files: `real_product_validation/manual_reference/real_prod_01_ref.json` through `real_prod_20_ref.json`
   - These 20 files were curated by human inspection of physical consumer product packaging (Himalaya Face Wash, Tata Salt, Dettol Handwash, Britannia Biscuits, Sunfeast Dark Fantasy, etc.).
   - In the manual reference files, **17 packages possess all 6 mandatory Rule 6 declarations** (D01-D06) printed clearly on the packaging.
   - **3 packages have explicit statutory violations**:
     - `real_prod_11` (Unbranded Kitchen Towel): D01 (Manufacturer Name & Address) ABSENT, D06 (Consumer Care) ABSENT.
     - `real_prod_12` (Crafted Wooden Coasters): D01 ABSENT, D04 (Date) CANNOT_DETERMINE, D06 ABSENT.
     - `real_prod_17` (KitchenPro Spatula): D01 ABSENT, D05 (MRP) ABSENT, D06 ABSENT.

2. **Why Earlier Reports Noted "20 NON_COMPLIANT, 0 COMPLIANT"**:
   - In earlier Phase 11 experiments, a model trained **exclusively on Flipkart catalog text** was evaluated against these 20 packaging samples.
   - Because Flipkart catalog listings omit on-pack declarations, the Flipkart-only model learned an extreme prior ($P(\text{NON\_COMPLIANT}) \approx 1.0$) and **predicted `NON_COMPLIANT` for all 20 packages** (0 predicted COMPLIANT, Accuracy = 3/20 = 15%).
   - In some text summaries, this **prediction output** (20 predicted NC) was conflated with the **ground truth**.

3. **Verification of Generation Method**:
   - `src/data/generate_real_gold_labels.py` loads the 20 manual reference JSON files, parses the human-verified declaration annotations, and derives the overall target via deterministic Legal Metrology rules (`derive_compliance_label`).
   - The labels are derived from **human manual inspection of packaging labels**, NOT automated Gemini self-agreement.
   - **Provenance Classification**: Classified as `REAL_HUMAN_CURATED_BENCHMARK (N=20)`. Because $N=20$ is small, confidence intervals are wide, but **COMPLIANT recall is genuinely measurable** on this benchmark.

---

## 4. Feature Space & Controlled Sets (Sets A through E)

The system extracts numeric features strictly from extracted declaration states, confidences, values, and visual packaging attributes:

| Feature Set | Dimensionality | Description | Included Features |
|---|:---:|---|---|
| **SET_A** | 16 | Core Declaration Presence & Confidence | D01–D07 presence indicators, D01–D07 confidences, mandatory present count, all-mandatory flag |
| **SET_B** | 23 | Set A + OCR Quality & Text Statistics | Set A + OCR quality score, text length, word count, line count, upper ratio, digit ratio, symbol ratio |
| **SET_C** | 27 | Set B + Value Validity Flags | Set B + valid price flag, valid date flag, valid quantity flag, customer care completeness flag |
| **SET_D** | 35 | Set C + YOLO Spatial Features | Set C + YOLO total detections, region count, max bbox area, total bbox area, 4 spatial class indicators |
| **SET_E** | 43 | Set D + Advanced Cross-Declaration Ratios | Set D + mandatory ratio, optional ratio, confidence mean/std/min/max, packaging type indicators |

### Target Leakage Audit:
- No ground-truth label, target variable, provenance tag, or violation reason is included in the feature vectors.
- Verified by automated tests in `tests/product_compliance/test_leakage_audit.py`.

---

## 5. Model Registry & Training Artifacts

The repository maintains trained model artifacts under `models/product_compliance/`:

- `best_product_compliance_model.joblib`: Production model (`RandomForest_SET_A`, multi-domain trained).
- `logisticregression_set_[a-e]_combined.joblib`: Logistic Regression baseline models.
- `randomforest_set_[a-e]_combined.joblib`: Random Forest candidate models.
- `xgboost_set_[a-e]_combined.joblib`: XGBoost candidate models.
- `model_registry.json`: JSON metadata linking model paths, feature schemas, validation metrics, and timestamps.

---

## 6. Audit Conclusions

1. **Target Formulation is Sound**: Clear binary separation for ML training, wrapped with evidence-aware 3-class abstention for legal enforcement.
2. **Provenance Discrepancy Resolved**: The 17 C / 3 NC distribution in Real Gold is verified from human manual reference annotations.
3. **Reproducibility Guarantee**: All metrics, curves, and figures will be generated by a single deterministic script `src/evaluation/generate_ml_evaluation.py` without hardcoded numbers.
