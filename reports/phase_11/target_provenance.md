# Phase 11 — Product-Level Target Provenance & Anti-Leakage Audit

**Project**: NitiMark-ML — Automated Compliance Checker for Legal Metrology (SIH25057)  
**Milestone**: Phase 11J — Target Provenance Audit  
**Date**: September 2026

---

## 1. Target Definition & Statutory Basis
The target variable $Y \in \{\text{COMPLIANT}, \text{NON\_COMPLIANT}, \text{UNCERTAIN}\}$ represents overall compliance with the **Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 6)**.

Statutory Rule 6 mandates 7 essential packaging declarations (D01 to D07):
- **D01**: Manufacturer / Packer / Importer Name and Address
- **D02**: Generic or Common Commodity Name
- **D03**: Net Quantity in Standard SI Units
- **D04**: Month and Year of Manufacture / Packing / Import
- **D05**: Maximum Retail Price (MRP) inclusive of all taxes
- **D06**: Consumer Care / Grievance Helpline Contact Details
- **D07**: Country of Origin

---

## 2. Multi-Dataset Target Provenance Analysis

### A. Flipkart Weak-Label Dataset (N=1,663)
- **Provenance**: Generated in Phase 6 via deterministic legal heuristics over Flipkart e-commerce catalog metadata.
- **Derivation Logic**:
  $$\text{Target} = \begin{cases} \text{NON\_COMPLIANT} & \text{if any mandatory declaration is ABSENT} \\ \text{COMPLIANT} & \text{if all mandatory declarations are PRESENT} \\ \text{UNCERTAIN} & \text{otherwise} \end{cases}$$
- **Observed Distribution**:
  - `NON_COMPLIANT`: 1,607 (96.63%)
  - `UNCERTAIN`: 56 (3.37%)
  - `COMPLIANT`: 0 (0.00%)
- **Root Cause of Extreme Skew**: Standard e-commerce catalog web pages consistently omit manufacturing dates (`D04`), customer care cell addresses (`D06`), and origin details (`D07`). Consequently, the heuristic rule classifies 97% as non-compliant.
- **Classification Nature**: **SILVER LABELS / WEAK SUPERVISION**.

### B. Human Reference Benchmark (N=75)
- **Provenance**: Manually annotated, reviewed, and finalized by expert human legal reviewers in Phase 4.5/5.
- **Cryptographic Lock**: SHA-256 `37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520`.
- **Observed Distribution**:
  - `NON_COMPLIANT`: 61 (81.33%)
  - `UNCERTAIN`: 12 (16.00%)
  - `COMPLIANT`: 2 (2.67%)
- **Classification Nature**: **GOLD STANDARD HUMAN GROUND TRUTH** (Catalog Domain). Strictly evaluation-only.

### C. Real Product Validation Dataset (N=20)
- **Provenance**: Independently collected from retail store physical packaging with multi-angle photography and manual expert verification.
- **Observed Distribution**:
  - `COMPLIANT`: 17 (85.00%)
  - `NON_COMPLIANT`: 3 (15.00%)
  - `UNCERTAIN`: 0 (0.00%)
- **Classification Nature**: **GOLD STANDARD PHYSICAL PACKAGING BENCHMARK**. Strictly evaluation-only.

---

## 3. Strict Anti-Leakage Verification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE INDEPENDENCE AUDIT                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Feature Set A: Structural Price, Category, Lengths  ───► [NO TARGET LEAK]  │
│  Feature Set B: Set A + TF-IDF Text N-Grams          ───► [NO TARGET LEAK]  │
│  Feature Set C: Set B + OCR Token Signals & Density  ───► [NO TARGET LEAK]  │
│  Feature Set D: Set C + YOLO Visual BBox Metrics     ───► [NO TARGET LEAK]  │
├─────────────────────────────────────────────────────────────────────────────┤
│  PROHIBITED INPUTS: D01_label, D02_label, ... D07_label, rule_engine_status │
│  STATUS: ZERO PROHIBITED INPUTS PERMITTED IN MODEL TRAINING                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Target Derivation Leakage Check**: In Phase 10 and Phase 11, raw D01–D07 target states are strictly banned from feature matrices. Models only receive independent text, metadata, OCR token statistics, and visual geometry.
2. **Group-Aware Splitting Check**: Train ($N=1,159$), Validation ($N=246$), and Test ($N=258$) splits are strictly partitioned by `product_family_id` using `GroupShuffleSplit`. No products sharing the same normalized title stem exist across partitions.
3. **External Benchmark Isolation**: Zero Human Reference ($N=75$) or Real Product ($N=20$) samples are used during model training, feature extraction fitting, or hyperparameter/threshold tuning.

---

## 4. Scientific Conclusion on Supervised Product-Level ML

Because catalog weak supervision contains 0% `COMPLIANT` samples, any direct supervised classifier trained strictly on weak catalog data will learn a degenerate majority prior.

Therefore, Phase 11 implements a dual-architecture paradigm:
1. **Direct ML Supervised Classifiers** (Random Forest, XGBoost, Logistic Regression) evaluated on Sets A–D to quantify the empirical boundary of weak catalog training.
2. **Evidence-Aware Decision Engine with Multimodal Visual Localization** combining OCR extraction, YOLO ROI tags, and Rule 6 statutory logic to deliver robust, false-alarm-free compliance verdicts on real physical packaging.
