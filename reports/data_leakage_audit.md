# Data Leakage Audit — NitiMark ML Pipeline

**Date**: 2026-09-05
**Phase**: 4 — Pilot Analysis

---

## Purpose

This audit identifies potential data leakage risks in the proposed ML training pipeline. Data leakage occurs when information from outside the training set is used to create the model, leading to overly optimistic performance estimates and unreliable predictions.

---

## Risk Assessment

### RISK 1 — Rule-Engine Labels Used as ML Targets

| Attribute | Value |
|---|---|
| **Severity** | 🔴 CRITICAL |
| **Status** | ✅ MITIGATED |
| **Description** | If the rule engine generates compliance labels and those labels are used to train the ML model, the ML model simply learns to imitate the rule engine — providing no additional predictive value. |
| **Mitigation** | Training labels come from human-reviewed annotations stored in `annotations/pilot_annotations.jsonl`. The draft engine was used for acceleration but human modifications are tracked separately via `machine_guidance.human_modified`. |
| **Residual Risk** | LOW — 36 annotations show possible human oversight (accepted overall label without updating it after changing declarations). These need re-review. |

---

### RISK 2 — Overall Label Included as Input Feature

| Attribute | Value |
|---|---|
| **Severity** | 🔴 CRITICAL |
| **Status** | ⚠️ REQUIRES ATTENTION AT FEATURE ENGINEERING |
| **Description** | If the overall compliance label (COMPLIANT/NON_COMPLIANT/UNCERTAIN) is accidentally used as an input feature during training, the model will achieve near-perfect accuracy but learn nothing. |
| **Mitigation** | The feature engineering phase must explicitly exclude `overall_assessment.label` from the feature matrix. Declaration-level labels may be used as multi-label targets but never as features. |

---

### RISK 3 — Machine Draft Labels Contaminating Ground Truth

| Attribute | Value |
|---|---|
| **Severity** | 🟡 MODERATE |
| **Status** | ⚠️ PARTIALLY MITIGATED |
| **Description** | The draft annotation engine generates suggested labels. If humans accept all drafts without critical review, the "human" labels are effectively machine-generated. |
| **Evidence** | Draft modification analysis shows: D02 had 0% modification (all accepted), D05 had 0% modification. D04 had 100% modification, D07 had 99%, D01 had 93%. |
| **Mitigation** | Modification tracking (`human_modified` field) allows post-hoc analysis. However, 36 records with inconsistent overall labels suggest some annotations were not fully reviewed. |
| **Recommendation** | Flag products where `human_modified=False` for all 7 declarations and re-review a sample. |

---

### RISK 4 — Duplicate Products in Train/Test Split

| Attribute | Value |
|---|---|
| **Severity** | 🟡 MODERATE |
| **Status** | ⚠️ REQUIRES ATTENTION AT TRAINING |
| **Description** | If the same product appears in both training and test sets, the model memorizes rather than generalizes. Also, products from the same brand/manufacturer may share similar patterns. |
| **Mitigation** | Use product-level splitting (never split a product's annotations across sets). Consider group-based splitting by brand to prevent brand-level leakage. |

---

### RISK 5 — Features Containing Future Information

| Attribute | Value |
|---|---|
| **Severity** | 🟡 MODERATE |
| **Status** | ✅ MITIGATED BY DESIGN |
| **Description** | If features include information not available at prediction time (e.g., annotation timestamps, review status), the model gets access to future data. |
| **Mitigation** | The feature matrix should only include product listing data (name, specs, price, description, brand, category). Annotation metadata must be excluded. |

---

### RISK 6 — OCR Extraction Features Containing Labels

| Attribute | Value |
|---|---|
| **Severity** | 🟡 MODERATE |
| **Status** | ✅ NOT APPLICABLE YET |
| **Description** | If OCR text extracted from product images accidentally contains the compliance label text (e.g., from an annotator's marking on the image), the model learns from the label. |
| **Mitigation** | Not applicable in the current listing-only pipeline. When image/OCR features are added, raw OCR output must be audited before use as features. |

---

### RISK 7 — Human Annotations Influenced by Machine Predictions

| Attribute | Value |
|---|---|
| **Severity** | 🟡 MODERATE |
| **Status** | ⚠️ KNOWN RISK |
| **Description** | The draft annotation system presents machine suggestions before the human annotates. This could anchor the human's judgment. |
| **Evidence** | D02 and D05 had 0% modification rate — humans accepted all drafts. This may reflect genuine agreement, or anchoring bias. |
| **Mitigation** | Future annotation rounds should include a subset annotated without draft assistance to measure inter-method agreement. |

---

### RISK 8 — Category Information Leaking into Compliance Labels

| Attribute | Value |
|---|---|
| **Severity** | 🟢 LOW |
| **Status** | ✅ ACCEPTABLE |
| **Description** | All 75 pilot products come from 3 categories (Beauty, Baby Care, Kitchen). If category is used as a feature and the model learns category-compliance correlations, it may not generalize to new categories. |
| **Mitigation** | Future dataset expansion should include additional categories. Evaluate model performance per-category, not just overall. |

---

## Summary

| Risk | Severity | Status |
|---|---|---|
| Rule-engine labels as targets | 🔴 CRITICAL | ✅ Mitigated |
| Overall label as input feature | 🔴 CRITICAL | ⚠️ Attention needed at feature engineering |
| Draft labels contaminating ground truth | 🟡 MODERATE | ⚠️ Partially mitigated (36 records need review) |
| Duplicate products in train/test | 🟡 MODERATE | ⚠️ Attention needed at training |
| Future information in features | 🟡 MODERATE | ✅ Mitigated by design |
| OCR features containing labels | 🟡 MODERATE | ✅ Not applicable yet |
| Anchoring bias from drafts | 🟡 MODERATE | ⚠️ Known risk |
| Category leakage | 🟢 LOW | ✅ Acceptable |
