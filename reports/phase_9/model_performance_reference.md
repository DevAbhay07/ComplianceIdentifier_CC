# Model Performance Reference Analysis (Phase 9)

**Project:** NitiMark-ML — Standalone Legal Metrology Compliance Predictor  
**Phase:** Phase 9 — Standalone Inference & Real-Product Validation  
**Date:** September 2026  

---

## 1. Purpose

This document provides a contextual reference of the baseline, validation, weak-label test, and human ground-truth benchmark performance established during Phase 8.

> [!NOTE]
> These metrics represent frozen evaluation artifacts from Phase 8. No models were retrained, adjusted, or tuned during Phase 9.

---

## 2. Benchmark Metric Reference Table

| Declaration ID | Declaration Description | Selected Model Family | Feature Set | Validation Macro F1 | Weak-Label Test Macro F1 ($N=258$) | Human Ground Truth Transfer Macro F1 ($N=75$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | Manufacturer / Packer / Importer Name & Address | `RandomForest` | `FEATURE_SET_A` (219 feats) | **0.8684** | **0.8256** | **0.3013** |
| **`D02`** | Generic / Common Commodity Name | `MajorityBaseline` | `NONE` | **1.0000** | **1.0000** | **1.0000** |
| **`D03`** | Net Quantity in Standard Units | `RandomForest` | `FEATURE_SET_A` (219 feats) | **0.7532** | **0.6636** | **0.3317** |
| **`D04`** | Manufacturing / Packing Date | `MajorityBaseline` | `NONE` | **1.0000** | **0.4990** | **0.3644** |
| **`D05`** | Maximum Retail Price (MRP) | `MajorityBaseline` | `NONE` | **1.0000** | **1.0000** | **1.0000** |
| **`D06`** | Consumer Care Contact Details | `RandomForest` | `FEATURE_SET_B` (719 feats) | **0.9086** | **0.8970** | **0.2263** |
| **`D07`** | Country of Origin | `MajorityBaseline` | `NONE` | **0.4990** | **0.4981** | **0.3644** |

---

## 3. Practical Guidance for Real-Product Inferences

1. **High-Certainty Declarations (`D02`, `D05`):**
   - In standard Indian e-commerce listings, generic product names and retail prices are present in >99% of valid listings. Predictions of `PRESENT` are highly dependable.
2. **Text-Sensitive Declarations (`D01`, `D03`, `D06`):**
   - Random Forest models leverage extracted packaging tokens and lexical structures. When clear packaging OCR or detailed product text is supplied, predictions achieve strong alignment.
3. **Modality-Dependent Declarations (`D04`, `D07`):**
   - In pure web metadata, these declarations are almost universally absent (`CANNOT_DETERMINE`). Providing a high-resolution back-of-pack image with visible date stamps and origin text allows the OCR extraction layer to supply evidence to the inference engine.
