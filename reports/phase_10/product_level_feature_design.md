# Phase 10 Product-Level Feature Strategy & Design

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Feature Scope**: Feature representation, anti-leakage boundaries, dimensional constraints, and feature set definitions.

---

## 1. Feature Set Specifications

Phase 10 evaluates Direct ML and Evidence-Aware Decision pipelines using strictly partitioned feature representations:

| Feature Set | Dimensionality | Constituent Subspaces | Typical Use Case |
|:---|:---:|:---|:---|
| **Feature Set A** | **219** | 21 Numerical / Structural (text length, word count, entity counts, prices) + 198 One-Hot Categoricals (brand, category). | Lightweight, high-speed CPU inference (<5ms). |
| **Feature Set B** | **719** | Set A (219) + 300 Word TF-IDF + 200 Char n-gram TF-IDF. | High-capacity linear and tree models (Optimal Validation Macro F1: 0.9354). |
| **Feature Set C** | **Internal Evidence ($D=7$)** | Internal predicted probabilities, confidence bounds, and status tokens from D01–D07. | Consumed by the Evidence-Aware Decision Engine. |

---

## 2. Anti-Leakage Boundary Assertions

To guarantee complete scientific integrity, the following features are **strictly forbidden** from entering feature matrices $X$:
1. `overall_target` (Ground truth label).
2. `D01_target` through `D07_target` (Declaration-level ground truth targets).
3. `label_source`, `label_confidence`, `quality_tier` (Provenance metadata).
4. `rule_derived`, `compliance_score` (Synthetic rule engine outputs).

---

## 3. Top Feature Importances (Direct ML Random Forest Set B)

| Feature Name | Subspace | Feature Importance (%) | Interpretability / Behavioral Impact |
|:---|:---|:---:|:---|
| `num_mrp_regex_matches` | Structural | 14.8% | High occurrence signals valid price declaration. |
| `raw_text_length` | Numerical | 11.2% | Comprehensive packaging text correlates with compliance completeness. |
| `num_email_regex_matches` | Structural | 9.5% | Direct indicator for D06 consumer care presence. |
| `num_phone_regex_matches` | Structural | 8.7% | Direct indicator for D06 consumer contact helpline. |
| `num_pin_regex_matches` | Structural | 7.9% | Strong predictor for D01 manufacturer postal PIN code. |
| `tfidf_word_mfd` | Text N-Gram | 6.4% | Manufacturing keyword token. |
| `tfidf_word_pkd` | Text N-Gram | 5.8% | Packaging date keyword token. |
| `num_unit_regex_matches` | Structural | 5.1% | Net quantity metric units predictor. |
