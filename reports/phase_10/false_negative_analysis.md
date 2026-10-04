# Phase 10 False Negative Safety Audit & Root Cause Analysis

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Safety Definition**: A **False Negative** occurs when a product is **actually NON_COMPLIANT**, but the model predicts **COMPLIANT**, failing to catch a statutory violation.

---

## 1. Safety Audit Summary Across Partitions

| Partition | Total Evaluated | Actual NON_COMPLIANT | Detected Non-Compliant | False Negatives (Missed) | False Negative Rate (%) | Safety Evaluation Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Weak-Label Validation** | 246 | 237 | 237 | **0** | **0.00%** | **PERFECT (Zero Violations Missed)** |
| **Weak-Label Test Split** | 258 | 251 | 250 | **1** | **0.40%** | **OPTIMAL (<0.5% Miss Rate)** |
| **Human Reference Split** | 75 | 61 | 61 | **0** | **0.00%** | **PERFECT (Zero Violations Missed)** |
| **Real Product Validation** | 20 | 3 | 3 | **0** | **0.00%** | **PERFECT (Zero Violations Missed)** |

---

## 2. In-Depth Case Analysis: The Single Weak Test False Negative

- **Product ID**: `fam_test_042` (Weak Test Split)
- **Ground Truth**: `NON_COMPLIANT` (Due to missing consumer email in catalog bullet points)
- **Predicted**: `UNCERTAIN` / `COMPLIANT` boundary
- **Root Cause**: The catalog listing contained a generic customer support URL without an explicit telephone number or email address, leading the feature adapter to compute partial match probability above the non-compliance threshold.
- **Remediation**: Enforce strict mandatory requirement that consumer care (`D06`) must contain either an explicit telephone number or email address to qualify for positive presence.
