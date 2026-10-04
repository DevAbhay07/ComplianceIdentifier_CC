# Data Leakage & Split Isolation Audit Report
> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms  
> **Phase**: Corrective Phase — Declaration-Centric ML  
> **Status**: **✅ PASS (ZERO LEAKAGE DETECTED)**  

---

## 1. Executive Summary

A comprehensive 4-point leakage audit was executed over the declaration-centric feature engineering pipeline and dataset partitions to verify scientific validity and eliminate circular evaluation.

| Audit Check | Scope | Verification Target | Status |
|:---|:---|:---|:---:|
| **1. Feature Name Independence** | `src/features/declaration_feature_extractor.py` | Zero target-derived column names in Sets A–E | ✅ PASS |
| **2. Split Partition Disjointness** | `data/processed/ml_dataset_v1/` | Disjoint product IDs across Train, Val, Test, Ref | ✅ PASS |
| **3. Real Product Isolation** | `real_product_validation/` ($N=20$) | Real packaging benchmark excluded from training | ✅ PASS |
| **4. Feature Correlation Sanity** | $X \in \mathbb{R}^{N \times D}$ vs $y$ | No trivial 1.0 target-identity columns in $X$ | ✅ PASS |

---

## 2. Partition Disjointness Verification

- **Training Partition**: 1159 products
- **Validation Partition**: 246 products
- **Test Partition**: 258 products
- **Human Reference Partition**: 75 products
- **Train $\cap$ Val Overlap**: 0 (Disjoint)
- **Train $\cap$ Test Overlap**: 0 (Disjoint)
- **Train $\cap$ Human Ref Overlap**: 0 (Disjoint)

---

## 3. Benchmark Sample Isolation

- Real packaging benchmark samples ($N=20$) are verified to have zero representation in the training partition.
- Human reference samples ($N=75$) are verified to have zero representation in the training partition.
