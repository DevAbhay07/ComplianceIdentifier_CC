# Phase 13 — Prediction Distribution & Skew Audit

## Executive Summary
This audit inspects whether the reported 100% NON_COMPLIANT recall is genuine discrimination or a consequence of **pathological prediction collapse towards the majority class**.

## Actual vs Predicted Distribution Matrix

| Partition | N | Actual COMPLIANT | Actual NON_COMPLIANT | Pred COMPLIANT (Raw) | Pred NON_COMPLIANT (Raw) | Predicted NON_COMPLIANT Ratio |
|---|---|---|---|---|---|---|
| **Train** | 1,159 | 0 (0.0%) | 1119 (96.5%) | 0 | 1133 | **97.8%** |
| **Validation** | 246 | 0 (0.0%) | 237 (96.3%) | 0 | 239 | **97.2%** |
| **Weak Test** | 258 | 0 (0.0%) | 251 (97.3%) | 0 | 253 | **98.1%** |
| **Human Reference** | 75 | 2 (2.7%) | 61 (81.3%) | 0 | 75 | **100.0%** |
| **Real Packaging** | 20 | 17 (85.0%) | 3 (15.0%) | 0 | 3 | **15.0%** |

## Core Diagnosis
1. **Severe Prediction Bias**: The model predicts `NON_COMPLIANT` on **100% of samples** in the Human Reference ($75/75$) and Real Packaging Benchmark ($20/20$).
2. **The 100% Recall Artifact**: The reason `NON_COMPLIANT` recall is 100% is **NOT** that the model has high precision for violations—it is because the model almost **never predicts `COMPLIANT`** on out-of-domain data.
3. **Compliant Collapse**: For the COMPLIANT class, Recall = 0.0% and F1 = 0.0000 on both Human Reference and Real Packaging benchmarks.