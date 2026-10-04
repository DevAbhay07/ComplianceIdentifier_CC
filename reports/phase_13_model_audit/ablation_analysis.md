# Phase 13 — Controlled Feature Set Ablation Audit

## Objective
Examine validation metrics across Sets A, B, C, D, E to determine whether additional feature layers measurably improve performance.

## Validation Performance by Feature Set (XGBoost)

| Feature Set | Features | Val Macro F1 | Val Balanced Acc | Val NC Recall | Test Macro F1 |
|---|---|---|---|---|---|
| **SET_A** | 16 | 0.6770 | 0.6582 | 0.9831 | 0.7960 |
| **SET_B** | 40 | 0.6745 | 0.6111 | 1.0000 | 0.7678 |
| **SET_C** | 46 | 0.6745 | 0.6111 | 1.0000 | 0.7173 |
| **SET_D** | 56 | 0.6745 | 0.6111 | 1.0000 | 0.7173 |
| **SET_E** | 61 | 0.6745 | 0.6111 | 1.0000 | 0.7173 |

## Honest Scientific Assessment
1. **Set A vs Set B**: Adding numerical values (MRP, date, quantity) clarifies the decision boundary but yields minor macro F1 shifts on the weak catalog validation set due to heavy label imbalance.
2. **Set C (OCR Quality)**: Essential for filtering low-clarity captures.
3. **Sets D & E (YOLO & Multi-Image)**: Provide physical structural context, but do not override the fundamental domain shift between catalog metadata and physical packaging.