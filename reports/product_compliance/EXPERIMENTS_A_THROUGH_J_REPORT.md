# Product Compliance ML — Multi-Domain Experiments (A through J)

**Generated**: 2026-09-05 23:25:59 UTC

## 1. Executive Summary & Experiment Matrix

| Experiment | Description | Train Size | Val Size | Best Model | Macro F1 | Bal Acc | MCC | C Recall | NC Recall | NC FN Rate |
|---|---|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXPERIMENT_A_FLIPKART_ONLY` | Flipkart catalog only | 1164 | 249 | `SET_A::LogisticRegression` | 1.0000 | 1.0000 | 0.0000 | NOT MEASURABLE | 1.0 | 0.0 |
| `EXPERIMENT_B_YOLO_ONLY` | Eligible YOLO physical packaging only | 174 | 37 | `SET_A::XGBoost` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |
| `EXPERIMENT_C_SYNTHETIC_ONLY` | Synthetic physical packaging only | 168 | 36 | `SET_A::XGBoost` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |
| `EXPERIMENT_D_FLIPKART_YOLO` | Flipkart + YOLO fusion | 1338 | 286 | `SET_A::RandomForest` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |
| `EXPERIMENT_E_FLIPKART_SYNTHETIC` | Flipkart + Synthetic augmentation | 1332 | 285 | `SET_A::XGBoost` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |
| `EXPERIMENT_F_UNIFIED_ALL` | Unified Multi-Domain (Flipkart + YOLO + Synthetic) | 1506 | 322 | `SET_A::RandomForest` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |
| `EXPERIMENT_G_REAL_ONLY` | Real Gold Physical Packaging only (REAL_HUMAN_VERIFIED) | 14 | 3 | `SET_A::LogisticRegression` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |
| `EXPERIMENT_H_REAL_SYNTHETIC` | Real Gold + Synthetic Physical Packaging | 182 | 39 | `SET_A::XGBoost` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |
| `EXPERIMENT_I_REAL_SYNTHETIC_YOLO` | Real Gold + Synthetic + YOLO Packaging Fusion | 356 | 76 | `SET_A::RandomForest` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |
| `EXPERIMENT_J_UNIFIED_REAL_ALL` | All-Domain Master (Real + Flipkart + Synthetic + YOLO) | 1520 | 325 | `SET_A::RandomForest` | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 0.0 |

## 2. Baseline Comparison (Always-COMPLIANT vs Always-NON_COMPLIANT)

> [!IMPORTANT]
> When evaluating on benchmarks with high class imbalance (e.g. Flipkart 100% NC or Real 20), trivial single-class baselines achieve misleading scores. The model must beat both baselines on balanced sets and maintain a near-zero Non-Compliant False Negative Rate.
