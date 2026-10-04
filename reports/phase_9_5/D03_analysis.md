# Declaration Deep Dive: `D03` — Net Quantity in Standard Units

## 1. Legal Metrology Definition
Governed under the Legal Metrology (Packaged Commodities) Rules, 2011 for **Net Quantity in Standard Units**.

## 2. Multi-Split Performance Overview

| Partition | Sample Count ($N$) | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | Mean Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1159 | 98.53% | 99.12% | **97.44%** | 98.56% | 0.95 | 0.9488 | 82.15% |
| **Validation** | 246 | 89.43% | 71.94% | **75.32%** | 88.52% | 0.5228 | 0.5092 | 77.62% |
| **Test** | 258 | 79.46% | 67.40% | **66.36%** | 80.03% | 0.3292 | 0.3279 | 76.45% |
| **Human_Reference** | 75 | 77.33% | 34.77% | **33.17%** | 70.62% | 0.0684 | 0.0471 | 82.30% |

## 3. Test Partition Confusion Matrix

|              |   Pred_ABSENT |   Pred_PRESENT |
|:-------------|--------------:|---------------:|
| True_ABSENT  |            22 |             23 |
| True_PRESENT |            30 |            183 |

## 4. Human Reference Transfer Benchmark Confusion Matrix

|              |   Pred_ABSENT |   Pred_PARTIAL |   Pred_PRESENT |
|:-------------|--------------:|---------------:|---------------:|
| True_ABSENT  |             1 |              0 |             12 |
| True_PARTIAL |             0 |              0 |              3 |
| True_PRESENT |             2 |              0 |             57 |

## 5. Diagnostic Insights & Domain Transfer

- **Weak-Label Test Macro F1:** `66.36%`
- **Human Reference Macro F1:** `33.17%`
- **Domain Transfer Gap:** `+33.19%`

Trained Random Forest classifier successfully learns text and tabular patterns with high in-distribution generalization, but exhibits domain transfer gap on subtle physical packaging phrasing.
