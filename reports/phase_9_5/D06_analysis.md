# Declaration Deep Dive: `D06` — Consumer Care Contact Details

## 1. Legal Metrology Definition
Governed under the Legal Metrology (Packaged Commodities) Rules, 2011 for **Consumer Care Contact Details**.

## 2. Multi-Split Performance Overview

| Partition | Sample Count ($N$) | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | Mean Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1159 | 100.00% | 100.00% | **100.00%** | 100.00% | 1.0 | 1.0 | 92.18% |
| **Validation** | 246 | 98.78% | 85.00% | **90.86%** | 98.68% | 0.8314 | 0.8174 | 90.35% |
| **Test** | 258 | 98.84% | 83.33% | **89.70%** | 98.72% | 0.8116 | 0.7943 | 91.07% |
| **Human_Reference** | 75 | 82.67% | 25.00% | **22.63%** | 74.82% | NOT APPLICABLE (Single Class) | 0.0 | 92.92% |

## 3. Test Partition Confusion Matrix

|              |   Pred_ABSENT |   Pred_PRESENT |
|:-------------|--------------:|---------------:|
| True_ABSENT  |           249 |              0 |
| True_PRESENT |             3 |              6 |

## 4. Human Reference Transfer Benchmark Confusion Matrix

|                       |   Pred_ABSENT |   Pred_CANNOT_DETERMINE |   Pred_PARTIAL |   Pred_PRESENT |
|:----------------------|--------------:|------------------------:|---------------:|---------------:|
| True_ABSENT           |            62 |                       0 |              0 |              0 |
| True_CANNOT_DETERMINE |             2 |                       0 |              0 |              0 |
| True_PARTIAL          |             3 |                       0 |              0 |              0 |
| True_PRESENT          |             8 |                       0 |              0 |              0 |

## 5. Diagnostic Insights & Domain Transfer

- **Weak-Label Test Macro F1:** `89.70%`
- **Human Reference Macro F1:** `22.63%`
- **Domain Transfer Gap:** `+67.07%`

Trained Random Forest classifier successfully learns text and tabular patterns with high in-distribution generalization, but exhibits domain transfer gap on subtle physical packaging phrasing.
