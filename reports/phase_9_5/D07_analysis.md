# Declaration Deep Dive: `D07` — Country of Origin

## 1. Legal Metrology Definition
Governed under the Legal Metrology (Packaged Commodities) Rules, 2011 for **Country of Origin**.

## 2. Multi-Split Performance Overview

| Partition | Sample Count ($N$) | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | Mean Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1159 | 99.57% | 50.00% | **49.89%** | 99.35% | NOT APPLICABLE (Single Class) | 0.0 | 100.00% |
| **Validation** | 246 | 99.59% | 50.00% | **49.90%** | 99.39% | NOT APPLICABLE (Single Class) | 0.0 | 100.00% |
| **Test** | 258 | 99.22% | 50.00% | **49.81%** | 98.84% | NOT APPLICABLE (Single Class) | 0.0 | 100.00% |
| **Human_Reference** | 75 | 57.33% | 50.00% | **36.44%** | 41.79% | NOT APPLICABLE (Single Class) | 0.0 | 100.00% |

## 3. Test Partition Confusion Matrix

|                       |   Pred_CANNOT_DETERMINE |   Pred_PRESENT |
|:----------------------|------------------------:|---------------:|
| True_CANNOT_DETERMINE |                     256 |              0 |
| True_PRESENT          |                       2 |              0 |

## 4. Human Reference Transfer Benchmark Confusion Matrix

|                       |   Pred_CANNOT_DETERMINE |   Pred_PRESENT |
|:----------------------|------------------------:|---------------:|
| True_CANNOT_DETERMINE |                      43 |              0 |
| True_PRESENT          |                      32 |              0 |

## 5. Diagnostic Insights & Domain Transfer

- **Weak-Label Test Macro F1:** `49.81%`
- **Human Reference Macro F1:** `36.44%`
- **Domain Transfer Gap:** `+13.37%`

Discrepancy stems from omission of date stamps and origin statements in web metadata listings versus physical packaging labels.
