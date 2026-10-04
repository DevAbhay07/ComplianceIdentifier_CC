# Declaration Deep Dive: `D04` — Date of Manufacture / Packing / Expiry

## 1. Legal Metrology Definition
Governed under the Legal Metrology (Packaged Commodities) Rules, 2011 for **Date of Manufacture / Packing / Expiry**.

## 2. Multi-Split Performance Overview

| Partition | Sample Count ($N$) | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | Mean Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1159 | 99.65% | 50.00% | **49.91%** | 99.48% | NOT APPLICABLE (Single Class) | 0.0 | 100.00% |
| **Validation** | 246 | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | 100.00% |
| **Test** | 258 | 99.61% | 50.00% | **49.90%** | 99.42% | NOT APPLICABLE (Single Class) | 0.0 | 100.00% |
| **Human_Reference** | 75 | 57.33% | 50.00% | **36.44%** | 41.79% | NOT APPLICABLE (Single Class) | 0.0 | 100.00% |

## 3. Test Partition Confusion Matrix

|                       |   Pred_CANNOT_DETERMINE |   Pred_PRESENT |
|:----------------------|------------------------:|---------------:|
| True_CANNOT_DETERMINE |                     257 |              0 |
| True_PRESENT          |                       1 |              0 |

## 4. Human Reference Transfer Benchmark Confusion Matrix

|                       |   Pred_CANNOT_DETERMINE |   Pred_PRESENT |
|:----------------------|------------------------:|---------------:|
| True_CANNOT_DETERMINE |                      43 |              0 |
| True_PRESENT          |                      32 |              0 |

## 5. Diagnostic Insights & Domain Transfer

- **Weak-Label Test Macro F1:** `49.90%`
- **Human Reference Macro F1:** `36.44%`
- **Domain Transfer Gap:** `+13.46%`

Discrepancy stems from omission of date stamps and origin statements in web metadata listings versus physical packaging labels.
