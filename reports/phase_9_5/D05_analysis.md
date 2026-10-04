# Declaration Deep Dive: `D05` — Maximum Retail Price (MRP)

## 1. Legal Metrology Definition
Governed under the Legal Metrology (Packaged Commodities) Rules, 2011 for **Maximum Retail Price (MRP)**.

## 2. Multi-Split Performance Overview

| Partition | Sample Count ($N$) | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | Mean Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1159 | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | 100.00% |
| **Validation** | 246 | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | 100.00% |
| **Test** | 258 | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | 100.00% |
| **Human_Reference** | 75 | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | 100.00% |

## 3. Test Partition Confusion Matrix

|              |   Pred_PRESENT |
|:-------------|---------------:|
| True_PRESENT |            258 |

## 4. Human Reference Transfer Benchmark Confusion Matrix

|              |   Pred_PRESENT |
|:-------------|---------------:|
| True_PRESENT |             75 |

## 5. Diagnostic Insights & Domain Transfer

- **Weak-Label Test Macro F1:** `100.00%`
- **Human Reference Macro F1:** `100.00%`
- **Domain Transfer Gap:** `+0.00%`

Universal e-commerce availability enables perfect 100.00% precision across both weak and human splits via majority baseline.
