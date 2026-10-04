# Declaration Deep Dive: `D01` — Name & Address of Manufacturer / Packer / Importer

## 1. Legal Metrology Definition
Governed under the Legal Metrology (Packaged Commodities) Rules, 2011 for **Name & Address of Manufacturer / Packer / Importer**.

## 2. Multi-Split Performance Overview

| Partition | Sample Count ($N$) | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | Mean Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1159 | 99.74% | 99.85% | **99.59%** | 99.74% | 0.9953 | 0.9953 | 90.36% |
| **Validation** | 246 | 93.50% | 86.84% | **86.84%** | 93.50% | 0.8862 | 0.8862 | 79.96% |
| **Test** | 258 | 90.70% | 82.15% | **82.56%** | 90.57% | 0.8415 | 0.8414 | 78.33% |
| **Human_Reference** | 75 | 40.00% | 47.28% | **30.13%** | 39.87% | 0.1205 | 0.1 | 77.41% |

## 3. Test Partition Confusion Matrix

|              |   Pred_ABSENT |   Pred_PARTIAL |   Pred_PRESENT |
|:-------------|--------------:|---------------:|---------------:|
| True_ABSENT  |           103 |              0 |              0 |
| True_PARTIAL |             0 |            115 |             11 |
| True_PRESENT |             0 |             13 |             16 |

## 4. Human Reference Transfer Benchmark Confusion Matrix

|              |   Pred_ABSENT |   Pred_PARTIAL |   Pred_PRESENT |
|:-------------|--------------:|---------------:|---------------:|
| True_ABSENT  |             4 |              0 |              1 |
| True_PARTIAL |            14 |             25 |              4 |
| True_PRESENT |            15 |             11 |              1 |

## 5. Diagnostic Insights & Domain Transfer

- **Weak-Label Test Macro F1:** `82.56%`
- **Human Reference Macro F1:** `30.13%`
- **Domain Transfer Gap:** `+52.43%`

Trained Random Forest classifier successfully learns text and tabular patterns with high in-distribution generalization, but exhibits domain transfer gap on subtle physical packaging phrasing.
