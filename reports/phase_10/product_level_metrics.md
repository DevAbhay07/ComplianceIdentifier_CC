# Phase 10 Comprehensive Product-Level Compliance Metrics

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Evaluation Scope**: Granular per-class metrics, confusion matrices, and safety indicators across all partitions.

---

## 1. Selected Direct ML Model (Random Forest Set B)

### In-Distribution Weak Test Partition ($N=258$)
- **Accuracy**: $98.45\%$
- **Balanced Accuracy**: $85.50\%$
- **Macro Precision**: $91.33\%$ | **Macro Recall**: $85.50\%$ | **Macro F1**: **$0.8607$**
- **Weighted F1**: $0.9840$ | **MCC**: $0.7812$ | **Cohen's Kappa**: $0.7725$
- **NON_COMPLIANT Precision**: $99.20\%$ | **Recall**: $99.20\%$ | **F1**: **$0.9920$**
- **UNCERTAIN Precision**: $75.00\%$ | **Recall**: $42.86\%$ | **F1**: **$0.5455$**
- **False Negative Count**: 0 ($0.00\%$ False Negative Rate)

### Frozen Human Reference Partition ($N=75$)
- **Accuracy**: $81.33\%$
- **Balanced Accuracy**: $33.33\%$
- **Macro F1**: **$0.2990$** (Class imbalance: COMPLIANT=2, UNCERTAIN=12, NON_COMPLIANT=61)
- **NON_COMPLIANT Recall**: **$100.00\%$**

### Real-Product Validation Partition ($N=20$)
- **Accuracy**: $15.00\%$
- **Balanced Accuracy**: $50.00\%$
- **Macro F1**: **$0.1304$**
- **NON_COMPLIANT Recall**: **$100.00\%$** (3/3 non-compliant products detected)
- **False Negatives**: 0 ($0.0\%$ False Negative Rate)
- **False Positives**: 17 ($100.0\%$ False Alarm Rate on compliant physical packages)

---

## 2. Evidence-Aware Decision Model Metrics

| Metric | Weak Validation ($N=246$) | Weak Test ($N=258$) | Human Reference ($N=75$) | Real Product ($N=20$) |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 96.34% | 96.90% | 81.33% | 15.00% |
| **Balanced Accuracy** | 50.00% | 64.29% | 33.33% | 50.00% |
| **Macro F1** | 0.4907 | 0.6191 | 0.2990 | 0.1304 |
| **NON_COMPLIANT Recall** | **100.00%** | **99.60%** | **100.00%** | **100.00%** |
| **NON_COMPLIANT Precision** | 96.34% | 97.28% | 81.33% | 15.00% |
| **False Negative Rate** | **0.00%** | **0.40%** | **0.00%** | **0.00%** |
| **Uncertainty Rate** | 0.00% | 1.16% | 0.00% | 0.00% |
| **Coverage** | 100.00% | 98.84% | 100.00% | 100.00% |
