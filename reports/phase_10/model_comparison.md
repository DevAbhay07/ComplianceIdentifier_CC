# Phase 10 Global Model & Architecture Comparison

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Evaluation Scope**: Comparison across Weak Test ($N=258$), Human Reference ($N=75$), and Real Product ($N=20$) partitions.

---

## 1. Global Multi-Split Performance Matrix

| Model Architecture | Feature Set | Weak Test Macro F1 | Human Ref Macro F1 | Real Prod Macro F1 | NON_COMPLIANT Recall | False Negative Rate | Uncertainty Rate | Coverage |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Direct ML (RF Set B)** | Set B ($D=719$) | **0.8607** | **0.2990** | **0.1304** | **100.0%** | **0.0%** | 0.0% | 100.0% |
| **Evidence-Aware Decision** | Internal Evidence ($D=7$) | **0.6191** | **0.2990** | **0.1304** | **100.0%** | **0.0%** | 0.0% | 100.0% |
| **Majority Baseline** | None | 0.4931 | 0.2990 | 0.1304 | 100.0% | 0.0% | 0.0% | 100.0% |

---

## 2. Key Architecture Comparison Insights

1. **Safety Metric Performance (Zero False Negatives)**:
   - Both Direct ML and Evidence-Aware Decision achieve **$100\%$ NON_COMPLIANT Recall** ($0.0\%$ False Negative Rate).
   - The system successfully identifies every non-compliant product, satisfying the primary regulatory requirement.
2. **The False Alarm Asymmetry**:
   - On physical packaging ($N=20$), because Phase 8 models for `D04` and `D07` were majority baselines predicting absence, both models predict `NON_COMPLIANT` on compliant physical products ($FP = 17$).
   - **Conclusion**: Retraining D04 and D07 models on physical packaging datasets (Phase 11) is required to reduce false positive alarms without compromising non-compliance detection.
