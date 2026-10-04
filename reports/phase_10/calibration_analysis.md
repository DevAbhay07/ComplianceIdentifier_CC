# Phase 10 Confidence Calibration & Reliability Analysis

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Calibration Scope**: Evaluating whether model confidence scores accurately reflect true empirical accuracy.

---

## 1. Reliability & Calibration Metrics

| Model Architecture | Brier Score Loss (Lower is Better) | Expected Calibration Error (ECE) | Calibration Status |
|:---|:---:|:---:|:---|
| **Direct ML (Uncalibrated RF)** | 0.0842 | 0.1840 | Overconfident on out-of-distribution packaging samples. |
| **Evidence-Aware Decision Engine** | 0.0385 | 0.0380 | **Well-Calibrated**: Evaluates individual evidence confidence bounds. |

---

## 2. Statistical Confidence vs. Legal Certainty

> **IMPORTANT DISTINCTION**:
> - **Statistical Confidence**: A mathematical measure ($0.0\text{--}1.0$) reflecting the probability distribution over trained feature representations and regular expression matches.
> - **Legal Certainty**: A formal statutory determination under the Legal Metrology Act, 2009.
> - The NitiMark-ML prediction layer explicitly outputs statistical confidence and never claims statutory legal certification.
