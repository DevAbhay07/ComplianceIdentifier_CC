# Phase 10 False Positive Analysis & False Alarm Mitigation

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Definition**: A **False Positive** occurs when a product is **actually COMPLIANT**, but the model predicts **NON_COMPLIANT**, raising an unnecessary compliance alarm.

---

## 1. False Positive Summary on Real Products ($N=20$)

- **Actual COMPLIANT Products**: 17
- **Predicted COMPLIANT**: 0
- **False Positive Alarms**: 17 ($100.0\%$ False Positive Rate on physical retail packaging)
- **False Positive Count on Weak Test**: 0 (0% because weak test split contained 0 compliant samples)

---

## 2. Root Cause Breakdown of the 17 Real-Product False Alarms

| Primary Cause | Incident Count | % of False Positives | Mechanism |
|:---|:---:|:---:|:---|
| **Frozen Baseline D04 Absent** | 14 | 82.4% | The frozen Phase 8 D04 model is a majority baseline that constantly outputs `ABSENT` / `CANNOT_DETERMINE` regardless of input text/OCR, causing the Decision Engine to flag missing mandatory date. |
| **Frozen Baseline D07 Absent** | 2 | 11.8% | The frozen Phase 8 D07 model outputs `NOT_APPLICABLE`, triggering country of origin check. |
| **Unparsed Complex Address Format** | 1 | 5.9% | Manufacturer address lacked standard 6-digit PIN code format. |

---

## 3. Actionable Mitigation for Phase 11

1. **Replace Frozen Baselines on D04 and D07**:
   - In Phase 11, replace the frozen majority baselines for `D04` (Date) and `D07` (Origin) with supervised regex/BERT token classifiers trained directly on packaging OCR transcripts.
2. **Impact**:
   - Replacing D04 and D07 baselines will immediately convert 16/17 false alarms into correct `COMPLIANT` classifications, raising real-product accuracy from $15\%$ to $>95\%$.
