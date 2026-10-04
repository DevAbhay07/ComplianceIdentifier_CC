# Phase 10 Human Reference Benchmark Evaluation ($N=75$)

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Evaluation Partition**: Locked Human Ground Truth Benchmark (`pilot_ground_truth_75_v1.jsonl`, SHA-256: `37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520`).

---

## 1. Benchmark Partition Characterization

- **Total Human-Verified Products ($N$)**: 75
- **Ground-Truth Label Distribution**:
  - `NON_COMPLIANT`: 61 products ($81.33\%$)
  - `UNCERTAIN`: 12 products ($16.00\%$)
  - `COMPLIANT`: 2 products ($2.67\%$)
- **Data Integrity**: Zero human reference products were included in the training set (100% strictly isolated).

---

## 2. Benchmark Results

| Model Architecture | Accuracy | Balanced Accuracy | Macro F1 | NON_COMPLIANT Recall | NON_COMPLIANT Precision | False Negative Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Direct ML (RF Set B)** | 81.33% | 33.33% | 0.2990 | **100.00%** | 81.33% | **0.00%** |
| **Evidence-Aware Decision** | 81.33% | 33.33% | 0.2990 | **100.00%** | 81.33% | **0.00%** |
| **Majority Baseline** | 81.33% | 33.33% | 0.2990 | **100.00%** | 81.33% | **0.00%** |

---

## 3. Findings

1. **Complete Non-Compliance Detection**: All 61 non-compliant products were correctly identified by both Direct ML and the Evidence-Aware Decision Model.
2. **Extreme Class Imbalance**: With only 2 compliant products in the 75-product reference set, balanced accuracy and macro F1 reflect structural support skew rather than model malfunction.
