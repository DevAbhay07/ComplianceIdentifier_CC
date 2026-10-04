# Phase 10 Uncertainty Quantification & Abstention Analysis

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Core Principle**: The system must be allowed to abstain with **`UNCERTAIN — insufficient evidence`** when packaging signals are ambiguous or unverified.

---

## 1. Uncertainty Metrics Across Evaluated Partitions

| Partition | Total Evaluated | Decisive Predictions (Covered) | Abstained Predictions (UNCERTAIN) | Uncertainty Rate (%) | Coverage (%) | Selective Accuracy on Covered (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weak-Label Validation** | 246 | 246 | 0 | 0.00% | 100.00% | 96.34% |
| **Weak-Label Test** | 258 | 255 | 3 | 1.16% | 98.84% | 98.04% |
| **Human Reference** | 75 | 75 | 0 | 0.00% | 100.00% | 81.33% |
| **Real Product Validation** | 20 | 20 | 0 | 0.00% | 100.00% | 15.00% |

---

## 2. Abstention Criteria & Gating Policy

The Decision Engine triggers `UNCERTAIN` when:
1. An applicable mandatory declaration (`D01`, `D02`, `D03`, `D05`, `D06`) returns `CANNOT_DETERMINE` without any established absence.
2. Prediction confidence falls below $\tau_{\text{pos}} = 0.40$ or entropy $H(P) > 0.65$.
3. Image validation indicates low resolution ($<100\text{px}$) or OCR text length $<10$ characters.
