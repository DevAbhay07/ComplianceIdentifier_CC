# Phase 11 — Model Probability Calibration & Reliability Analysis

**Project**: NitiMark-ML (SIH25057)

**Milestone**: Phase 11U — Calibration

---


## 1. Brier Score & Confidence Reliability


| Model Architecture | Weak Test Brier Score | Real Product Brier Score | Reliability Assessment |

| :--- | :--- | :--- | :--- |

| **Phase11_RandomForest_SetA** | 0.0807 | 1.3626 | Moderate Overconfidence |

| **Phase11_RandomForest_SetB** | 0.0251 | 1.1002 | Moderate Overconfidence |

| **Phase11_RandomForest_SetC** | 0.0258 | 1.0521 | Moderate Overconfidence |

| **Phase11_RandomForest_SetD** | 0.0258 | 1.1064 | Moderate Overconfidence |

| **Phase11_XGBoost_SetD** | 0.0322 | 1.6937 | Moderate Overconfidence |

| **Phase11_LogisticRegression_SetD** | 0.0852 | 1.6764 | Moderate Overconfidence |

| **Phase11_EvidenceAware_Decision_Engine** | 0.0216 | 0.0150 | Well Calibrated |
