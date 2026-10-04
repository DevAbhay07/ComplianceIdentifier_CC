# Phase 11 — False Positive / False Alarm Root-Cause Analysis

**Project**: NitiMark-ML (SIH25057)

**Milestone**: Phase 11 — False Positive Analysis

---


## 1. False Positive Rate Reduction

In Phase 10, direct ML models flagged 17 out of 17 compliant physical packages as non-compliant (100% false alarm rate) due to catalog domain shift.


| Model Architecture | Real Product False Positive Rate | False Positive Count / 17 |

| :--- | :--- | :--- |

| **Phase11_RandomForest_SetA** | 100.0% | 17 / 17 |

| **Phase11_RandomForest_SetB** | 100.0% | 17 / 17 |

| **Phase11_RandomForest_SetC** | 100.0% | 17 / 17 |

| **Phase11_RandomForest_SetD** | 100.0% | 17 / 17 |

| **Phase11_XGBoost_SetD** | 100.0% | 17 / 17 |

| **Phase11_LogisticRegression_SetD** | 100.0% | 17 / 17 |

| **Phase11_EvidenceAware_Decision_Engine** | 0.0% | 0 / 17 |
