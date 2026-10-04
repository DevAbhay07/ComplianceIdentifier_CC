# Phase 11 — Multi-Model Multi-Partition Benchmark Comparison

**Project**: NitiMark-ML (SIH25057)

**Milestone**: Phase 11P — Comprehensive Evaluation

---


## 1. Multi-Partition Performance Comparison


| Model Architecture | Feature Set | Weak Test Macro F1 | Human Ref Macro F1 | Real Product Macro F1 | Real Product False Positive Rate | Safety: NON_COMPLIANT Recall |

| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| **Phase11_RandomForest_SetA** | `SET_A` | 0.3267 | 0.2990 | **0.1304** | 100.0% | **100.0%** |

| **Phase11_RandomForest_SetB** | `SET_B` | 0.3307 | 0.2990 | **0.1304** | 100.0% | **100.0%** |

| **Phase11_RandomForest_SetC** | `SET_C` | 0.3307 | 0.2990 | **0.1304** | 100.0% | **100.0%** |

| **Phase11_RandomForest_SetD** | `SET_D` | 0.3307 | 0.2990 | **0.1304** | 100.0% | **100.0%** |

| **Phase11_XGBoost_SetD** | `SET_D` | 0.3307 | 0.2990 | **0.1304** | 100.0% | **100.0%** |

| **Phase11_LogisticRegression_SetD** | `SET_D` | 0.3273 | 0.2990 | **0.1304** | 100.0% | **100.0%** |

| **Phase11_EvidenceAware_Decision_Engine** | `MULTIMODAL_EVIDENCE_RULES` | 0.6410 | 0.5577 | **1.0000** | 0.0% | **100.0%** |


## 2. Key Takeaways

1. **Direct ML Catalog Skew**: Direct ML classifiers trained on weak catalog metadata continue to predict non-compliance heavily on real physical packaging because catalog data lacks manufacturing dates and helpline numbers.

2. **Multimodal Decision Engine Superiority on Physical Packaging**: The Evidence-Aware Decision Engine, utilizing visual and OCR packaging evidence, achieves the highest real packaging Macro F1 while maintaining 100% safety recall on defective products.
