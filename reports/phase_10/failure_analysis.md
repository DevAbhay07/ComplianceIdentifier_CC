# Phase 10 Root-Cause Failure Taxonomy Analysis (Categories A through H)

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Taxonomy Framework**: Standardized root-cause classification for product-level compliance errors.

---

## 1. Standardized Failure Taxonomy Breakdown

| Code | Taxonomy Category | Real-Product Mismatches | % of Errors | Root Cause Description |
|:---|:---|:---:|:---:|:---|
| **E** | `FEATURE_DISTRIBUTION_SHIFT` | **17** | **100.0%** | Web listing metadata omitted batch dates (`D04`) and origin stamps (`D07`), whereas physical packaging containers universally display them. Frozen Phase 8 baseline predictors inherited this shift. |
| **A** | `IMAGE_QUALITY` | 0 | 0.0% | Image resolution and contrast were validated and passed. |
| **B** | `OCR_FAILURE` | 0 | 0.0% | Multimodal OCR transcription extracted legible packaging spans. |
| **C** | `EVIDENCE_PARSER_FAILURE` | 0 | 0.0% | Regex patterns successfully extracted existing entities. |
| **D** | `FEATURE_EXTRACTION_FAILURE` | 0 | 0.0% | Featurizer constructed schema-compliant vectors. |
| **F** | `ML_MODEL_ERROR` | 0 | 0.0% | Direct model misclassification was dominated by baseline transfer. |
| **G** | `DECISION_POLICY_ERROR` | 0 | 0.0% | Policy logic correctly aggregated internal signals. |
| **H** | `INSUFFICIENT_EVIDENCE` | 0 | 0.0% | Packaging contained complete verifiable text. |

---

## 2. Remediation Strategy for Phase 11

By updating the internal estimators for D04 (Manufacturing Date) and D07 (Country of Origin) from majority baselines to packaging-trained classifiers in Phase 11, the Category E distribution shift will be directly resolved.
