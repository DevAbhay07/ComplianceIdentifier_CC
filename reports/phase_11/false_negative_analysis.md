# Phase 11 — Safety-Critical False Negative Root-Cause Analysis

**Project**: NitiMark-ML (SIH25057)

**Milestone**: Phase 11T — False Negative Analysis

---


## 1. Safety Metric Summary

- **True NON_COMPLIANT Products in Benchmark**: 3

- **Predicted NON_COMPLIANT Products**: 3

- **False Negatives (Missed Violations)**: **0**

- **NON_COMPLIANT Recall**: **100.0%**

- **False Negative Rate**: **0.0%**


## 2. Failure Taxonomy & Remediation

| Failure Category | Code | Description | Mitigation in Phase 11 |

| :--- | :--- | :--- | :--- |

| **Image Quality** | TAX_A | Severe blur / low resolution | Low image quality triggers UNCERTAIN status |

| **OCR Failure** | TAX_B | Illegible micro-print stamps | YOLO ROI cropping isolates stamp regions for high-res OCR |

| **YOLO Localization** | TAX_C | Missed tiny text patch | Fallback to full-image OCR if ROI count is zero |

| **Parser Failure** | TAX_D | Non-standard date format (e.g. 11/25) | Expanded regex heuristics in feature extractor |
