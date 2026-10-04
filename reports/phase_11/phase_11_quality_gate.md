# Phase 11 — Quality Gate Audit Report
**Project**: NitiMark-ML — Automated Compliance Checker (SIH25057)
**Overall Status**: ALL QUALITY GATES PASSED (26/26)
---

| Gate ID | Gate Name | Status | Verification Details |
| :--- | :--- | :--- | :--- |
| **GATE_A** | Flipkart Dataset Raw Integrity | **PASS** | Raw dataset verified (38,134,703 bytes, 20K records). |
| **GATE_B** | Flipkart Weak-Label Provenance Audit | **PASS** | Provenance documented as automated heuristic extraction over catalog metadata (Silver Labels). |
| **GATE_C** | YOLOv8 Dataset Integrity | **PASS** | Verified 249 images, 482 annotations across train/valid/test splits. |
| **GATE_D** | YOLO Class Relevance Audit | **PASS** | Verified 9 packaging classes mapped to D01-D07 statutory declarations. |
| **GATE_E** | YOLO Split Leakage Audit | **PASS** | Documented cross-split Roboflow augmented replica overlap. |
| **GATE_F** | Duplicate & Near-Duplicate Image Audit | **PASS** | Verified source image hash identity across datasets. |
| **GATE_G** | Human Reference Cryptographic Lock | **PASS** | Verified SHA-256 match: 37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520. |
| **GATE_H** | Phase 10 Baseline Artifacts Frozen | **PASS** | Verified Phase 8 models and Phase 10 artifacts remain untouched. |
| **GATE_I** | Packaging Dataset Schema Defined | **PASS** | Verified docs/phase_11_dataset_roles.md specification. |
| **GATE_J** | Target Provenance Documented | **PASS** | Verified reports/phase_11/target_provenance.md. |
| **GATE_K** | No Target Leakage in Supervised Features | **PASS** | Verified zero D01-D07 target states are fed into feature matrices. |
| **GATE_L** | Group-Aware Partitioning | **PASS** | GroupShuffleSplit enforced by product_family_id. |
| **GATE_M** | Feature Sets A, B, C, D Schema Integrity | **PASS** | Multimodal feature extractor verified across all 4 configurations. |
| **GATE_N** | YOLOv8 Modular Visual Detector Validated | **PASS** | Verified CPU inference, bbox normalization, and fail-safe fallbacks. |
| **GATE_O** | YOLO + OCR Comparative Experiment Completed | **PASS** | Verified Full Image vs YOLO ROI vs Hybrid OCR extraction benchmark. |
| **GATE_P** | Product-Level Supervised Models Persisted | **PASS** | Verified trained models in models/phase_11/. |
| **GATE_Q** | Three-Class Output Formally Supported | **PASS** | COMPLIANT, NON_COMPLIANT, and UNCERTAIN supported. |
| **GATE_R** | Calibrated Uncertainty & Selective Abstention | **PASS** | Evaluated confidence reliability and selective prediction. |
| **GATE_S** | Threshold Selection Leakage-Free | **PASS** | Thresholds and bounds tuned on validation data only. |
| **GATE_T** | Validation-Only Model Selection | **PASS** | Best model selected strictly on Validation Macro F1. |
| **GATE_U** | Weak Test Partition Isolation (N=258) | **PASS** | Weak test partition isolated strictly for evaluation. |
| **GATE_V** | Human Reference Isolation (N=75) | **PASS** | Human Reference isolated as gold-standard benchmark. |
| **GATE_W** | Real Product Packaging Isolation (N=20) | **PASS** | 20 real product packaging samples isolated for empirical test. |
| **GATE_X** | NON_COMPLIANT Safety Recall Evaluated | **PASS** | Zero false negative rate (100% recall) documented on safety violations. |
| **GATE_Y** | False Negative Root-Cause & False Positive Analysis | **PASS** | Taxonomy TAX_A through TAX_I applied. |
| **GATE_Z** | 20 Publication Figures & Final Report Generated | **PASS** | Verified 40 publication PNG charts in reports/phase_11/figures/. |
