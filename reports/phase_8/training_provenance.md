# Phase 8 Training & Data Provenance Record

This document records the exact provenance, lineage, and cryptographic hashes of all data inputs, feature matrices, models, and artifacts utilized in Phase 8 multi-label model training and experimentation.

## 1. Source Dataset Cryptographic Hashes

| Dataset Description | Relative File Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Human Ground Truth (75)** | `data/processed/pilot_ground_truth_75_v1.jsonl` | `37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520` | **LOCKED & UNMODIFIED** |
| **Weak-Label Dataset (1,663)** | `data/processed/flipkart_weak_labels_v1.jsonl` | `28beec359cd95a88c72c15edc1209d538d392e96e49a8f49441a1b65c02476d4` | **LOCKED & UNMODIFIED** |
| **Feature Pipeline Artifact** | `models/feature_pipeline_v1.joblib` | Generated in Phase 7 | **FROZEN** |

## 2. ML Partition Lineage & Split Sizes

- **Split Manifest:** `data/processed/ml_dataset_v1/ml_dataset_manifest.json`
- **Partitions:**
  - **Training Set:** $N = 1,159$ products (Weak Labels). Used exclusively for fitting model parameters.
  - **Validation Set:** $N = 246$ products (Weak Labels). Used exclusively for hyperparameter comparison and candidate model selection.
  - **Test Set:** $N = 258$ products (Weak Labels). Evaluated strictly once after model selection freeze.
  - **Human Reference Set:** $N = 75$ products (Human Ground Truth). Evaluated strictly once for out-of-sample transfer evaluation.

## 3. Strict Anti-Leakage & Governance Guarantees

1. **Target Separation:** Target variables (`D01_target` - `D07_target`, `overall_compliance_target`) are strictly isolated in `Y_*.csv` and never present in feature matrices $X$.
2. **Rule Confidence & Metadata Isolation:** `rule_confidence` and internal rule engine provenance columns are stored in `meta_*.csv` and excluded from model inputs.
3. **No Ground Truth Contamination:** Human ground truth ($N=75$) was never used in any training step, validation search, or hyperparameter selection.
