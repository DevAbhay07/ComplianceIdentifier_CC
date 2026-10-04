# Phase 8 Execution Environment Specification

This document records the exact runtime environment, system dependencies, hardware configuration, and package versions used during Phase 8 multi-label model training and evaluation.

## 1. System & Runtime Information

- **Operating System:** Linux x86_64 (`Ubuntu / Debian kernel`)
- **Python Version:** Python 3.12 (inside virtual environment `venv/`)
- **Execution Timestamp:** September 2026 (Phase 8 Execution)
- **Random Seed / State:** Fixed global seed = `42` across all training runs

## 2. Python Package Versions

| Package | Version | Purpose in Phase 8 |
| :--- | :--- | :--- |
| `scikit-learn` | `1.9.0` | Logistic Regression, Random Forest, Metrics, LabelEncoder |
| `xgboost` | `3.4.1` | Gradient Boosted Decision Trees |
| `pandas` | `3.0.5` | Split data loading, dataset slicing, tabular reporting |
| `numpy` | `2.5.2` | Matrix manipulations, array evaluation, binning |
| `joblib` | `1.6.0` | Model artifact persistence & pipeline serialization |

## 3. Preprocessing & Feature Pipeline Specification

- **Fitted Pipeline:** `models/feature_pipeline_v1.joblib` (frozen in Phase 7)
- **Feature Set A (Lexical / Engineered):** 219 features
- **Feature Set B (Complete Representation):** 719 features (219 Set A + 500 TF-IDF features)
- **Data Partitions:**
  - `Train`: 1,159 products (`data/processed/ml_dataset_v1/train/`)
  - `Validation`: 246 products (`data/processed/ml_dataset_v1/validation/`)
  - `Test`: 258 products (`data/processed/ml_dataset_v1/test/`)
  - `Human Reference`: 75 products (`data/processed/ml_dataset_v1/human_reference/`)
