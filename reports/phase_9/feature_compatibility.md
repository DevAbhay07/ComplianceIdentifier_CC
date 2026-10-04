# Feature Schema Compatibility Report (Phase 9)

**Project:** NitiMark-ML — Standalone Legal Metrology Compliance Predictor  
**Phase:** Phase 9 — Standalone Inference & Real-Product Validation  
**Date:** September 2026  

---

## 1. Objective

This report documents and verifies the complete feature schema alignment between the training feature pipeline (`models/feature_pipeline_v1.joblib` created in Phase 7 and used in Phase 8) and the real-time inference feature adapter (`src/inference/inference_features.py`).

---

## 2. Verification Summary

| Feature Property | Phase 7/8 Training Pipeline | Phase 9 Inference Adapter | Verification Status |
| :--- | :--- | :--- | :--- |
| **Feature Set A Dimension** | 219 columns | 219 columns | **IDENTICAL (100% MATCH)** |
| **Feature Set B Dimension** | 719 columns | 719 columns | **IDENTICAL (100% MATCH)** |
| **Numeric Structural Columns** | 21 columns | 21 columns | **IDENTICAL (100% MATCH)** |
| **Categorical One-Hot Columns**| 198 columns | 198 columns | **IDENTICAL (100% MATCH)** |
| **TF-IDF Word Features** | 300 columns | 300 columns | **IDENTICAL (100% MATCH)** |
| **TF-IDF Char Features** | 200 columns | 200 columns | **IDENTICAL (100% MATCH)** |
| **Column Ordering** | Deterministic fixed order | Derived via `pipeline.transform()` | **EXACT ORDER PRESERVED** |
| **Data Types** | `float64` / numeric | `float64` / numeric | **COMPATIBLE** |
| **Imputation Strategy** | Median (fitted on train) | Reuses fitted SimpleImputer | **LEAKAGE-FREE REUSE** |
| **Scaling Strategy** | Standard Scaler (train) | Reuses fitted StandardScaler | **LEAKAGE-FREE REUSE** |

---

## 3. Detailed Structural Feature Inventory

### 3.1 Numeric Structural Features (21 Features)
1. `title_length`
2. `title_word_count`
3. `desc_length`
4. `desc_word_count`
5. `spec_count`
6. `spec_text_length`
7. `numeric_token_count`
8. `measurement_token_count`
9. `currency_token_count`
10. `contact_token_count`
11. `origin_token_count`
12. `retail_price`
13. `discounted_price`
14. `discount_amount`
15. `discount_percentage`
16. `image_count`
17. `has_description`
18. `has_specifications`
19. `has_brand`
20. `has_image`
21. `has_price`

### 3.2 Categorical One-Hot Features (198 Features)
- Categorical features: `category` (194 one-hot columns) + `price_tier` (4 one-hot columns: `price_tier_BUDGET`, `price_tier_MID_RANGE`, `price_tier_PREMIUM`, `price_tier_LUXURY`).
- Handled with `handle_unknown="ignore"` to guarantee that unseen real-product categories map safely to all-zero one-hot vectors without runtime exceptions.

### 3.3 Multimodal Text TF-IDF Features (500 Features)
- 300 sublinear TF-IDF word n-grams ($1 \le n \le 2$).
- 200 sublinear TF-IDF character n-grams ($3 \le n \le 5$).
- Applied seamlessly over `title`, `description`, `specifications`, and `raw_ocr_text`.

---

## 4. Anti-Leakage & Safety Guarantees

1. **No Re-Fitting during Inference:** The inference adapter strictly calls `transform()` on the serialized `feature_pipeline_v1.joblib`. Under no circumstances does it invoke `fit()` or `fit_transform()`.
2. **Strict Schema Gatekeeper:** Every inference vector is passed through `validate_inference_feature_schema()` prior to model invocation. Any shape, naming, or dtype deviation halts execution with `INFERENCE_SCHEMA_ERROR`.
