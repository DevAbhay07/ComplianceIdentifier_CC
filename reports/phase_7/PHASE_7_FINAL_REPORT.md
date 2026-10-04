# Phase 7 Final Milestone Report — Leakage-Safe Feature Engineering & ML Dataset Construction

**Project:** NitiMark ML Prediction Layer (SIH25057)  
**Milestone:** Phase 7 — Leakage-Safe Feature Engineering & Dataset Construction  
**Dataset Release Tag:** `ml_dataset_v1` (Version 1.0.0)  
**Master Quality Gate Status:** **`PASS`** (All 17 Quality Gates Cleared)  
**Phase 8 Readiness Status:** **`PASS`** (Ready for Model Training)  

---

## 1. Objective
Phase 7 established a fully leakage-safe, reproducible feature engineering and dataset construction pipeline to prepare the 1,663 weakly labeled Flipkart products (`flipkart_weak_labels_v1`) for supervised multi-label machine learning in Phase 8.

Key guarantees enforced:
- **Zero Human Contamination**: The 75 human ground-truth products (`pilot_ground_truth_75_v1.jsonl`) remain strictly locked and isolated as an independent evaluation reference.
- **Zero Target / Rule Leakage**: Target labels (`D01`–`D07`, `overall_label`), rule engine confidence scores, rule IDs, and quality tiers are strictly excluded from the input feature matrix $X$.
- **Train-Only Preprocessing**: All transformers (Imputer, StandardScaler, OneHotEncoder, TF-IDF vectorizers) are fitted **only on the training split** after a group-aware partition.
- **Group-Aware Splitting**: No product variant or duplicate cluster crosses between Train, Validation, and Test splits.

---

## 2. Input Datasets

1. **Weak-Label Dataset (Primary Training Corpus):**
   - Source: `data/processed/flipkart_weak_labels_v1.jsonl` (1,663 products)
   - Provenance: `AUTOMATED_WEAK_LABEL_PIPELINE_V1` (Phase 6)
   - Quality Tier: 100% `TIER_A_HIGH_CONFIDENCE`
2. **Human Ground Truth Reference (Independent Evaluation Benchmark):**
   - Source: `data/processed/pilot_ground_truth_75_v1.jsonl` (75 products)
   - Provenance: 100% Human Verified (Phase 5)
   - SHA-256: `37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520` (Immutable)

---

## 3. Human Reference Isolation
- Direct product ID overlap between the human reference set (75 items) and the weak training dataset is **0**.
- The human reference set is transformed strictly using the already-fitted training transformer pipeline, ensuring independent out-of-sample evaluation.

---

## 4. Weak-Label Dataset Summary
- The 1,663-product dataset contains automatically generated weak/silver labels and is not equivalent to human-verified ground truth.
- Pre-filtered to eliminate corrupted listings, missing prices, and unparseable specifications.

---

## 5. Feature Engineering Methodology
- Raw independent features are extracted using `extract_raw_features_from_record()`.
- Text attributes (`title`, `description`, `specifications`) are normalized, concatenated, and tokenized into word (1, 2) and char (3, 5) n-gram TF-IDF representations.
- Numeric attributes (lengths, word counts, token counts, prices, discounts) are median-imputed and z-score standardized.
- Categorical attributes (`category`, `price_tier`) are one-hot encoded with `handle_unknown='ignore'`.

---

## 6. Feature Categories & Dimensionality

| Feature Sub-Group | Count | Description |
| :--- | :--- | :--- |
| **Numeric Structural** | 21 | Title/description lengths, word counts, token counts, prices, discount percentages, image counts, binary completeness flags |
| **Categorical One-Hot** | 198 | Root category and price tier one-hot encodings |
| **TF-IDF Word N-Grams** | 300 | Top unigrams and bigrams fitted on training vocabulary |
| **TF-IDF Char N-Grams** | 200 | Character 3-5 n-grams capturing subword patterns |
| **Feature Set B Total** | **719** | **Recommended Multi-Modal ML Input** |
| **Feature Set A Total** | **219** | Lightweight Tabular Baseline (No TF-IDF) |

---

## 7. Excluded Features (Leakage Prevention)
The following fields are strictly quarantined to metadata/target tables and never appear in feature matrices:
- `D01_target` through `D07_target`, `overall_target`
- `D01_confidence` through `D07_confidence`, `overall_confidence`
- `D01_evidence` through `D07_evidence`, `overall_reason`
- `rule_ids`, `label_type`, `quality_tier`, `rule_engine_version`, `provenance`
- `product_id`, `product_url`, `duplicate_group_id`, `product_family_id`

---

## 8. Data Leakage Analysis
- **Audit Status:** **`PASS`** (10/10 checks cleared in `reports/phase_7/feature_leakage_audit.md`).
- Verified zero target columns, zero rule confidence metrics, zero cross-partition group overlaps, and complete imputation without residual NaNs.

---

## 9. Product Family & Duplicate Grouping
- **Total Unique Product Families:** 1,539 families across 1,663 products.
- **Singleton Families (1 product):** 1,438 (93.4%).
- **Multi-Product Variant Clusters:** 101 clusters (6.6%), maximum cluster size = 6 products.
- All product variants are bound to identical partition splits.

---

## 10. Split Strategy & Partition Sizes
Splitting executed via `GroupShuffleSplit(seed=42, group_column='product_family_id')`:

| Partition | Product Count | Proportion | Unique Product Families |
| :--- | :--- | :--- | :--- |
| **TRAIN** | **1,159** | 69.7% | 1,077 |
| **VALIDATION** | **246** | 14.8% | 231 |
| **TEST** | **258** | 15.5% | 231 |
| **HUMAN REFERENCE** | **75** | Independent Benchmark | 75 |

---

## 11. Preprocessing Strategy
- Transformer object `NitiMarkFeaturePipeline` is instantiated and fitted **exclusively on the 1,159 training records**.
- The validation, test, and human reference splits are transformed strictly out-of-sample using `transform()`.
- Pipeline artifact saved to `models/feature_pipeline_v1.joblib` and `data/processed/ml_dataset_v1/feature_pipeline.joblib`.

---

## 12. Target Representation
Multi-label declaration prediction targets ($D01$–$D07$) are preserved with full 5-state categorical semantics:
- `PRESENT`, `ABSENT`, `PARTIAL`, `NOT_APPLICABLE`, `CANNOT_DETERMINE`.
- Overall compliance (`COMPLIANT`, `NON_COMPLIANT`, `UNCERTAIN`) is stored in target tables for downstream legal aggregation.

---

## 13. Class Imbalance Analysis (Training Split, N=1159)
| Declaration | Name | PRESENT | ABSENT | PARTIAL | CANNOT_DET | Imbalance Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | Manufacturer Name & Address | 37.3% | 2.9% | **59.8%** | 0.0% | `MODERATE_IMBALANCE` |
| **`D02`** | Commodity Name | **100.0%** | 0.0% | 0.0% | 0.0% | `SEVERE_IMBALANCE` |
| **`D03`** | Net Quantity | **96.6%** | 3.4% | 0.0% | 0.0% | `MODERATE_IMBALANCE` |
| **`D04`** | Month/Year of Manufacture | 0.7% | 0.0% | 0.0% | **99.3%** | `SEVERE_IMBALANCE` |
| **`D05`** | Maximum Retail Price (MRP) | **100.0%** | 0.0% | 0.0% | 0.0% | `SEVERE_IMBALANCE` |
| **`D06`** | Consumer Care Details | 0.8% | **96.3%** | 2.9% | 0.0% | `SEVERE_IMBALANCE` |
| **`D07`** | Country of Origin | 0.9% | 0.0% | 0.0% | **99.1%** | `SEVERE_IMBALANCE` |

---

## 14. Missingness Profiling
- Handled natively in pipeline via `SimpleImputer(strategy='median')` and binary presence flags (`has_description`, `has_brand`, `has_price`, `has_image`, `has_specifications`).
- Zero residual NaNs in exported feature matrices.

---

## 15. Exported ML Datasets
Directory: `data/processed/ml_dataset_v1/`
- `train/`: `X_train.csv` (1159x719), `X_train_set_a.csv` (1159x219), `Y_train.csv` (1159x8), `metadata_train.csv` (1159x8)
- `validation/`: `X_val.csv` (246x719), `X_val_set_a.csv` (246x219), `Y_val.csv` (246x8), `metadata_val.csv` (246x8)
- `test/`: `X_test.csv` (258x719), `X_test_set_a.csv` (258x219), `Y_test.csv` (258x8), `metadata_test.csv` (258x8)
- `human_reference/`: `X_human_ref.csv` (75x719), `X_human_ref_set_a.csv` (75x219), `Y_human_ref.csv` (75x8), `metadata_human_ref.csv` (75x8)
- Manifest: `ml_dataset_manifest.json` with file-level SHA-256 checksums.

---

## 16. Unit Test Results
- Suite: `src/features/test_phase_7.py`
- Executed: **15 tests**
- Result: **`Ran 15 tests in 0.215s — OK`**

---

## 17. Master Quality Gate Results
- Evaluated: **17 Quality Gates** (GATE_A through GATE_Q in `reports/phase_7/PHASE_7_QUALITY_GATE.md`)
- Status: **`PASS` (17/17 Gates Cleared)**

---

## 18. Limitations
1. TF-IDF feature vocabulary is bounded to top 300 words and 200 char n-grams to prevent high-dimensional sparsity on $N=1159$ training rows.
2. The weak labels reflect online listing metadata declarations, not physical label packaging. Physical verification requires visual packaging OCR (Phase 8/9).
3. The 75 human ground-truth records serve strictly as a holdout evaluation benchmark and must not be used to train models.

---

## 19. Phase 8 Readiness Determination

> [!IMPORTANT]
> **PHASE 8 READINESS = PASS**  
> The ML dataset construction, train-only feature engineering, group-aware partitioning, and leakage controls are fully validated. The project is ready for multi-label classifier training in Phase 8.
