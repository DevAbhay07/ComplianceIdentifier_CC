# Feature Engineering Readiness & Representation Roadmap (Phase 6)

## 1. Objective & Scope
This report establishes the feature engineering specification for subsequent machine learning model training on the expanded NitiMark dataset (`flipkart_weak_labels_v1` + `pilot_ground_truth_75_v1`).

---

## 2. Multi-Modal Feature Inventory

| Feature Name | Type | Extraction Source | Description & Potential Transformation | Leakage Risk / Flag |
| :--- | :--- | :--- | :--- | :--- |
| `title_length` | Numeric | `PRODUCT_TITLE` | Character count of normalized title | **LOW** (Raw text structural) |
| `title_word_count` | Numeric | `PRODUCT_TITLE` | Token count of title | **LOW** (Raw text structural) |
| `title_has_numbers` | Binary | `PRODUCT_TITLE` | Presence of numeric digits in title | **LOW** |
| `desc_length` | Numeric | `PRODUCT_DESCRIPTION` | Character length of product description | **LOW** |
| `desc_word_count` | Numeric | `PRODUCT_DESCRIPTION` | Word count in product description | **LOW** |
| `spec_count` | Numeric | `PRODUCT_SPECIFICATIONS` | Number of parsed key-value specification pairs | **LOW** |
| `spec_keys_text` | Text | `PRODUCT_SPECIFICATIONS` | Bag-of-words / TF-IDF of specification key names | **MEDIUM** (Avoid literal rule keywords) |
| `brand_present` | Binary | `BRAND_FIELD` | Whether structured brand field is populated | **MEDIUM** (Overlaps with D01 brand rule) |
| `retail_price` | Numeric | `PRICE_FIELD` | Raw retail price (log-transformed: $\log(1 + p)$) | **LOW** |
| `discount_ratio` | Numeric | `PRICE_FIELD` | $1 - (\text{discounted\_price} / \text{retail\_price})$ | **LOW** |
| `price_tier` | Categorical | `PRICE_FIELD` | One-hot encoded `[BUDGET, MID_RANGE, PREMIUM, LUXURY]` | **LOW** |
| `root_category` | Categorical | `PRODUCT_CATEGORY` | Target-encoded or one-hot root category | **LOW** |
| `image_url_count` | Numeric | `IMAGE_METADATA` | Number of provided image URLs | **LOW** |
| `has_image_urls` | Binary | `IMAGE_METADATA` | Whether $\ge 1$ image URL is available | **LOW** |
| `text_embeddings` | Dense Vector | `TITLE + DESC` | Pre-trained Sentence-Transformer / DeBERTa embeddings | **LOW** (Independent neural representation) |

---

## 3. Strict Data Leakage Controls in Feature Engineering

> [!WARNING]
> **Rule-Derived Feature Flagging**:
> Any feature that directly checks for exact regex keywords used by the weak-label generator (such as a hardcoded boolean `has_mrp_regex_match` or `has_helpline_pattern`) creates an artificial identity mapping between feature and target.
>
> **Enforced Architecture**:
> 1. Use generalized lexical representations (TF-IDF, n-grams, character n-grams, dense Sentence-BERT embeddings) rather than duplicating weak-label rule matchers.
> 2. Prohibit target labels (`D01_label` ... `D07_label`, `overall_label`) from entering feature vectors.
> 3. Do not use post-decision evidence fields (`evidence_text`, `evidence_nature`, `confidence`) as predictive features.

---

## 4. Group-Aware Split Strategy

To prevent data contamination from product variants and duplicate listings:

1. **Stratified Group Split**:
   - Split dataset by `product_family_id` (clusters of brand + category + normalized title prefix).
   - Ensures that all color, size, and pack-count variants of a product reside entirely in either `TRAIN`, `VAL`, or `TEST`.
2. **Locked Human Ground-Truth Benchmark**:
   - The 75 human-verified pilot products are retained as an independent, untouched **`HUMAN_REFERENCE_EVAL`** split.
   - Models trained on weak labels must be evaluated against this reference set without ever having seen it during training or rule construction.
