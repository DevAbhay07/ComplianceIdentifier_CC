# Phase 7 Feature Dictionary & Representation Specification

## 1. Overview
This document defines every feature extracted and transformed for supervised machine learning in the NitiMark compliance prediction pipeline.

---

## 2. Feature Definitions & Metadata

| Feature Name | Source | Type | Description | Transformation / Encoding | Leakage Status | Legal Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `title_length` | `PRODUCT_TITLE` | Numeric | Character length of product title | Imputed (median), StandardScaled | `SAFE` | Longer titles frequently embed generic commodity and pack size terms. |
| `title_word_count` | `PRODUCT_TITLE` | Numeric | Word token count in title | Imputed (median), StandardScaled | `SAFE` | Structural text richness signal. |
| `desc_length` | `PRODUCT_DESCRIPTION` | Numeric | Character length of description | Imputed (median), StandardScaled | `SAFE` | Measures listing descriptive depth. |
| `desc_word_count` | `PRODUCT_DESCRIPTION` | Numeric | Word count of description | Imputed (median), StandardScaled | `SAFE` | Descriptive text volume. |
| `spec_count` | `PRODUCT_SPECIFICATIONS` | Numeric | Number of structured spec key-value pairs | Imputed (median), StandardScaled | `SAFE` | Key predictor of specification availability (D01, D03, D06, D07). |
| `spec_text_length` | `PRODUCT_SPECIFICATIONS` | Numeric | Character length of serialized specs | Imputed (median), StandardScaled | `SAFE` | Density of structured attribute information. |
| `numeric_token_count` | `TITLE + DESC + SPECS` | Numeric | Count of numeric digits | Imputed (median), StandardScaled | `SAFE` | Density of quantitative numbers (prices, quantities, dates). |
| `measurement_token_count` | `TITLE + DESC + SPECS` | Numeric | Count of measurement expressions | Imputed (median), StandardScaled | `SAFE` | Correlates directly with Net Quantity (D03). |
| `currency_token_count` | `TITLE + DESC + SPECS` | Numeric | Count of currency tokens (₹, Rs, INR) | Imputed (median), StandardScaled | `SAFE` | Correlates with MRP presence (D05). |
| `contact_token_count` | `TITLE + DESC + SPECS` | Numeric | Count of helpline / email / phone patterns | Imputed (median), StandardScaled | `SAFE` | Correlates with Consumer Care (D06). |
| `origin_token_count` | `TITLE + DESC + SPECS` | Numeric | Count of country/origin tokens | Imputed (median), StandardScaled | `SAFE` | Correlates with Country of Origin (D07). |
| `retail_price` | `PRICE_FIELD` | Numeric | Listing retail price in INR | Imputed (median), StandardScaled | `SAFE` | Economic indicator & MRP baseline (D05). |
| `discounted_price` | `PRICE_FIELD` | Numeric | Discounted selling price in INR | Imputed (median), StandardScaled | `SAFE` | Price differentiation indicator. |
| `discount_amount` | `PRICE_FIELD` | Numeric | Absolute discount in INR | Imputed (median), StandardScaled | `SAFE` | Commercial listing characteristic. |
| `discount_percentage` | `PRICE_FIELD` | Numeric | Percentage discount | Imputed (median), StandardScaled | `SAFE` | Commercial listing characteristic. |
| `image_count` | `IMAGE_METADATA` | Numeric | Number of product image URLs | Imputed (median), StandardScaled | `SAFE` | Visual evidence availability. |
| `has_description` | `PRODUCT_DESCRIPTION` | Binary | Whether description is non-empty | Binary [0, 1] | `SAFE` | Fundamental listing completeness. |
| `has_specifications` | `PRODUCT_SPECIFICATIONS` | Binary | Whether specs are non-empty | Binary [0, 1] | `SAFE` | Structured evidence completeness. |
| `has_brand` | `BRAND_FIELD` | Binary | Whether brand field is populated | Binary [0, 1] | `SAFE` | Partial manufacturer evidence (D01). |
| `has_image` | `IMAGE_METADATA` | Binary | Whether $\ge 1$ image URL exists | Binary [0, 1] | `SAFE` | Multi-modal capability flag. |
| `has_price` | `PRICE_FIELD` | Binary | Whether retail price is present | Binary [0, 1] | `SAFE` | Mandatory price requirement (D05). |
| `category_*` | `PRODUCT_CATEGORY` | Categorical | One-hot encoded root categories | `OneHotEncoder(handle_unknown='ignore')` | `SAFE` | Establishes commodity domain and legal requirement applicability. |
| `price_tier_*` | `PRICE_FIELD` | Categorical | One-hot encoded `[BUDGET, MID_RANGE, PREMIUM, LUXURY]` | `OneHotEncoder(handle_unknown='ignore')` | `SAFE` | Market segment categorization. |
| `tfidf_word_*` | `ALL_TEXT` | Sparse / Dense | Top 300 unigram and bigram TF-IDF tokens | Fitted strictly on TRAIN | `SAFE` | Captures domain vocabulary and specific declaration keywords. |
| `tfidf_char_*` | `ALL_TEXT` | Sparse / Dense | Top 200 character 3-5 n-gram TF-IDF tokens | Fitted strictly on TRAIN | `SAFE` | Captures morphological and subword unit patterns robust to typos. |

---

## 3. Excluded Columns Summary

The following fields are strictly **EXCLUDED** from the input feature matrix $X$:
1. `D01_target` through `D07_target` (Target matrix $Y$).
2. `overall_target` (Derived overall compliance).
3. `D01_confidence` through `D07_confidence` and `overall_confidence` (Weak-label generation metadata).
4. `D01_evidence` through `D07_evidence` and `overall_reason` (Post-decision text).
5. `rule_ids`, `label_type`, `quality_tier`, `rule_engine_version`, `provenance` (Pipeline lineage).
6. `product_id`, `product_url`, `duplicate_group_id`, `product_family_id` (Identifiers isolated to metadata).
