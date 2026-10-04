# ML Dataset Statistics & Dimensionality Report (Phase 7)

**Release Tag:** `ml_dataset_v1`  
**Grouping Key:** `product_family_id`  

## 1. Partition Size & Group Breakdown

| Partition | Product Count | Percentage | Unique Product Families | Unique Duplicate Groups |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 1,159 | 69.7% | 1,077 | 1,159 |
| **VALIDATION** | 246 | 14.8% | 231 | 246 |
| **TEST** | 258 | 15.5% | 231 | 258 |
| **TOTAL WEAK DATASET** | 1,663 | 100.0% | 1,539 | 1,663 |
| **HUMAN REFERENCE** | 75 | Independent | 1 | 1 |

## 2. Feature Representation Breakdown

| Feature Sub-Group | Feature Count | Description |
| :--- | :--- | :--- |
| **Numeric Structural Features** | 21 | Lengths, token counts, price, and binary flags |
| **Categorical One-Hot Encodings** | 198 | Category and price tier encodings |
| **TF-IDF Word N-Grams (1,2)** | 300 | Top informative unigram and bigram tokens |
| **TF-IDF Char N-Grams (3,5)** | 200 | Subword and morphological character n-grams |
| **TOTAL (Feature Set B)** | **719** | Recommended full multi-modal representation |
| **TOTAL (Feature Set A)** | **219** | Lightweight tabular baseline |

