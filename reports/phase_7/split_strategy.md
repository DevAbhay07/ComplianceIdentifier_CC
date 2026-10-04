# Group-Aware Dataset Splitting Strategy (Phase 7)

## 1. Split Philosophy & Group Isolation
In e-commerce datasets, multiple listings often represent color, size, pack, or packaging variants of the exact same underlying commodity. If variants are randomly allocated across Train and Test splits, near-identical titles and descriptions leak across partitions, producing artificially inflated test scores.

To prevent this:
1. **Clustering Key (`product_family_id`):** Formed from `brand + category + normalized title prefix`.
2. **Duplicate Key (`duplicate_group_id`):** Exact and near-exact URL / title clusters.
3. **Partitioning Algorithm:** Two-stage `GroupShuffleSplit` with fixed seed (`42`).

---

## 2. Partition Allocations

| Partition | Product Count | Proportion | Unique Product Families | Isolation Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 1,159 | 69.7% | 1,077 | Preprocessing fitted exclusively on this split |
| **VALIDATION** | 246 | 14.8% | 231 | 0 family overlap with Train or Test |
| **TEST** | 258 | 15.5% | 231 | 0 family overlap with Train or Val |
| **HUMAN REFERENCE** | 75 | Independent | 75 | Completely independent evaluation benchmark |

---

## 3. Preprocessing Workflow

```
       RAW WEAK DATASET (N=1663)
                   │
                   ▼ GroupShuffleSplit(seed=42)
   ┌───────────────┼───────────────┐
   ▼               ▼               ▼
TRAIN (1159)   VAL (246)       TEST (258)
   │
   ▼ fit()
TRANSFORMER
(Imputer + Scaler + OHE + TF-IDF)
   │
   ├───────────────► transform(TRAIN)        ──► X_train.csv
   ├───────────────► transform(VALIDATION)   ──► X_val.csv
   ├───────────────► transform(TEST)         ──► X_test.csv
   └───────────────► transform(HUMAN_REF)    ──► X_human_ref.csv
```

---

## 4. Verification
- **Group Overlap:** Confirmed **0 overlapping groups** across Train, Validation, and Test.
- **Product ID Overlap:** Confirmed **0 overlapping product IDs** between human reference and weak training sets.
- **Reproducibility:** Splitting logic is fully deterministic using `seed=42`.
