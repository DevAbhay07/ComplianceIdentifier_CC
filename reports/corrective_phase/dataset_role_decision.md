# Dataset Role & Provenance Decision
> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms  
> **Phase**: Corrective Phase — Declaration-Centric Product Compliance ML  
> **Date**: September 2026  

---

## 1. Strict Dataset Role Separation

To preserve scientific rigor, prevent circular evaluation, and eliminate target leakage, all datasets in the repository are assigned immutable, explicit roles:

```
+----------------------------------------------------------------------------------------------------+
|                                    REPOSITORY DATASET ROLES                                        |
+------------------------------+--------------------+--------------------------------+---------------+
| Dataset                      | Domain             | Annotation & Target Type       | Assigned Role |
+------------------------------+--------------------+--------------------------------+---------------+
| YOLO Physical Packaging      | Physical Packaging | Bounding Boxes (9 Classes)     | VISUAL REGION |
| (249 Images, 482 BBoxes)     | Real Imagery       | NO Compliance Labels           | LOCALIZATION  |
+------------------------------+--------------------+--------------------------------+---------------+
| Flipkart Weak Silver         | E-Commerce Catalog | Rule-Derived D01-D07 Weak      | WEAK-LABEL    |
| (1,663 Products)             | Metadata & Specs   | Silver Labels (Rule-Derived)   | PROTOTYPING   |
+------------------------------+--------------------+--------------------------------+---------------+
| Human Reference              | E-Commerce Catalog | Human-Verified D01-D07         | IMMUTABLE     |
| (75 Products, SHA-256 Locked)| Catalog & Images   | Gold Reference Labels          | CATALOG REF   |
+------------------------------+--------------------+--------------------------------+---------------+
| Real Packaging Benchmark     | Physical Packaging | Human-Verified Real Packaging  | INDEPENDENT   |
| (20 Products, N=20)          | High-Res Photos    | Ground Truth (D01-D07 & Text)  | BENCHMARK     |
+------------------------------+--------------------+--------------------------------+---------------+
```

---

## 2. Dataset-by-Dataset Specifications

### 2.1 YOLO Physical Packaging Dataset (`data/external/yolo_candidate_dataset`)
- **Total Images**: 249 images (`train`: 219, `valid`: 20, `test`: 10).
- **Total Bounding Boxes**: 482 annotated regions.
- **Classes (9)**: `brand` (151), `mrp details` (94), `barcode` (71), `weight` (61), `details` (33), `address` (33), `consumer` (24), `tax` (9), `origin` (6).
- **Role**: **Packaging Visual Localization Supervision**.
- **Crucial Constraint**: These annotations teach *where* packaging zones occur. They do NOT contain product-level compliance labels (`COMPLIANT` / `NON_COMPLIANT` / `UNCERTAIN`). YOLO presence $\neq$ legal compliance; YOLO non-detection $\neq$ legal absence.

### 2.2 Flipkart Weak Dataset (`data/processed/flipkart_weak_labels_v1.jsonl`)
- **Total Samples**: 1,663 catalog products.
- **Domain**: E-commerce catalog text, structured specifications, and prices.
- **Role**: **Weak/Silver Supervision Prototyping**.
- **Crucial Constraint**: Catalog metadata contains distinct domain characteristics from physical packaging. Models trained on catalog text must be evaluated for domain shift when applied to physical packaging images.

### 2.3 Human Reference Dataset (`data/processed/pilot_ground_truth_75_v1.jsonl`)
- **Total Samples**: 75 products.
- **Integrity**: Frozen and immutable. Exact SHA-256 hash: `37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520`.
- **Role**: **Catalog-Domain Human Reference Benchmark**.
- **Crucial Constraint**: Must never be used in training, threshold tuning, or feature selection.

### 2.4 Real Packaging Benchmark (`real_product_validation/`)
- **Total Samples**: 20 physical consumer products across 5 categories.
- **Domain**: Real-world packaging photographs with transcribed observable text and verified declaration statuses.
- **Role**: **Independent Physical Packaging Evaluation Benchmark ($N=20$)**.
- **Crucial Constraint**: Strictly evaluation-only. Never used for model training or tuning. Headline accuracy must explicitly display $N=20$ and avoid claiming production-level certification from a 20-product benchmark.

---

## 3. Scientific Integrity Affirmation

1. No gold compliance labels will be fabricated from detector classes.
2. No training will take place on the evaluation benchmarks ($N=75$ or $N=20$).
3. Zero target leakage will be enforced across all feature matrices.
