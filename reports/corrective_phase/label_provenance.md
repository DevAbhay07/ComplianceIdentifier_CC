# Dataset & Target Label Provenance Report

## Purpose & Policy
Scientific integrity requires strict isolation of dataset roles and formal documentation of label provenance.
Under the **Corrective Phase Specification**, models must never fabricate gold labels or conflate visual localization bounding boxes with legal compliance ground truth.

## Dataset Provenance Matrix

| Dataset | Domain | Samples | Target / Annotation Type | Provenance Class | Human Verified | Intended Role |
|---|---|---|---|---|---|---|
| **Flipkart Raw Catalog** | E-Commerce Catalog Metadata | 20002 | Unlabeled / Raw text & specs | `RAW_CATALOG` | ❌ NO (Weak/Synthetic) | Evaluation Only |
| **Flipkart Weak Silver Dataset** | E-Commerce Catalog Metadata | 1663 | Rule-derived synthetic silver targets | `WEAKLY_DERIVED_SILVER` | ❌ NO (Weak/Synthetic) | Training & Val |
| **Human Reference Dataset** | E-Commerce Catalog Metadata | 75 | Manual dual-adjudicated human labels | `HUMAN_VERIFIED_CATALOG` | ✅ YES | Evaluation Only |
| **Real Product Packaging Benchmark** | Physical Product Packaging Images | 20 | Physical package image audit | `HUMAN_VERIFIED_PHYSICAL_BENCHMARK` | ✅ YES | Evaluation Only |
| **YOLO Physical Packaging Dataset** | Physical Packaging Images | 249 | 482 bounding boxes across 9 visual classes | `VISUAL_LOCALIZATION_SUPERVISION` | ✅ YES | Training & Val |

## Key Governance Decisions
1. **YOLO Physical Packaging Dataset (249 images, 482 bboxes, 9 classes)**: Used exclusively for packaging-region localization supervision. YOLO classes (e.g. `mrp details`, `weight`, `consumer`) are visual object regions, NOT legal compliance judgments.
2. **Human Reference (75 samples)**: Fully isolated. SHA-256 hash verified as `37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520`. Strictly immutable.
3. **Real Product Benchmark (N=20 physical products)**: Strictly isolated as a final evaluation benchmark. Zero training, zero threshold tuning, zero feature selection.
4. **Scientific Transparency**: The weakly supervised model is explicitly documented as trained on silver catalog rules and evaluated against independent physical benchmarks without inflating claims.