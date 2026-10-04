# Phase 11 — Dataset Role Decision & Utilization Strategy

**Project**: NitiMark-ML — Automated Compliance Checker for Legal Metrology (SIH25057)  
**Milestone**: Phase 11B — Dataset Role Decision  
**Audit Reference**: [dataset_audit.md](file:///home/devabhay07/Desktop/MLmodelV2/reports/phase_11/dataset_audit.md) & [dataset_audit.json](file:///home/devabhay07/Desktop/MLmodelV2/reports/phase_11/dataset_audit.json)

---

## 1. Executive Policy & Scientific Principles

Following our empirical findings in Phase 10 and the comprehensive Phase 11 dataset audit, we establish strict, immutable policies regarding dataset utilization:

1. **Zero Ground-Truth Fabrication**: Weak labels, LLM/Vision drafts, and heuristic predictions are **SILVER LABELS** or **PSEUDO-LABELS**. They shall NEVER be represented as human ground truth.
2. **Strict Benchmark Isolation**: The 75-product Human Reference dataset and the 20-product Real Packaging validation dataset are strictly **EVALUATION-ONLY**. Zero training, feature selection, or threshold tuning is permitted on these benchmarks.
3. **No Target Leakage in Supervised Models**: Product-level weak labels were generated via statutory rule heuristics on D01–D07 declarations. Therefore, models must NOT use target-derived D01–D07 states as input features.
4. **Visual Localization Isolation**: YOLOv8 models are strictly restricted to localizing visual bounding boxes (ROIs). YOLO does not output legal compliance verdicts.
5. **Group-Aware Source Image Partitioning**: Because Roboflow datasets contain 3x augmented views per source image, any packaging-domain visual training or cross-validation must group by source image identity to prevent cross-partition image leakage.

---

## 2. Formal Dataset Role Matrix

| Dataset | Primary Role | Training? | Validation? | Test? | Ground Truth? | Domain | Statutory Alignment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Flipkart 20K Raw** | Unlabeled catalog baseline | No | No | No | No | E-commerce catalog | Full catalog text & structured metadata |
| **Flipkart Weak Labels (N=1,663)** | Silver-standard weak supervision | **Yes** (N=1,159) | **Yes** (N=246) | **Yes** (N=258) | **No (Silver)** | E-commerce catalog | Rule 6 statutory declarations (D01–D07) |
| **Human Reference (N=75)** | Locked gold-standard benchmark | **No** | **No** | **Yes (Eval)** | **Yes (Gold)** | E-commerce catalog | Verified Human Adjudicated (SHA-256 Locked) |
| **YOLOv8 Candidate Dataset (N=249)** | Visual packaging ROI detector | **Yes** (N=219 img) | **Yes** (N=20 img) | **Yes** (N=10 img) | **No (Visual)** | Physical Packaging | Bounding boxes for Brand, MRP, Weight, Address, Consumer, Origin |
| **Real Product Packaging (N=20)** | Physical packaging benchmark | **No** | **No** | **Yes (Eval)** | **Yes (Gold)** | Physical Packaging | Complete statutory declarations on physical retail packs |

---

## 3. Detailed Dataset Role Decisions

### A. Flipkart 20K Dataset
- **Role**: Catalog-domain silver dataset for training text and metadata encoders.
- **Allowed Usage**: Supervised training of metadata/text baselines using feature sets A and B (TF-IDF, price, category, brand).
- **Disallowed Usage**: Direct prediction of physical packaging declarations (D04 date, D06 helpline) without packaging visual evidence.
- **Provenance**: Derived via Phase 6 heuristic extraction over 20,000 Flipkart listings.

### B. Human Reference 75 Dataset
- **Role**: Frozen catalog-domain evaluation benchmark.
- **Cryptographic Hash**: `37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520` (Immutable).
- **Allowed Usage**: Benchmarking catalog-domain accuracy and model generalizability across e-commerce product listings.
- **Disallowed Usage**: Any gradient updates, hyperparameter search, or threshold calibration.

### C. YOLOv8 Candidate Dataset
- **Role**: Visual evidence localizer.
- **Class Mappings**:
  - `brand` (151 bboxes) $\rightarrow$ D01 (Product Identity / Brand Name)
  - `address` (33 bboxes) $\rightarrow$ D02 / D07 (Manufacturer & Packer Address)
  - `weight` (61 bboxes) $\rightarrow$ D03 (Net Quantity / Weight)
  - `mrp details` (94 bboxes) $\rightarrow$ D05 (MRP Declarations)
  - `consumer` (24 bboxes) $\rightarrow$ D06 (Consumer Care & Helpline)
  - `origin` (6 bboxes) $\rightarrow$ D07 (Country of Origin)
  - `details` (33 bboxes) $\rightarrow$ General Specification Block
  - `tax` (9 bboxes) $\rightarrow$ Inclusive of all taxes clause
  - `barcode` (71 bboxes) $\rightarrow$ Non-statutory (excluded from legal decision)
- **Allowed Usage**: Training and executing bounding-box detection to extract cropped image regions for OCR/Vision evidence parsing.

### D. Real Product Validation Dataset (N=20)
- **Role**: Gold-standard physical packaging domain evaluation benchmark.
- **Characteristics**: 20 distinct retail consumer products with complete high-resolution physical packaging images, OCR transcriptions, and human-verified ground-truth declarations.
- **Allowed Usage**: Empirical measurement of packaging-domain Macro F1, NON_COMPLIANT Recall, False Positive Rate, and domain shift.
- **Disallowed Usage**: Training or threshold tuning.

---

## 4. Cross-Dataset Overlap & Target Independence

1. **Flipkart vs. Real Products**: ZERO overlap. Real product images were independently photographed from physical retail items.
2. **Flipkart vs. YOLOv8**: ZERO overlap. YOLOv8 images originate from Roboflow retail packaging label datasets.
3. **YOLOv8 vs. Real Products**: ZERO overlap.
4. **Target Independence**: Supervised models in Phase 11 will be trained exclusively on independent metadata, text, OCR tokens, and visual geometry features. No D01–D07 target states are supplied as inputs.
