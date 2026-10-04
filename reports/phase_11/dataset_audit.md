# Phase 11 — Comprehensive Dataset Audit Report

**Project**: NitiMark-ML — Automated Compliance Checker (SIH25057)

**Milestone**: Phase 11A — Dataset Audit First

---

## 1. Multi-Dataset Audit Summary

| Dataset | Domain | Samples | Split Status | Provenance / Role | Target Leakage Risk |

| :--- | :--- | :--- | :--- | :--- | :--- |

| **Flipkart 20K (Weak)** | E-Commerce Catalog | 1663 | Train: 1159, Val: 246, Test: 258 | Silver / Heuristic Rule Extraction | **HIGH** if D01-D07 labels used as inputs |

| **YOLOv8 Candidate** | Retail Packaging Visuals | 249 img (90 unique) | Train: 219, Val: 20, Test: 10 | Visual Localization / Packaging ROIs | **ZERO** (Unsupervised for compliance label) |

| **Human Reference** | E-Commerce Catalog | 75 | Locked Benchmark (N=75) | Gold Standard Human Adjudicated | **ZERO** (Evaluation-only, SHA-256 Locked) |

| **Real Product Packaging** | Physical Retail Packaging | 20 | Isolated Evaluation (N=20) | In-Store Ground Truth Packaging | **ZERO** (Evaluation-only) |


## 2. YOLOv8 Candidate Dataset In-Depth Audit

- **Dataset Path**: `data/external/yolo_candidate_dataset`

- **Total Images**: 249 (90 unique source images across 3x Roboflow augmentations)

- **Total Bounding Boxes**: 482

- **Mean Bounding Boxes per Image**: 1.94

- **Split Leakage Audit**: `has_split_leakage=True` (Source image overlap detected: train-val=4, train-test=2, val-test=0)


### Class Relevance Taxonomy to Legal Metrology Declarations (D01–D07)

| Class ID | Class Name | Statutory Relevance | Legal Metrology Mapping (Rule 6) |

| :--- | :--- | :--- | :--- |

| 0 | `address` | **DIRECTLY_RELEVANT** | D02 (Manufacturer/Packer Address), D07 (Country/Origin Address) |

| 1 | `barcode` | **NOT_RELEVANT** | EAN/Barcode (Not a statutory Legal Metrology text declaration) |

| 2 | `brand` | **DIRECTLY_RELEVANT** | D01 (Product Identity / Brand Name) |

| 3 | `consumer` | **DIRECTLY_RELEVANT** | D06 (Consumer Care / Helpline Details) |

| 4 | `details` | **POTENTIALLY_RELEVANT** | General Declaration Block / Specifications |

| 5 | `mrp details` | **DIRECTLY_RELEVANT** | D05 (Maximum Retail Price / MRP) |

| 6 | `origin` | **DIRECTLY_RELEVANT** | D07 (Country of Origin Declaration) |

| 7 | `tax` | **POTENTIALLY_RELEVANT** | D05 Support (Inclusive of all taxes clause) |

| 8 | `weight` | **DIRECTLY_RELEVANT** | D03 (Net Quantity / Net Weight) |



### Bounding Box Geometry Statistics

- **Normalized Width**: Min=0.0258, Mean=0.2638, Median=0.2230, Max=0.9547

- **Normalized Height**: Min=0.0172, Mean=0.1909, Median=0.1441, Max=0.8156

- **Normalized Area**: Min=0.0007, Mean=0.0543, Median=0.0366, Max=0.5042


## 3. Target Provenance & Anti-Leakage Verification

> [!IMPORTANT]

> **Critical Leakage Rule**: In the Flipkart weak-label dataset, product-level `NON_COMPLIANT` vs `COMPLIANT` labels were derived using statutory rule heuristics over D01–D07 declarations. Therefore, **raw D01–D07 target states CANNOT be fed as input features to product-level supervised ML models**. Instead, models must strictly receive independent raw metadata, text TF-IDF/embeddings, OCR transcription features, and YOLO visual bounding box metrics.


## 4. Benchmark Isolation & Cryptographic Integrity

- **Human Reference SHA-256**: `37dc62a308a67899ee40d782b2163d56b6c472edc47b6f56059048649b877520` (Verified match against locked hash: `True`)

- **Real Product Validation Isolation**: 20 physical packaging images verified isolated. Zero sample overlap with training datasets.
