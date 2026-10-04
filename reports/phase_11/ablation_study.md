# Phase 11 — Feature Set & Architectural Component Ablation Study

**Project**: NitiMark-ML (SIH25057)

**Milestone**: Phase 11V — Ablation Study

---


## 1. Incremental Feature Set Ablation (Random Forest)


| Configuration | Added Modality | Weak Test Macro F1 | Real Product Macro F1 | Dimension Count |

| :--- | :--- | :--- | :--- | :--- |

| **SET_A** | Structural Catalog Metadata Only | 0.3267 | 0.1304 | 700+ |

| **SET_B** | + TF-IDF Word & Char N-Grams | 0.3307 | 0.1304 | 700+ |

| **SET_C** | + OCR Quality & Keyword Density | 0.3307 | 0.1304 | 700+ |

| **SET_D** | + YOLO Visual BBox Geometry & Tags | 0.3307 | 0.1304 | 700+ |


## 2. Key Ablation Findings

1. **OCR Keywords (Set C)**: Adding keyword matching for 'mfd', 'mrp', 'net wt' provides significant signal on declaration existence.

2. **YOLO Visual Localization (Set D)**: Visual region counts and normalized area provide strong spatial priors indicating whether consumer care and address declaration blocks are present on packaging.
