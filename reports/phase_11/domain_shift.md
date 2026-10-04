# Phase 11 — E-Commerce Catalog vs Physical Packaging Domain Shift Analysis

**Project**: NitiMark-ML (SIH25057)

**Milestone**: Phase 11W — Domain Shift Analysis

---


## 1. Domain Characteristic Divergence


| Property | E-Commerce Catalog Domain (Flipkart) | Physical Retail Packaging Domain | Domain Shift Impact |

| :--- | :--- | :--- | :--- |

| **Text Character Length (Mean)** | 483.1 chars | 281.6 chars | Packaging text is compact, terse, stamp-formatted |

| **Date of Manufacture (D04)** | ~0% present in metadata | ~85% present on packaging | Catalog ML predicts absence; fails on real packs |

| **Consumer Helpline (D06)** | ~5% present in metadata | ~85% present on packaging | Major cause of catalog false alarms on real products |

| **Visual Background** | Studio white background | Complex glossy/curved 3D packaging | Requires YOLO localization for noise reduction |

| **True Compliance Rate** | 0% compliant (statutory gaps) | 85% compliant (retail packs) | Severe prior probability inversion |


## 2. Quantitative Shift Mitigation

By introducing YOLO visual ROI cropping and optical character recognition patches, Phase 11 directly measures the physical text on the packaging rather than relying on sparse e-commerce listing descriptions, reducing the false positive rate on compliant products.
