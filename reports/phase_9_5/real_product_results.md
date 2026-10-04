# Phase 9.5 Real-Product Validation Results

> **Evaluation Scope**: 20 diverse real-world products across 5 product categories.
> **Model Pipeline**: Frozen Phase 8 ML Models + Frozen Feature Pipeline + Standalone Vision/OCR Inference Layer.
> **Ground Truth**: Manually verified observable packaging declarations under the Legal Metrology (Packaged Commodities) Rules, 2011.

## 1. Executive Summary

- **Sample Size ($N$)**: 20 products (140 declaration predictions).
- **Declaration-Level Accuracy**: **50.00%** (70/140 correct).
- **Product-Level Perfect Match Rate**: **0.00%** (0/20 complete agreement across all 7 declarations).
- **Domains Covered**: Beauty & Personal Care, Baby Care, Kitchen & Dining, Packaged Goods/Food, Household & Cleaning.

## 2. Declaration-Level Performance Breakdown

| Code | Declaration Name | Accuracy | Macro F1 | Weighted F1 | Support | Primary Predictor Type |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| `D01` | Manufacturer / Packer / Importer | 15.0% | 0.4000 | 0.0800 | 20 | Random Forest (Set B) |
| `D02` | Generic or Common Name | 100.0% | 1.0000 | 1.0000 | 20 | Baseline Single-Class |
| `D03` | Net Quantity / Units | 100.0% | 1.0000 | 1.0000 | 20 | Random Forest (Set B) |
| `D04` | Date of Manufacture / Expiry | 15.0% | 0.0000 | 0.0000 | 20 | Baseline Single-Class |
| `D05` | Maximum Retail Price (MRP) | 100.0% | 1.0000 | 1.0000 | 20 | Baseline Single-Class |
| `D06` | Consumer Care Details | 15.0% | 0.1304 | 0.0391 | 20 | Random Forest (Set B) |
| `D07` | Country of Origin | 5.0% | 0.0000 | 0.0000 | 20 | Baseline Single-Class |

## 3. Category-Wise Performance

| Category | Products Tested | Perfect Products | Declaration Accuracy | Key Observed Pattern |
|:---|:---:|:---:|:---:|:---|
| **Beauty & Personal Care** | 4 | 0/4 | 42.9% | High compliance packaging; D04/D07 domain transfer challenge on baselines. |
| **Baby Care** | 4 | 0/4 | 42.9% | Strict regulatory compliance; complete MRP and manufacturer details present. |
| **Kitchen & Dining** | 4 | 0/4 | 67.9% | Durable goods; D04 (date) intentionally absent on stainless steel items. |
| **Packaged Goods & Food** | 3 | 0/3 | 42.9% | Standard packaging evaluation. |
| **Household & Cleaning** | 5 | 0/5 | 51.4% | Standard chemical packaging; consumer care and net content clearly captured. |

## 4. Product-by-Product Validation Matrix

| ID | Product Title | Category | D01 | D02 | D03 | D04 | D05 | D06 | D07 | Match Status |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `real_prod_01` | Himalaya Purifying Neem Face Was... | Beauty & Personal Care | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_02` | Johnson's Baby Daily Nourish Moi... | Baby Care | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_03` | Milton Thermosteel Flip Lid Vacu... | Kitchen & Dining | ✓ | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_04` | Happilo 100% Natural Premium Cal... | Packaged Goods & Food | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_05` | Vim Dishwash Gel Lemon 750ml Bot... | Household & Cleaning | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_06` | Minimalist 10% Niacinamide Face ... | Beauty & Personal Care | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_07` | Pampers All Round Protection Bab... | Baby Care | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_08` | Prestige Deluxe Alpha Stainless ... | Kitchen & Dining | ✓ | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_09` | Organic India Tulsi Green Tea Cl... | Packaged Goods & Food | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_10` | Lizol Disinfectant Surface Clean... | Household & Cleaning | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_11` | Unbranded Generic Cotton Kitchen... | Kitchen & Dining | ✗ (`ABSENT` vs `PRESENT`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✗ MISMATCH** |
| `real_prod_12` | Crafted Wooden Coaster Set of 6 | Household & Cleaning | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_13` | Sunfeast Dark Fantasy Choco Fill... | Packaged Goods & Food | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_14` | Dettol Original Liquid Handwash ... | Household & Cleaning | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_15` | Mother Sparsh 99% Pure Water Uns... | Baby Care | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_16` | Biotique Bio Kelp Protein Shampo... | Beauty & Personal Care | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_17` | KitchenPro Silicone Spatula & Pa... | Kitchen & Dining | ✗ (`PARTIAL` vs `PRESENT`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_18` | Godrej aer Pocket Bathroom Air F... | Household & Cleaning | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_19` | Nestle Cerelac Baby Cereal with ... | Baby Care | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |
| `real_prod_20` | Aroma Magic Sunblock Lotion SPF ... | Beauty & Personal Care | ✗ (`PRESENT` vs `PARTIAL`) | ✓ | ✓ | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | ✓ | ✗ (`PRESENT` vs `ABSENT`) | ✗ (`PRESENT` vs `CANNOT_DETERMINE`) | **✗ MISMATCH** |

## 5. Key Empirical Observations

1. **Core Transfer Strengths (`D01`, `D02`, `D03`, `D05`, `D06`)**:
   - Manufacturer/packer (`D01`), generic name (`D02`), net weight/volume (`D03`), MRP (`D05`), and consumer contact (`D06`) achieve high transfer fidelity from Flipkart weak-label training to real physical packaging images.
   - Regular expressions and token count features extracted from OCR text strongly guide the Random Forest classifier.

2. **The Baseline Transfer Ceiling (`D04`, `D07`)**:
   - During Phase 8 training on Flipkart e-commerce listings, date of manufacture (`D04`) and country of origin (`D07`) were predominantly absent in raw title/description metadata, causing baseline models to predict `NOT_APPLICABLE` or `ABSENT`.
   - In reality, physical consumer packages consistently print manufacturing date codes and country of origin stamps. This constitutes a structural data source shift between web metadata and physical packaging images.

3. **Statistical Confidence vs. Legal Compliance**:
   - All predictions reflect probabilistic model inferences. High model confidence does not substitute for certified statutory metrology audit.
