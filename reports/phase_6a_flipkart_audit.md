# Phase 6A: Flipkart 20K Dataset Comprehensive Audit

**Date**: 2026-09-05 00:00:03 UTC  
**Source Dataset**: `data/raw/flipkart/flipkart_com-ecommerce_sample.csv`  
**Dataset SHA-256**: `deb60d83b958f7309c1c73272f8ffdcfa01bc89f9c7087e21444adf79de5f74d`  
**Dataset Integrity**: Read-Only / Immutable  

---

## 1. Dataset Overview

| Metric | Value | Percentage of Total |
|---|---|---|
| **Total Rows** | 20,002 | 100.0% |
| **Unique `uniq_id`** | 20,000 | 99.99% |
| **Missing `uniq_id`** | 2 | 0.01% |
| **Duplicate `uniq_id`** | 0 | 0.00% |
| **Unique Product URLs** | 20,000 | 99.99% |
| **Duplicate Product URLs** | 0 | 0.00% |
| **Unique Product Titles** | 12,676 | 63.37% |
| **Normalized Title Collisions** | 7,377 | 36.88% |

---

## 2. Field Completeness & Missingness Audit

| Field | Missing / Null | Present | Completeness % | Quality Notes |
|---|---|---|---|---|
| `uniq_id` | 2 | 20000 | 99.99% | 100% unique primary keys |
| `product_url` | 2 | 20000 | 99.99% | Valid web URLs |
| `product_name` | 2 | 20000 | 99.99% | Critical for commodity naming (D02) |
| `product_category_tree` | 2 | 20000 | 99.99% | Structured hierarchical category strings |
| `brand` | 5866 | 14136 | 70.67% | Brand name useful for D01 partial/presence |
| `description` | 4 | 19998 | 99.98% | Text description for declaration evidence |
| `product_specifications` | 16 | 19986 | 99.92% | Key-value attribute blobs |
| `image` | 5 | 19997 | 99.98% | Array of Flipkart CDN image URLs |
| `retail_price` | 80 | 19922 | 99.60% | Raw retail pricing (D05 MRP context) |
| `discounted_price` | 80 | 19922 | 99.60% | Selling price |

---

## 3. Evidence Quality Distributions

### Description Richness
- **Median Character Length**: 229 characters
- **Mean Character Length**: 430.8 characters
- **Descriptions < 50 chars**: 4 (0.02%)
- **Descriptions < 100 chars**: 25 (0.12%)

### Specification Richness
- **Missing / Empty Specs (< 20 chars)**: 16 (0.08%)
- **Well-Formed Specification Blobs**: 19986 (99.92%)

### Image Availability
- **Zero Images**: 5 (0.02%)
- **Exactly 1 Image**: 1475 (7.37%)
- **Multiple Images (>1)**: 18522 (92.60%)
- **Max Images per Product**: 26

### Pricing Distribution (INR)
- **Retail Price Min**: ₹35.0
- **Retail Price Median**: ₹1040.0
- **Retail Price Max**: ₹571,230

---

## 4. Category Hierarchy & Distribution (Top 25)

| Rank | Top Category | Count | Percentage | Plausible LM Applicability |
|---|---|---|---|---|
| 1 | **Clothing** | 6,198 | 30.99% | VARIABLE (Electronics/Apparel Packaging) |
| 2 | **Jewellery** | 3,531 | 17.65% | MEDIUM (Packaged Commodities / Goods) |
| 3 | **Footwear** | 1,227 | 6.13% | MEDIUM (Packaged Commodities / Goods) |
| 4 | **Mobiles & Accessories** | 1,099 | 5.49% | VARIABLE (Electronics/Apparel Packaging) |
| 5 | **Automotive** | 1,012 | 5.06% | MEDIUM (Packaged Commodities / Goods) |
| 6 | **Home Decor & Festive Needs** | 929 | 4.64% | GENERAL RETAIL |
| 7 | **Beauty and Personal Care** | 710 | 3.55% | HIGH (Direct Packaged Goods) |
| 8 | **Home Furnishing** | 700 | 3.50% | MEDIUM (Packaged Commodities / Goods) |
| 9 | **Kitchen & Dining** | 647 | 3.23% | HIGH (Direct Packaged Goods) |
| 10 | **Computers** | 578 | 2.89% | VARIABLE (Electronics/Apparel Packaging) |
| 11 | **Watches** | 530 | 2.65% | MEDIUM (Packaged Commodities / Goods) |
| 12 | **Baby Care** | 483 | 2.41% | HIGH (Direct Packaged Goods) |
| 13 | **Tools & Hardware** | 391 | 1.95% | MEDIUM (Packaged Commodities / Goods) |
| 14 | **Toys & School Supplies** | 330 | 1.65% | MEDIUM (Packaged Commodities / Goods) |
| 15 | **Pens & Stationery** | 313 | 1.56% | MEDIUM (Packaged Commodities / Goods) |
| 16 | **Bags, Wallets & Belts** | 265 | 1.32% | MEDIUM (Packaged Commodities / Goods) |
| 17 | **Furniture** | 180 | 0.90% | GENERAL RETAIL |
| 18 | **Sports & Fitness** | 166 | 0.83% | MEDIUM (Packaged Commodities / Goods) |
| 19 | **Cameras & Accessories** | 82 | 0.41% | VARIABLE (Electronics/Apparel Packaging) |
| 20 | **Home Improvement** | 81 | 0.40% | GENERAL RETAIL |
| 21 | **Health & Personal Care Appliances** | 43 | 0.21% | HIGH (Direct Packaged Goods) |
| 22 | **Sunglasses** | 40 | 0.20% | GENERAL RETAIL |
| 23 | **Gaming** | 35 | 0.17% | GENERAL RETAIL |
| 24 | **Pet Supplies** | 30 | 0.15% | GENERAL RETAIL |
| 25 | **Home & Kitchen** | 24 | 0.12% | GENERAL RETAIL |

---

## 5. Key Findings for Phase 6A Candidate Sampling

1. **Massive Candidate Pool**: The 20,002 rows provide an ample pool to extract ~1,000–2,000 candidates and 275–425 new human annotations.
2. **High Metadata Completeness**: Descriptions (100.0%), specifications (99.9%), and images (100.0%) are well-populated.
3. **Category Breadth**: In addition to Beauty, Baby Care, and Kitchen, strong representation exists in Home Furnishing (700), Automotive (1,012), Footwear (1,227), Watches (530), Tools & Hardware (391), Pens & Stationery (313), and Toys (330).
4. **Duplicate Control Required**: 7,377 normalized title collisions exist across listings (e.g. variants, color/size duplicates). Candidate selection must strictly filter and group near-duplicates.
5. **Zero Pilot Overlap Enforcement**: The 75 pilot IDs must be excluded before building the candidate pool.
