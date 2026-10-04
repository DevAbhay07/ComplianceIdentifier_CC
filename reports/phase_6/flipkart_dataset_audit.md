# Flipkart Dataset Audit Report (Phase 6A)

**Source File:** `data/raw/flipkart/flipkart_com-ecommerce_sample.csv`  
**Total Records:** 20,002  
**Total Columns:** 15  

## 1. Executive Summary

- **Exact Duplicate Rows:** 1
- **Unique Product URLs:** 20,001 (1 duplicate URLs)
- **Unique Uniq IDs:** 20,001
- **Unique Flipkart PIDs:** 19,998
- **Records with Image URLs:** 19,997 (100.0%)
- **Records with Structured Specs:** 19,986 (99.9%)
- **Records with Retail Price:** 19,922 (99.6%)

## 2. Field Missingness Profile

| Column Name | Missing Count | Missing % | Available Count |
| :--- | :--- | :--- | :--- |
| `uniq_id` | 2 | 0.01% | 20,000 |
| `crawl_timestamp` | 2 | 0.01% | 20,000 |
| `product_url` | 2 | 0.01% | 20,000 |
| `product_name` | 2 | 0.01% | 20,000 |
| `product_category_tree` | 2 | 0.01% | 20,000 |
| `pid` | 2 | 0.01% | 20,000 |
| `retail_price` | 80 | 0.4% | 19,922 |
| `discounted_price` | 80 | 0.4% | 19,922 |
| `image` | 5 | 0.02% | 19,997 |
| `is_FK_Advantage_product` | 2 | 0.01% | 20,000 |
| `description` | 4 | 0.02% | 19,998 |
| `product_rating` | 2 | 0.01% | 20,000 |
| `overall_rating` | 2 | 0.01% | 20,000 |
| `brand` | 5,866 | 29.33% | 14,136 |
| `product_specifications` | 16 | 0.08% | 19,986 |

## 3. Legal Metrology Category Relevance

| Relevance Tier | Record Count | Percentage |
| :--- | :--- | :--- |
| **`MEDIUM_RELEVANCE`** | 8,677 | 43.38% |
| **`LOW_RELEVANCE`** | 7,801 | 39.0% |
| **`HIGH_RELEVANCE`** | 1,885 | 9.42% |
| **`UNKNOWN`** | 1,639 | 8.19% |

### Top 20 Root Categories

| Root Category | Count | Relevance Group |
| :--- | :--- | :--- |
| Clothing | 6,198 | `MEDIUM_RELEVANCE` |
| Jewellery | 3,531 | `LOW_RELEVANCE` |
| Footwear | 1,227 | `MEDIUM_RELEVANCE` |
| Mobiles & Accessories | 1,099 | `LOW_RELEVANCE` |
| Automotive | 1,012 | `LOW_RELEVANCE` |
| Home Decor & Festive Needs | 929 | `UNKNOWN` |
| Beauty and Personal Care | 710 | `HIGH_RELEVANCE` |
| Home Furnishing | 700 | `MEDIUM_RELEVANCE` |
| Kitchen & Dining | 647 | `HIGH_RELEVANCE` |
| Computers | 578 | `LOW_RELEVANCE` |
| Watches | 530 | `LOW_RELEVANCE` |
| Baby Care | 483 | `HIGH_RELEVANCE` |
| Tools & Hardware | 391 | `LOW_RELEVANCE` |
| Toys & School Supplies | 330 | `MEDIUM_RELEVANCE` |
| Pens & Stationery | 313 | `LOW_RELEVANCE` |
| Bags, Wallets & Belts | 265 | `LOW_RELEVANCE` |
| Furniture | 180 | `UNKNOWN` |
| Sports & Fitness | 166 | `MEDIUM_RELEVANCE` |
| Cameras & Accessories | 82 | `LOW_RELEVANCE` |
| Home Improvement | 81 | `UNKNOWN` |

## 4. Evidence Richness Statistics

### Descriptions
- Mean Length: 430.8 characters
- Median Length: 229.0 characters
- Empty Descriptions: 4 (0.02%)

### Specifications
- Records with Specs: 19,986 (99.92%)
- Mean Key-Value Pairs: 11.3
- Ruby Hash Format Count: 19,986

### Product Images
- Records with >=1 Image URL: 19,997 (99.98%)
- Mean Images per Product: 3.54
