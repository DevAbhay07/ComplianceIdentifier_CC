# Phase 12 Real Packaging End-to-End Validation Report

> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations  
> **Benchmark**: 20 Real Physical Packaging Products (`real_product_validation/`)  
> **Execution Date**: September 2026

---

## 1. Executive Summary

The Phase 12 Streamlit inference bridge was executed across all **20 internal real physical packaging validation samples** to verify end-to-end operational integrity, latency performance, and safety metrics.

| Performance Metric | Measured Value | Statutory Target | Status |
|:---|:---:|:---:|:---:|
| **Sample Size ($N$)** | **$N=20$** | 20 Products | PASS |
| **Overall Accuracy** | **100.0%** (20/20) | $\ge 85.0\%$ | **PASS** |
| **NON_COMPLIANT Safety Recall** | **100.0%** (3/3) | $100.0\%$ (Zero Misses) | **PASS** |
| **False Positive Rate (False Alarms)** | **0.0%** (0/17) | $\le 10.0\%$ | **PASS** |
| **Mean Execution Latency** | **145.57 ms** | $\le 3000\text{ ms}$ | **PASS** |

---

## 2. Sample-by-Sample Validation Log

| Sample ID | Product Name | Ground Truth | Predicted Verdict | Confidence | Uncertainty | YOLO Detections | Latency | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `real_prod_01.png` | Himalaya Purifying Neem Face Wash 150ml | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 1172.6 ms | ✅ PASS |
| `real_prod_02.png` | Johnson's Baby Daily Nourish Moisture Lotion 200ml | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 90.7 ms | ✅ PASS |
| `real_prod_03.png` | Milton Thermosteel Flip Lid Vacuum Flask 1000ml | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 101.4 ms | ✅ PASS |
| `real_prod_04.png` | Happilo 100% Natural Premium California Almonds 500g | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 90.5 ms | ✅ PASS |
| `real_prod_05.png` | Vim Dishwash Gel Lemon 750ml Bottle | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 90.1 ms | ✅ PASS |
| `real_prod_06.png` | Minimalist 10% Niacinamide Face Serum 30ml | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 90.3 ms | ✅ PASS |
| `real_prod_07.png` | Pampers All Round Protection Baby Diapers (Large, 64 Count) | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 88.6 ms | ✅ PASS |
| `real_prod_08.png` | Prestige Deluxe Alpha Stainless Steel Pressure Cooker 3L | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 92.0 ms | ✅ PASS |
| `real_prod_09.png` | Organic India Tulsi Green Tea Classic 25 Tea Bags (45g) | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 92.8 ms | ✅ PASS |
| `real_prod_10.png` | Lizol Disinfectant Surface Cleaner Citrus 2 Litres | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 93.7 ms | ✅ PASS |
| `real_prod_11.png` | Unbranded Generic Cotton Kitchen Towel Set (Pack of 4) | **NON_COMPLIANT** | **NON_COMPLIANT** | 90.0% | LOW | 0 | 91.5 ms | ✅ PASS |
| `real_prod_12.png` | Crafted Wooden Coaster Set of 6 | **NON_COMPLIANT** | **NON_COMPLIANT** | 90.0% | LOW | 0 | 92.5 ms | ✅ PASS |
| `real_prod_13.png` | Sunfeast Dark Fantasy Choco Fills 300g (Pack of 4) | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 90.4 ms | ✅ PASS |
| `real_prod_14.png` | Dettol Original Liquid Handwash Refill Pouch 1500ml | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 90.0 ms | ✅ PASS |
| `real_prod_15.png` | Mother Sparsh 99% Pure Water Unscented Baby Wipes (72 Wipes) | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 89.8 ms | ✅ PASS |
| `real_prod_16.png` | Biotique Bio Kelp Protein Shampoo 340ml | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 92.5 ms | ✅ PASS |
| `real_prod_17.png` | KitchenPro Silicone Spatula & Pastry Brush Set | **NON_COMPLIANT** | **NON_COMPLIANT** | 90.0% | LOW | 0 | 89.5 ms | ✅ PASS |
| `real_prod_18.png` | Godrej aer Pocket Bathroom Air Fragrance Petal Crush Pink 10g | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 91.5 ms | ✅ PASS |
| `real_prod_19.png` | Nestle Cerelac Baby Cereal with Milk Wheat Apple 300g | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 88.6 ms | ✅ PASS |
| `real_prod_20.png` | Aroma Magic Sunblock Lotion SPF 30 (100ml) | **COMPLIANT** | **COMPLIANT** | 95.0% | LOW | 0 | 92.5 ms | ✅ PASS |

---

## 3. Scientific Note on Benchmark Scope

> [!NOTE]
> Performance reflects evaluation over the **current 20-product internal benchmark**. It does not constitute a legal certification or a 100% production accuracy claim across unconstrained consumer packaging domains.
