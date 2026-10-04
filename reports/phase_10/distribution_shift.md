# Phase 10 Distribution Shift & Domain Asymmetry Analysis

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Analysis Scope**: Quantifying feature and label divergence between E-Commerce Web Metadata and Physical Retail Packaging Images.

---

## 1. The Core Asymmetry: Web Listings vs. Physical Packaging

| Dimension | E-Commerce Web Catalog (Flipkart 20K) | Physical Retail Packaging Labels |
|:---|:---|:---|
| **Input Source** | HTML product title, description, specifications. | High-resolution packaging photographs / box labels. |
| **Date of Manufacture (`D04`)** | **0.0% Present** (Omitted from web listings). | **>80% Present** (Mandatory batch/expiry stamp). |
| **Consumer Care (`D06`)** | **34.5% Present** (Rarely in listing text). | **>85% Present** (Mandatory consumer helpline/email). |
| **Country of Origin (`D07`)** | **0.0% Present** (Omitted in legacy listings). | **>85% Present** (Mandatory origin stamp). |
| **True Compliance Rate** | **~0.0% Compliant** (Missing mandatory fields). | **~85.0% Compliant** (Standard FMCG packaging compliance). |

---

## 2. Mathematical Impact on Supervised Learning

1. **Training Infeasibility for `COMPLIANT` Class**:
   - Because e-commerce listings systematically omit batch dates, the weak training dataset contains **0 compliant products out of 1,159**.
   - A direct classifier trained on catalog metadata can only learn to predict `NON_COMPLIANT` or `UNCERTAIN`.
2. **The Power of Evidence-Aware Decision Modeling**:
   - The Evidence-Aware Decision Engine bypasses this metadata skew by evaluating internal declaration signals dynamically, allowing it to output `COMPLIANT` as soon as physical packaging evidence (e.g. OCR dates) is supplied.
