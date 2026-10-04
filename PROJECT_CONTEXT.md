# NitiMark — Project Context

## 1. Project Overview

NitiMark is an AI-powered product compliance verification system designed to help identify potential compliance issues in packaged commodity product listings.

The system combines product listing information, packaging images, OCR/Vision extraction, machine learning, deterministic legal validation, and regulatory evidence retrieval.

NitiMark is designed as a hybrid compliance intelligence system rather than relying on a single AI model.

---

# 2. Problem Context

E-commerce platforms contain a large number of product listings and packaged commodities.

Checking whether a product listing and its packaging contain required declarations can be difficult because information may be distributed across:

- Product titles
- Product descriptions
- Product specifications
- Product images
- Packaging labels
- Seller information
- External product pages

Manual verification can be time-consuming and difficult to scale.

NitiMark aims to assist with the identification, analysis, and reporting of potential compliance issues using AI-assisted automation.

---

# 3. Complete NitiMark System

The complete NitiMark platform is designed around multiple input sources and multiple analysis layers.

## Supported Input Methods

The system is intended to support:

1. Product image upload
2. Product URL analysis
3. Bulk URL analysis
4. Product metadata extraction
5. Product listing analysis

---

# 4. High-Level System Workflow

```text
PRODUCT IMAGE / PRODUCT URL
            │
            ▼
   DATA COLLECTION & EXTRACTION
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
PRODUCT      IMAGE / OCR
METADATA      EXTRACTION
      │           │
      └─────┬─────┘
            │
            ▼
    STRUCTURED FEATURES
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
    ML    RULES   RAG
   MODEL  ENGINE EVIDENCE
     │      │      │
     └──────┼──────┘
            │
            ▼
     HYBRID DECISION
            │
            ▼
    COMPLIANCE REPORT
            │
            ▼
        DASHBOARD
```
