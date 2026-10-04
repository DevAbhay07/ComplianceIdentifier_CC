I am rebuilding the machine learning prediction layer for my SIH project, NitiMark, from scratch.

IMPORTANT:
This ML project must initially remain independent from the main NitiMark backend. After the ML prediction layer is properly built, tested, and validated, I will integrate it into the main backend.

PROJECT CONTEXT

NitiMark is an AI-powered product compliance verification system for packaged commodities.

The complete NitiMark system includes:

1. Product image upload
2. Product URL analysis
3. Bulk URL analysis
4. Product metadata extraction
5. Gemini Vision / OCR for packaging information extraction
6. Machine learning prediction layer
7. Deterministic legal rule validation
8. RAG-based legal evidence retrieval
9. AI chatbot
10. Chrome extension
11. Geographic heatmap and repeat violator tracking
12. Compliance dashboard and reports

The ML model being built now is only one component of the larger hybrid system.

==================================================
THE NEW ML GOAL
==================================================

I want to build a scientifically defensible real-world compliance prediction model.

My previous ML model had a weakness:
features were passed into Python compliance rules,
the rules generated labels,
and then ML was trained to reproduce those labels.

I do NOT want to repeat that approach.

The new pipeline should use:

1. Official Legal Metrology rules as the legal ground-truth framework
2. Flipkart product records as real-world product samples
3. Product metadata and specifications
4. Product images and OCR/Vision extraction where possible
5. Human-reviewed annotations for at least part of the dataset
6. Strict train/validation/test evaluation without data leakage

The official legal documents are NOT directly the ML dataset.
They should be converted into a structured legal requirements schema.

The Flipkart dataset is NOT automatically a compliance dataset.
It should be treated as a source of real product records.

==================================================
DATA SOURCES
==================================================

Source 1:
Official Legal Metrology (Packaged Commodities) Rules, 2011 and relevant amendments.

Purpose:
Create the legal requirement framework.

Source 2:
Flipkart e-commerce dataset.

Expected fields may include:

- product_name
- product_category_tree
- brand
- description
- product_specifications
- retail_price
- image
- product_url
- product ID

Purpose:
Use real-world products as samples.

==================================================
TARGET ARCHITECTURE
==================================================

OFFICIAL LEGAL RULES +
REAL PRODUCT RECORDS +
PRODUCT IMAGES / OCR +
HUMAN REVIEWED ANNOTATIONS
↓
REAL-WORLD COMPLIANCE DATASET
↓
TRAIN / VALIDATION / TEST
↓
ML PREDICTION MODEL
↓
LATER INTEGRATION WITH HYBRID NITIMARK SYSTEM

The final production architecture will be:

Product Image / Product URL
↓
Metadata + Image Extraction
↓
Structured Features
↓
┌──────┼───────┐
↓ ↓ ↓
ML Rules RAG
Model Engine Evidence
└──────┼───────┘
↓
Hybrid Decision
↓
Compliance Report

==================================================
FEATURE ENGINEERING
==================================================

Do not use only simple binary features.

Create multiple feature groups.

A. LISTING METADATA FEATURES

Examples:

- brand_available
- description_length
- specification_count
- price_available
- category
- subcategory
- manufacturer_info_available

B. IMAGE / OCR FEATURES

Examples:

- ocr_text_length
- net_quantity_detected
- mrp_detected
- manufacturer_detected
- importer_detected
- country_origin_detected
- customer_care_detected
- date_detected
- batch_detected

C. CONFIDENCE FEATURES

Do not only store:

mrp_present = 1

Also store:

mrp_present = 1
mrp_confidence = 0.91

Possible confidence features:

- brand_confidence
- mrp_confidence
- net_quantity_confidence
- manufacturer_confidence
- country_origin_confidence
- overall_ocr_confidence
- image_quality_score

D. CROSS-SOURCE VERIFICATION

Compare product listing information against information extracted from the product image.

Examples:

- brand_match
- quantity_match
- price_match
- manufacturer_match
- country_match
- metadata_image_conflict_score

Example:

Listing says brand = Samsung
Image extraction says brand = Samsung

brand_match = 1

If sources conflict:

cross_source_conflict_score increases.

==================================================
LABELING STRATEGY
==================================================

Do not generate all ML labels directly from the same Python rules used as features.

Create a human-reviewed dataset.

Each reviewed product should include:

- product metadata
- product image if available
- applicable legal requirements
- extracted declarations
- compliance decision
- reason for the decision

Possible labels:

COMPLIANT
NON_COMPLIANT
UNCERTAIN

Do not force uncertain cases into a false binary label.

The annotation structure should ideally include:

overall_label

and individual checks such as:

- brand_compliant
- mrp_compliant
- net_quantity_compliant
- manufacturer_compliant
- country_origin_compliant
- customer_care_compliant

==================================================
MODELS
==================================================

Start by comparing:

1. Logistic Regression
   Purpose: baseline

2. Random Forest
   Purpose: primary tree-based model

3. XGBoost
   Purpose: advanced gradient boosting model

Do not add unnecessary models initially.

==================================================
DATA SPLITTING
==================================================

Use a strict evaluation methodology.

Target split:

70% Training
15% Validation
15% Test

Prevent product leakage.

Use grouping based on normalized product title or product family where appropriate.

Possible tools:

- GroupShuffleSplit
- StratifiedGroupKFold

The same or highly similar product should not appear in both training and test data.

Keep a final locked test set of approximately 200–300 products.

Do not use this locked test set during tuning.

==================================================
EVALUATION
==================================================

Do not chase a specific accuracy target.

Report honest results.

Measure:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Class-specific metrics

Evaluate different scenarios where possible:

A. Listing metadata only

B. Image/OCR extracted features

C. Hybrid listing + image features

Compare whether cross-source information improves performance.

==================================================
IMPORTANT DEVELOPMENT RULES
==================================================

1. Work step by step.
2. Do not generate the complete project blindly in one response.
3. Before writing major code, inspect the actual dataset columns.
4. Do not assume dataset fields.
5. Explain every decision.
6. Keep all code modular.
7. Use reproducible random seeds.
8. Log experiments.
9. Preserve raw data.
10. Never modify the test set to improve accuracy.
11. Never fabricate labels or results.
12. Clearly distinguish:

- legal rule validation
- ML prediction
- OCR extraction
- human-reviewed ground truth
