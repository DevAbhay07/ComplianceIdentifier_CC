# Phase 11 Feature Provenance & Leakage Risk Audit

> **Project**: NitiMark-ML — Standalone ML Prediction Layer  
> **SIH Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms  
> **Phase**: Phase 11 — True Packaging-Domain ML (YOLOv8 + OCR/Vision + Multimodal Feature Engineering)  
> **Strict Anti-Leakage Policy**: ZERO D01–D07 target states or derived compliance decisions are permitted as model input features.

---

## 1. Feature Space Architecture Overview

Phase 11 implements four controlled feature representations:
- **Set A ($D=202$)**: Structured Catalog Metadata + Scaled Numerical Attributes + Category One-Hot Encoding.
- **Set B ($D=702$)**: Set A + Text Word N-Grams ($D=300$) + Character N-Grams ($D=200$).
- **Set C ($D=712$)**: Set B + OCR Text Lengths, Token Densities, OCR Quality Confidence, and 7 Statutory Keyword Matches.
- **Set D ($D=741$)**: Set C + 29 YOLOv8 Visual Packaging Detection Features (9 Class Presences, 9 Class Confidences, 5 Geometry Signals, 6 Aggregate Visual Statistics).

---

## 2. Comprehensive Feature Provenance Matrix

| Feature Group | Feature Name / Pattern | Raw Source | Extraction Method | Derived from Image? | Derived from YOLO? | Derived from OCR? | Derived from Metadata? | Derived from Target? | Leakage Risk Assessment |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Structural** | `retail_price`, `discounted_price` | Flipkart Catalog CSV | Float conversion | NO | NO | NO | YES | **NO** | **ZERO RISK**: Raw commercial price. |
| **Structural** | `discount_ratio` | Flipkart Catalog CSV | `(price - disc) / price` | NO | NO | NO | YES | **NO** | **ZERO RISK**: Pricing geometry. |
| **Structural** | `title_len`, `desc_len`, `spec_len` | Product listing text | Character length | NO | NO | NO | YES | **NO** | **ZERO RISK**: Text volume signal. |
| **Structural** | `has_brand`, `has_image` | Product metadata | Binary string check | NO | NO | NO | YES | **NO** | **ZERO RISK**: Listing completeness. |
| **Categorical** | `category_*` (One-Hot) | Product category hierarchy | `OneHotEncoder(handle_unknown='ignore')` | NO | NO | NO | YES | **NO** | **ZERO RISK**: Product taxonomy. |
| **Text NLP** | `tfidf_word_000` to `tfidf_word_299` | Title, description, specifications | `TfidfVectorizer(max_features=300, ngram=(1,2))` | NO | NO | NO | YES | **NO** | **ZERO RISK**: Raw text frequencies. |
| **Text NLP** | `tfidf_char_000` to `tfidf_char_199` | Title, description, specifications | `TfidfVectorizer(max_features=200, char_wb)` | NO | NO | NO | YES | **NO** | **ZERO RISK**: Sub-word n-gram statistics. |
| **OCR Signals** | `ocr_char_count`, `ocr_word_count` | Vision API / OCR transcription | Length of raw string | YES | NO | YES | NO | **NO** | **ZERO RISK**: Raw transcribed text length. |
| **OCR Signals** | `ocr_confidence` | Vision OCR / Engine confidence | Engine confidence metric | YES | NO | YES | NO | **NO** | **ZERO RISK**: Optical transcription clarity. |
| **OCR Keywords** | `kw_mfg_date` | OCR transcription | Regex: `\b(mfd\|mfg\|packed\|exp)\b` | YES | NO | YES | NO | **NO** | **ZERO RISK**: Keyword token occurrence count. |
| **OCR Keywords** | `kw_mrp_price` | OCR transcription | Regex: `\b(mrp\|rs\.?\|inr\|₹)\b` | YES | NO | YES | NO | **NO** | **ZERO RISK**: Keyword token occurrence count. |
| **OCR Keywords** | `kw_net_quantity` | OCR transcription | Regex: `\b(net\s*wt\|ml\|g\|kg\|pcs)\b` | YES | NO | YES | NO | **NO** | **ZERO RISK**: Keyword token occurrence count. |
| **OCR Keywords** | `kw_consumer_care` | OCR transcription | Regex: `\b(consumer\|care\|helpline)\b` | YES | NO | YES | NO | **NO** | **ZERO RISK**: Keyword token occurrence count. |
| **OCR Keywords** | `kw_origin` | OCR transcription | Regex: `\b(country\s*of\s*origin\|made\s*in)\b` | YES | NO | YES | NO | **NO** | **ZERO RISK**: Keyword token occurrence count. |
| **OCR Keywords** | `kw_manufacturer` | OCR transcription | Regex: `\b(mfd\s*by\|manufactured\s*by)\b` | YES | NO | YES | NO | **NO** | **ZERO RISK**: Keyword token occurrence count. |
| **OCR Keywords** | `kw_commodity_name` | OCR transcription | Regex: `\b(shampoo\|oil\|soap\|pack)\b` | YES | NO | YES | NO | **NO** | **ZERO RISK**: Keyword token occurrence count. |
| **YOLO Presence** | `brand_detected` | `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Presence** | `mrp_detected` | `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Presence** | `net_weight_detected` | `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Presence** | `address_detected` | `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Presence** | `consumer_care_detected` | `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Presence** | `country_origin_detected` | `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Presence** | `product_details_detected`| `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Presence** | `tax_info_detected` | `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Presence** | `barcode_detected` | `yolov8_packaging.pt` | Bounding box class presence | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detector label. |
| **YOLO Conf** | `*_detection_confidence` | `yolov8_packaging.pt` | Model detection confidence score | YES | YES | NO | NO | **NO** | **ZERO RISK**: Raw detection probability. |
| **YOLO Geometry** | `total_region_area`, `normalized_region_area` | YOLO bounding box coords | Sum of normalized bbox areas | YES | YES | NO | NO | **NO** | **ZERO RISK**: Spatial packaging layout. |
| **YOLO Geometry** | `max_region_width`, `max_region_height`, `mean_aspect_ratio` | YOLO bounding box coords | Spatial geometric dimensions | YES | YES | NO | NO | **NO** | **ZERO RISK**: Spatial aspect ratio. |
| **YOLO Aggregate**| `total_detection_count`, `average_detection_confidence`, `maximum_detection_confidence` | Aggregated detections | Array statistics | YES | YES | NO | NO | **NO** | **ZERO RISK**: Visual detection statistics. |
| **YOLO Aggregate**| `number_of_unique_classes`, `visual_evidence_coverage`, `text_region_density` | Class diversity and density | Ratio & count formulas | YES | YES | NO | NO | **NO** | **ZERO RISK**: Packaging region density. |

---

## 3. Strict Non-Leakage Validation

1. **Target Disconnect**:
   - The supervised product-level label `PRODUCT_COMPLIANCE` (`COMPLIANT`, `NON_COMPLIANT`, `UNCERTAIN`) is derived exclusively by statutory rule assessment during ground-truth formation.
   - None of the 741 features in Set D contain D01–D07 target states (`PRESENT`, `ABSENT`, `PARTIAL`, `CANNOT_DETERMINE`, `EXEMPT`).
2. **Preprocessing Isolation**:
   - `TfidfVectorizer`, `StandardScaler`, and `OneHotEncoder` are strictly fitted on the training split ($N=1,159$).
   - Validation ($N=246$), Test ($N=258$), Human Reference ($N=75$), and Real Products ($N=20$) are exclusively transformed using the frozen `feature_extractor_v2.joblib` pipeline.
