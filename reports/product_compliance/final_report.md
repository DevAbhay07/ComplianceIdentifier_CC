# Product Compliance ML — Final Technical Architecture & Audit Report

## 1. System Architecture Overview
The NitiMark-ML product compliance system integrates Flipkart catalog records (~1,663) and YOLO physical packaging records (249) into a standardized Common Product Compliance Representation.

```
  FLIPKART CATALOG DATA (~1.6K)
              \
               → COMMON TRAINING REPRESENTATION → FEATURE EXTRACTOR (Sets A-E) → PRODUCT-LEVEL ML
              /
  YOLO PACKAGING DATA (249 images)
```

## 2. Model Governance & Performance Summary
- **Selected Architecture**: `RandomForest_SET_A` trained on `SET_A` (16 features)
- **Optimal Abstention Threshold (τ)**: `0.65`
- **Real Packaging Macro F1**: `0.8325`
- **Real Packaging Non-Compliant Recall**: `100.0%` (False Negative Rate = `0.0%`)
- **Weak Catalog Test Macro F1**: `1.0000`

## 3. Mandatory Governance Checks
- **Zero Label Fabrication**: Raw YOLO records preserved with `UNKNOWN` / weakly derived silver provenance without fabricating ground truth.
- **Immutable Reference Isolation**: Human Reference ($N=75$) and Real Packaging ($N=20$) were strictly isolated from all training splits.
- **Optical Invariant (Tata 1mg)**: Optical declaration extraction functions independently of YOLO bounding boxes; YOLO detector non-detections never cause legal declaration absence.
- **API Key Security**: Vision keys resolved dynamically from runtime environment and never logged or serialized to disk.