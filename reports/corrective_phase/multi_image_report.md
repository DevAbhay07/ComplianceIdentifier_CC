# Multi-Image Packaging Analysis Specification & Results

## Context & Objectives
In compliance screening under Legal Metrology Act (Rule 6), pre-packaged commodities require seven mandatory declarations.
Because physical packaging occupies a 3D geometry, capturing all mandatory declarations typically requires multiple images (e.g. front label, nutrition/statutory back panel, side bar).

---

## Multi-Image Handling Pipeline
1. **Per-Image Independent Processing**:
   - Each uploaded image view is processed independently through image validation, visual localization (YOLO), and optical character recognition (OCR/Vision).
2. **Intermediate Structured Representation**:
   - An intermediate `ProductEvidenceRecord` is constructed per image view.
3. **Cross-View Evidence Fusion**:
   - `EvidenceFusionEngine` reconciles declarations across views.
   - Declarations found in any view are incorporated into the product record.
   - Source images and region coordinates are tracked for auditability.
   - Values differing by more than threshold margins (e.g. different prices or manufacture dates) are flagged as conflicting evidence.
4. **Product-Level Feature Transformation**:
   - Multi-image consistency features (`image_count`, `images_with_evidence`, `cross_image_consistency`, `has_conflicts`) enter the feature matrix (Set E).
5. **Unified Model Prediction**:
   - The product-level model produces a single verdict for the entire physical product.

---

## Multi-View Verification Matrix
| Test Case | View 1 Input | View 2 Input | Fused State | Prediction |
|---|---|---|---|---|
| **Complementary Declarations** | D02 (Name), D03 (Qty) | D01 (Mfg), D05 (MRP), D06 (Care), D07 (Origin) | All 7 Present | `COMPLIANT` |
| **Missing Declaration** | D02, D03 | D01, D06 | D05 (MRP) Absent | `NON_COMPLIANT` |
| **Conflicting MRPs** | MRP ₹999.00 | MRP ₹1099.00 | D05 Conflict | `UNCERTAIN` |
| **Low Quality Captures** | Low resolution (conf 0.15) | Blurry capture (conf 0.20) | Insufficient Evidence | `UNCERTAIN` |
