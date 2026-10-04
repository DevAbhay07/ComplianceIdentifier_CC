# Pilot Edge Cases — NitiMark Annotation Pilot

**Date**: 2026-09-05
**Phase**: 4 — Pilot Analysis

---

## Overview

This document catalogs edge cases discovered during the 75-product pilot annotation. These inform both the annotation guidelines refinement and the ML model design.

---

## Edge Case Category 1: Inconsistent Overall Labels (36 products)

**Issue**: 36 products have ALL 7 declarations marked PRESENT but the overall label is NON_COMPLIANT. Investigation revealed these annotations used the draft engine's overall label (NON_COMPLIANT, based on D06 being ABSENT) but the human annotator upgraded D06 to PRESENT without also updating the overall label.

**Impact**: HIGH — these 36 records have incorrect overall labels.

**Recommendation**:
- Re-review these 36 products (listed in `annotations/review_required.csv`)
- The annotation tool should consider computing a suggested overall label after all declarations are set
- Add a consistency warning before save

---

## Edge Case Category 2: Consumer Care (D06) Variability (29 ABSENT)

**Issue**: D06 (Consumer Care) is the most frequently ABSENT declaration at 29/75 (39%). However, 44 products were marked PRESENT. The split depends on whether the annotator found any customer care/helpline reference in description or specs.

**Impact**: MEDIUM — this is the single biggest driver of NON_COMPLIANT labels.

**Recommendation**:
- D06 evidence quality should be verified — some "PRESENT" annotations may be based on generic seller info rather than mandatory consumer care details
- Image-based verification would help: packaging labels typically print toll-free numbers

---

## Edge Case Category 3: Manufacturing Date (D04) — Mostly PRESENT (68/75)

**Issue**: In Phase 2, we expected D04 to be almost always CANNOT_DETERMINE from listing data. However, 68/75 (91%) products were annotated PRESENT. The machine draft correctly predicted CANNOT_DETERMINE, but humans modified this to PRESENT.

**Impact**: MEDIUM — suggests annotators may have found date information in specifications or descriptions that the machine draft missed, or may have applied a lenient interpretation.

**Recommendation**: Spot-check a sample of D04=PRESENT annotations to verify the evidence genuinely contains month/year of manufacture.

---

## Edge Case Category 4: Country of Origin (D07) — Mostly PRESENT (68/75)

**Issue**: D07 was expected to be frequently CANNOT_DETERMINE, but 68/75 were marked PRESENT. Similar to D04, humans significantly modified the machine draft (99% modification rate).

**Impact**: LOW-MEDIUM — this may indicate that specifications contain "Country of Origin" fields more often than expected.

**Recommendation**: Verify that D07 evidence comes from explicit "Country of Origin" specification fields.

---

## Edge Case Category 5: Missing Net Quantity (D03) — 7 ABSENT

**Issue**: 7 products have no net quantity information (ABSENT). These are likely non-consumable products (kitchen utensils, tools) where net quantity is expressed as item count rather than weight/volume.

**Impact**: LOW — expected for certain product types.

**Recommendation**: Consider whether "Pack of 1" implicit in single-item products should be treated as PRESENT or ABSENT.

---

## Edge Case Category 6: Missing Manufacturer Info (D01) — 5 ABSENT, 7 PARTIAL

**Issue**: 12 products lack complete manufacturer information. 5 have no brand at all; 7 have a brand name but no manufacturer address.

**Impact**: LOW-MEDIUM — expected pattern. Brand ≠ manufacturer legally.

**Recommendation**: Image OCR may reveal manufacturer labels. Consider a separate annotation pass with images for D01.

---

## Edge Case Category 7: COMPLIANT Products (2 only)

**Issue**: Only 2 products were marked COMPLIANT (2.7%). This extreme imbalance means a binary classifier would achieve ~97% accuracy by always predicting NON_COMPLIANT.

**Impact**: HIGH for ML training — class imbalance is severe.

**Recommendation**:
- Expand the dataset to include more COMPLIANT products
- Consider whether the annotation criteria are too strict
- Consider the multi-label approach (predict each declaration independently) instead of overall compliance
- Use stratified sampling or oversampling techniques if training a binary classifier

---

## Summary of Edge Case Counts

| Edge Case | Count | Severity |
|---|---|---|
| Inconsistent overall labels | 36 | HIGH — requires re-review |
| Consumer care ABSENT | 29 | MEDIUM |
| Manufacturing date unexpectedly PRESENT | 68 | MEDIUM — verify evidence |
| Country of origin unexpectedly PRESENT | 68 | LOW-MEDIUM — verify |
| Net quantity ABSENT | 7 | LOW |
| Manufacturer info missing/partial | 12 | LOW-MEDIUM |
| COMPLIANT (rare class) | 2 | HIGH for ML |
