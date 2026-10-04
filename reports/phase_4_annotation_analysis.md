# Phase 4 — Pilot Annotation Analysis Report

**Date**: 2026-09-05
**Status**: ANALYSIS COMPLETE

---

## 1. Annotation Completion Status

| Metric | Value |
|---|---|
| Pilot products (manifest) | 75 |
| Total annotation records | 78 |
| Unique products annotated | **75 (100%)** |
| Duplicate records (re-annotations) | 3 |
| Missing annotations | 0 |
| JSON parse errors | 0 |
| Schema versions | v1.0.0 (7), v1.1.0 (68) |
| Draft-assisted annotations | 68 (91%) |
| Manual-only annotations | 7 (9%) |
| Annotators | 1 (annotator_01) |

**Status**: ✅ ALL 75 PRODUCTS ANNOTATED

---

## 2. Annotation Quality Audit

| Issue Type | Count | Severity |
|---|---|---|
| `NON_COMPLIANT_ALL_PRESENT` | 36 | 🔴 HIGH — overall label inconsistent with declaration labels |
| `PRESENT_NO_EVIDENCE` | 420 | 🟡 MODERATE — evidence sources array empty for PRESENT labels |
| Invalid labels | 0 | ✅ None |
| Missing declarations | 0 | ✅ None |

### Critical Finding: 36 Inconsistent Overall Labels

36 products (48%) have ALL declarations marked PRESENT but the overall label is NON_COMPLIANT. Root cause: the human annotator modified individual declaration labels (upgrading from machine draft) but did not update the overall assessment, which retained the machine draft's NON_COMPLIANT.

**These 36 records need human re-review** before they can be used for ML training.

Flagged products saved to: `annotations/review_required.csv`

---

## 3. Overall Label Distribution

| Label | Count | Percentage |
|---|---|---|
| COMPLIANT | 2 | 2.7% |
| NON_COMPLIANT | 73 | 97.3% |
| UNCERTAIN | 0 | 0.0% |

### Analysis

- **Extreme class imbalance**: 97.3% NON_COMPLIANT vs. 2.7% COMPLIANT
- **Zero UNCERTAIN**: Annotator never used UNCERTAIN, which may indicate over-confidence or annotation style
- **COMPLIANT is rare**: Only 2 products were deemed compliant — too few for meaningful binary classification
- **The 36 inconsistent records, if corrected, could increase COMPLIANT to up to 38 (50.7%)** — this would dramatically change the class balance

---

## 4. Declaration-Level Statistics

| Declaration | PRESENT | ABSENT | PARTIAL | N/A | CANNOT_DETERMINE |
|---|---|---|---|---|---|
| D01 Manufacturer | 63 (84%) | 5 (7%) | 7 (9%) | 0 | 0 |
| D02 Generic Name | **75 (100%)** | 0 | 0 | 0 | 0 |
| D03 Net Quantity | 66 (88%) | 7 (9%) | 2 (3%) | 0 | 0 |
| D04 Mfg Date | 68 (91%) | 0 | 0 | 0 | 7 (9%) |
| D05 MRP | **75 (100%)** | 0 | 0 | 0 | 0 |
| D06 Consumer Care | 44 (59%) | **29 (39%)** | 0 | 0 | 2 (3%) |
| D07 Country of Origin | 68 (91%) | 0 | 0 | 0 | 7 (9%) |

### Key Findings

1. **D02 and D05 are always PRESENT** — no predictive value for these (constant features)
2. **D06 is the most variable** — 59% PRESENT vs 39% ABSENT. This is the strongest candidate for ML prediction
3. **D01 has useful variability** — 84% PRESENT, 7% ABSENT, 9% PARTIAL
4. **D03 has some variability** — 88% PRESENT, 9% ABSENT
5. **D04 and D07 are surprisingly PRESENT** at 91% — this differs from Phase 2 expectations. Requires verification of evidence quality

---

## 5. Evidence Source Analysis

| Declaration | Most Common Sources |
|---|---|
| D01 | BRAND_FIELD (7), LISTING_TITLE (3), SPECIFICATION (2), DESCRIPTION (2) |
| D02 | LISTING_TITLE (7), DESCRIPTION (2), SPECIFICATION (1) |
| D03 | SPECIFICATION (31), DESCRIPTION (24), LISTING_TITLE (10) |
| D04 | LISTING_TITLE (5), DESCRIPTION (5), SPECIFICATION (5) |
| D05 | PRICE_FIELD (7) |
| D06 | DESCRIPTION (8), SPECIFICATION (8), LISTING_TITLE (6) |
| D07 | SPECIFICATION (5), DESCRIPTION (4) |

### Key Insight

Evidence source tracking shows **low fill rates** — most draft-assisted annotations did not explicitly select evidence sources (420 PRESENT_NO_EVIDENCE issues). Only the 7 manual v1.0.0 annotations consistently recorded sources.

**Implication**: Evidence source data is unreliable for source importance analysis. Future annotation should enforce source selection.

---

## 6. Draft Modification Analysis

| Declaration | Accepted | Modified | Modification Rate |
|---|---|---|---|
| D01 | 5 | 63 | **93%** |
| D02 | 68 | 0 | 0% |
| D03 | 50 | 18 | 26% |
| D04 | 0 | 68 | **100%** |
| D05 | 68 | 0 | 0% |
| D06 | 22 | 46 | **68%** |
| D07 | 1 | 67 | **99%** |
| Overall | 66 | 2 | 3% |

### Analysis

- **D04, D07**: Nearly 100% modified — machine draft (CANNOT_DETERMINE) was systematically changed to PRESENT. This suggests the annotator found evidence the draft engine missed, or applied different criteria
- **D01**: 93% modified — machine draft (PARTIAL/ABSENT) upgraded to PRESENT. Same pattern
- **D06**: 68% modified — significant human correction, but ABSENT was accepted in 22 cases
- **D02, D05**: 0% modified — machine draft perfectly aligned with human judgment
- **Overall**: Only 3% modified — BUT this is the core quality issue (36 records where declarations were changed but overall was not)

---

## 7. Image Feasibility

| Metric | Value |
|---|---|
| Products tested | 25 |
| Accessible (HTTP 200) | **24 (96%)** |
| Inaccessible (HTTP 404) | 1 (4%) |
| File format | JPEG (23), WebP (1) |
| File size | 24 KB — 231 KB (median: 129 KB) |
| Image dimensions | 220×551 — 1299×3149 |
| URLs per product | 1 — 4 (median: 2) |

**Images are available** and high-resolution enough for OCR. Full report: `reports/image_download_feasibility.md`

---

## 8. Data Leakage Risks

| Risk | Severity | Status |
|---|---|---|
| Rule-engine labels as ML targets | 🔴 CRITICAL | ✅ Mitigated (human labels used) |
| Overall label as input feature | 🔴 CRITICAL | ⚠️ Needs attention at feature engineering |
| Draft labels contaminating ground truth | 🟡 MODERATE | ⚠️ 36 records affected |
| Duplicate products in train/test | 🟡 MODERATE | ⚠️ Needs attention at training |
| Anchoring bias from draft suggestions | 🟡 MODERATE | ⚠️ Known risk |

Full audit: `reports/data_leakage_audit.md`

---

## 9. Category Distribution

| Category | Count |
|---|---|
| Beauty and Personal Care | 29 |
| Kitchen & Dining | 26 |
| Baby Care | 20 |

---

## 10. Confidence Distribution

| Level | Count |
|---|---|
| MEDIUM | 75 (100%) |

All annotations have MEDIUM confidence — annotator did not vary this field.

---

## 11. Feature Signal Assessment

### Features with Predictive Signal

| Feature Group | Signal Strength | Rationale |
|---|---|---|
| D06 (Consumer Care) evidence | **HIGH** | 59/39/2 split — strongest discriminator |
| D01 (Manufacturer) evidence | MEDIUM | 84/7/9 split — some variability |
| D03 (Net Quantity) evidence | MEDIUM | 88/9/3 split |
| Brand presence | MEDIUM | 46.8% missing in candidate pool |
| Specification key count | LOW-MEDIUM | Correlates with information richness |
| Description length | LOW | May correlate weakly with listing quality |

### Features with No Predictive Signal

| Feature | Reason |
|---|---|
| D02 (Generic Name) | 100% PRESENT — constant |
| D05 (MRP) | 100% PRESENT — constant |
| Category | Only 3 categories in pilot — insufficient diversity |
| Confidence | 100% MEDIUM — constant |

---

## Deliverables Created

| File | Contents |
|---|---|
| `reports/pilot_label_distribution.csv` | Overall label counts |
| `reports/declaration_statistics.csv` | Per-declaration label breakdown |
| `reports/evidence_source_analysis.csv` | Evidence source usage by declaration |
| `reports/image_download_summary.csv` | Per-product image test results |
| `reports/image_download_feasibility.md` | Image collection strategy |
| `reports/data_leakage_audit.md` | 8 leakage risks with mitigations |
| `reports/pilot_edge_cases.md` | 7 edge case categories |
| `reports/pilot_edge_cases_data.csv` | Raw edge case data |
| `annotations/review_required.csv` | 456 review items (36 critical) |
| `src/data/phase4_analysis.py` | Analysis script |
