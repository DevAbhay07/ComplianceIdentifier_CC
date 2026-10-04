# Final Consistency Audit (Phase 5)

## Overview

This audit evaluates the logical consistency between individual declaration labels (D01–D07) and the overall compliance label across all 75 finalized pilot products in `data/processed/pilot_ground_truth_75_v1.jsonl`.

---

## 1. Legal Consistency Criteria

Under Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011:
1. **`NON_COMPLIANT`**: A product is non-compliant if ANY mandatory declaration is `ABSENT`, or if mandatory details (such as manufacturer address D01 or consumer care D06) are missing/partial without legal exemption.
2. **`COMPLIANT`**: A product is compliant if ALL applicable declarations (D01 through D07) are `PRESENT` (or `NOT_APPLICABLE`).
3. **`UNCERTAIN`**: A product is uncertain if key packaging information is `CANNOT_DETERMINE` or `PARTIAL` from listing metadata alone, and absence cannot be definitively established without packaging inspection.

---

## 2. Inconsistency Analysis Before vs. After Phase 4.5/5

| State | Total Products | Inconsistent Annotations | Consistency Rate |
|---|---|---|---|
| **Phase 4 (Post-Pilot)** | 75 | 36 (All 7 `PRESENT` but overall `NON_COMPLIANT`) | 52.0% |
| **Phase 5 (Post-Review & Finalization)** | 75 | **0** | **100.0%** |

---

## 3. Resolution Details of the 36 Inconsistent Cases

In Phase 4, 36 products had been labeled with all 7 declarations marked `PRESENT` because the annotator marked listing-level presence without inspecting whether packaging-specific declarations (like complete postal address D01, manufacturing date D04, or consumer helpline D06) were actually present.

During Phase 4.5/4.5B human review:
- **Manufacturer/Packer/Importer Name & Address (D01)**: Re-evaluated based on listing specifications and packaging images. Products with brand names only but missing full address/PIN code were corrected to `PARTIAL`.
- **Month/Year of Manufacture (D04)**: Properly corrected to `CANNOT_DETERMINE` (as manufacturing date is almost never provided in e-commerce listing metadata).
- **Consumer Care Details (D06)**: Corrected to `ABSENT` where no phone, email, or physical complaint address was provided.
- **Country of Origin (D07)**: Corrected to `CANNOT_DETERMINE` when origin was omitted from metadata and packaging images.
- **Overall Labels**:
  - **24 products** confirmed as `NON_COMPLIANT` due to explicit absence of mandatory declarations (e.g. D06 absent, D01 partial).
  - **12 products** categorized as `UNCERTAIN` where declarations were missing from listing text but required full physical packaging inspection.

---

## 4. Final Classification Matrix

| Consistency Status | Product Count | Percentage | Action Required |
|---|---|---|---|
| **`VALID`** | 75 | 100.0% | None — Passed |
| **`NEEDS_REVIEW`** | 0 | 0.0% | None |
| **`POSSIBLE_ANNOTATION_ERROR`** | 0 | 0.0% | None |

**Result**: Quality Gate G (Consistency) = **`PASS`**.
