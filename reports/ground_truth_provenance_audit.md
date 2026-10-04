# Ground-Truth Provenance Audit (Phase 5)

## Overview

This audit establishes the provenance and human verification status of every label in the finalized dataset:
`data/processed/pilot_ground_truth_75_v1.jsonl`.

---

## 1. Provenance Breakdown

| Provenance Source | Description | Record Count | Percentage |
|---|---|---|---|
| **`ORIGINAL_PILOT_ANNOTATION`** | Consistent original human annotations from Phase 3 | 39 | 52.0% |
| **`PHASE_4.5_HUMAN_REVIEW`** | Human-reviewed & resolved annotations from Phase 4.5/4.5B | 36 | 48.0% |
| **Total Finalized Ground Truth** | Unique pilot products with complete human verification | **75** | **100.0%** |

---

## 2. Ground-Truth Rule Compliance Check

| Check Item | Target | Result | Status |
|---|---|---|---|
| **Human Verification Rate** | 100.0% | 100.0% (`human_confirmed == True` for 75/75) | ✅ **PASS** |
| **Machine Draft Isolation** | Machine drafts $\neq$ Ground Truth | Isolated in `audit` metadata object | ✅ **PASS** |
| **Synthetic Label Exclusion** | 0 synthetic labels | 0 synthetic labels present | ✅ **PASS** |
| **Rule Engine Extraction Exclusion** | 0 rule-engine generated labels | All labels derived from human actions | ✅ **PASS** |
| **Annotator / Reviewer Attribution** | 100% of records attributed | 39 attributed to `annotator_01`, 36 to `reviewer_01` | ✅ **PASS** |

---

## 3. Reviewer Action Attribution across 36 Reviewed Products

For all 36 products that underwent Phase 4.5 review:
- **Decision Type**: 36 / 36 marked `CONFIRMED` by the human reviewer.
- **Review Reason Provided**: 36 / 36 records have detailed, evidence-based human justifications recorded in `audit.review_reason`.
- **Review Confidence**: High (36/36).
- **Declaration Review Actions**:
  - `ACCEPTED_AI_DRAFT`: 91 declarations where the reviewer reviewed and confirmed the proposed AI recommendation with evidence.
  - `MODIFIED_AI_DRAFT` / `MANUAL`: Handled according to explicit human choices.

---

## 4. Conclusion

The provenance of `pilot_ground_truth_75_v1` is **100% human-verified, auditable, and unpolluted by unverified machine outputs**.
