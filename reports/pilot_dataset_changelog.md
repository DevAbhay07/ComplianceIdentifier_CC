# Pilot Dataset Changelog (Version 1.0.0)

## Release Information
- **Dataset Name**: NitiMark Pilot Ground Truth
- **Version**: `1.0.0`
- **Release Tag**: `pilot_ground_truth_75_v1`
- **Date**: September 5, 2026
- **Files**:
  - `data/processed/pilot_ground_truth_75_v1.jsonl` (SHA-256: `617b076cf9e07877207dcba13f28cf01b1cb6a72ec226485806ce605151523ec`)
  - `data/processed/pilot_ground_truth_75_v1.csv` (SHA-256: `0a316279fcf4cb09e13b86027a0092fa56ea6d817f54c25fca275c6023bb7e54`)
  - `data/processed/pilot_ground_truth_manifest.json`

---

## Changes from Raw Pilot Annotations to Version 1.0.0

1. **Reconciliation of 78 Annotation Records**:
   - Reconciled 78 raw annotation entries to exactly 75 unique products.
   - Identified and resolved 3 duplicate entries as consecutive human re-annotations (`150f0154de8f25165c9b25db63f9ed11`, `9ed8c3eeef76ac201ddebab9dbb5c04b`, `735820e1614052fde55659da9dc10ccc`).
2. **Reconciliation of 36 Flagged Inconsistencies**:
   - Integrated the 36 human review results from [`annotations/review_36_results.jsonl`](file:///home/devabhay07/Desktop/MLmodelV2/annotations/review_36_results.jsonl).
   - Inconsistency rate reduced from 48.0% (36/75) down to **0.0% (0/75)**.
3. **Canonical Schema Normalization**:
   - Enforced approved legal declaration schema for D01–D07 and overall compliance.
   - Standardized evidence source names to approved schema (`PRODUCT_TITLE`, `PRODUCT_SPECIFICATIONS`, `PRODUCT_DESCRIPTION`, `PRICE_FIELD`, `BRAND_FIELD`, `PRODUCT_IMAGE`, `PRODUCT_PAGE`, `NO_EVIDENCE`).
4. **Audit Metadata Isolation**:
   - Isolated machine drafts, reviewer IDs, timestamps, and confidence scores under the `audit` metadata structure.
5. **Quality Gate Compliance**:
   - All 14 Quality Gates (A through N) evaluated and verified as **`PASS`**.
