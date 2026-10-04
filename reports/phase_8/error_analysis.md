# Error Analysis & Failure Profiling (Phase 8)

This report catalogs classification error patterns across all splits.

## Overall Error Rates by Split and Declaration

| Split | Declaration | Model Family | Total Samples | Errors | Error Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `validation` | **`D01`** | RandomForest | 246 | 16 | `0.065` |
| `validation` | **`D02`** | MajorityBaseline | 246 | 0 | `0.0` |
| `validation` | **`D03`** | RandomForest | 246 | 26 | `0.1057` |
| `validation` | **`D04`** | MajorityBaseline | 246 | 0 | `0.0` |
| `validation` | **`D05`** | MajorityBaseline | 246 | 0 | `0.0` |
| `validation` | **`D06`** | RandomForest | 246 | 3 | `0.0122` |
| `validation` | **`D07`** | MajorityBaseline | 246 | 1 | `0.0041` |
| `test` | **`D01`** | RandomForest | 258 | 24 | `0.093` |
| `test` | **`D02`** | MajorityBaseline | 258 | 0 | `0.0` |
| `test` | **`D03`** | RandomForest | 258 | 53 | `0.2054` |
| `test` | **`D04`** | MajorityBaseline | 258 | 1 | `0.0039` |
| `test` | **`D05`** | MajorityBaseline | 258 | 0 | `0.0` |
| `test` | **`D06`** | RandomForest | 258 | 3 | `0.0116` |
| `test` | **`D07`** | MajorityBaseline | 258 | 2 | `0.0078` |
| `human_reference` | **`D01`** | RandomForest | 75 | 45 | `0.6` |
| `human_reference` | **`D02`** | MajorityBaseline | 75 | 0 | `0.0` |
| `human_reference` | **`D03`** | RandomForest | 75 | 17 | `0.2267` |
| `human_reference` | **`D04`** | MajorityBaseline | 75 | 32 | `0.4267` |
| `human_reference` | **`D05`** | MajorityBaseline | 75 | 0 | `0.0` |
| `human_reference` | **`D06`** | RandomForest | 75 | 13 | `0.1733` |
| `human_reference` | **`D07`** | MajorityBaseline | 75 | 32 | `0.4267` |

## Key Root Cause Diagnoses

1. **Declaration `D01` (Name & Address of Manufacturer):**
   - Primary errors occur between `PARTIAL` (brand/name only) and `ABSENT`. When descriptions mention brands vaguely without physical address markers, models occasionally overpredict `ABSENT`.

2. **Declaration `D03` (Net Quantity):**
   - Errors stem from edge-case units (e.g. piece counts 'pack of 2', combo items) where standard SI unit regex patterns yielded false negatives in weak labelling.

3. **Declaration `D04` & `D07` (Dates & Country of Origin):**
   - E-commerce weak labels reflect web metadata (mostly absent -> `CANNOT_DETERMINE`), while human annotations in the pilot benchmark captured physical packaging details. This accounts for the apparent transfer discrepancy.
