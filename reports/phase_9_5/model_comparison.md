# Phase 8 Model Comparison & Baseline Benchmark (Phase 9.5)

This report compiles performance comparisons across all evaluated model families and feature sets.

| Declaration | Evaluated Model Family | Feature Set | Val Macro F1 | Test Macro F1 | Human Ref Macro F1 | Selection Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | **RandomForest** | `FEATURE_SET_A` | **86.84%** | 82.56% | 30.13% | **FROZEN WINNER** |
| **`D02`** | **MajorityBaseline** | `NONE` | **100.00%** | 100.00% | 100.00% | **FROZEN WINNER** |
| **`D03`** | **RandomForest** | `FEATURE_SET_A` | **75.32%** | 66.36% | 33.17% | **FROZEN WINNER** |
| **`D04`** | **MajorityBaseline** | `NONE` | **100.00%** | 49.90% | 36.44% | **FROZEN WINNER** |
| **`D05`** | **MajorityBaseline** | `NONE` | **100.00%** | 100.00% | 100.00% | **FROZEN WINNER** |
| **`D06`** | **RandomForest** | `FEATURE_SET_B` | **90.86%** | 89.70% | 22.63% | **FROZEN WINNER** |
| **`D07`** | **MajorityBaseline** | `NONE` | **49.90%** | 49.81% | 36.44% | **FROZEN WINNER** |

