# Phase 8 Model Freeze Record

**Status:** **FROZEN**  
**Freeze Timestamp:** 2026-09-05T00:34:05.498909+00:00  
**Selection Criterion:** Maximum Validation Macro F1  

## Selected & Frozen Models per Declaration

| Declaration | Model Family | Feature Set | Hyperparameters | Val Macro F1 | Val Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | **RandomForest** | `FEATURE_SET_A` | `{"n_estimators": 50, "max_depth": null, "class_weight": "balanced"}` | **0.8684** | 0.935 |
| **`D02`** | **MajorityBaseline** | `NONE` | `constant=PRESENT` | **1.0** | 1.0 |
| **`D03`** | **RandomForest** | `FEATURE_SET_A` | `{"n_estimators": 50, "max_depth": 20, "class_weight": "balanced"}` | **0.7532** | 0.8943 |
| **`D04`** | **MajorityBaseline** | `NONE` | `constant=CANNOT_DETERMINE` | **1.0** | 1.0 |
| **`D05`** | **MajorityBaseline** | `NONE` | `constant=PRESENT` | **1.0** | 1.0 |
| **`D06`** | **RandomForest** | `FEATURE_SET_B` | `{"n_estimators": 50, "max_depth": 10, "class_weight": "balanced"}` | **0.9086** | 0.9878 |
| **`D07`** | **MajorityBaseline** | `NONE` | `constant=CANNOT_DETERMINE` | **0.499** | 0.9959 |

## Protocol Verification

- Test split ($N=258$) remained completely untouched during model selection.
- Human reference split ($N=75$) remained completely untouched during model selection.
- All model configurations are locked prior to out-of-sample evaluation.
