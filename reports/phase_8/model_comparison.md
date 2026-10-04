# Model Family Comparison on Validation Partition (Phase 8)

> [!NOTE]
> Model selection is strictly driven by **Validation Macro F1** to prevent majority-class dominance.

| Declaration | Model Family | Feature Set | Val Accuracy | Val Macro F1 | Val Weighted F1 | Balanced Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | MajorityBaseline | NONE | 0.5285 | **0.2305** | 0.3654 | 0.3333 |
| **`D02`** | MajorityBaseline | NONE | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | MajorityBaseline | NONE | 0.8537 | **0.4605** | 0.7863 | 0.5 |
| **`D04`** | MajorityBaseline | NONE | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D05`** | MajorityBaseline | NONE | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | MajorityBaseline | NONE | 0.9593 | **0.4896** | 0.9394 | 0.5 |
| **`D07`** | MajorityBaseline | NONE | 0.9959 | **0.499** | 0.9939 | 0.5 |
| **`D01`** | LogisticRegression | FEATURE_SET_A | 0.9187 | **0.8228** | 0.9157 | 0.8118 |
| **`D02`** | LogisticRegression | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | LogisticRegression | FEATURE_SET_A | 0.7439 | **0.5955** | 0.7688 | 0.6313 |
| **`D04`** | LogisticRegression | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D05`** | LogisticRegression | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | LogisticRegression | FEATURE_SET_A | 0.9553 | **0.4886** | 0.9374 | 0.4979 |
| **`D07`** | LogisticRegression | FEATURE_SET_A | 0.9959 | **0.499** | 0.9939 | 0.5 |
| **`D01`** | LogisticRegression | FEATURE_SET_B | 0.935 | **0.8525** | 0.9311 | 0.8333 |
| **`D02`** | LogisticRegression | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | LogisticRegression | FEATURE_SET_B | 0.8455 | **0.6675** | 0.8396 | 0.6563 |
| **`D04`** | LogisticRegression | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D05`** | LogisticRegression | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | LogisticRegression | FEATURE_SET_B | 0.9634 | **0.7948** | 0.9657 | 0.8373 |
| **`D07`** | LogisticRegression | FEATURE_SET_B | 0.9959 | **0.499** | 0.9939 | 0.5 |
| **`D01`** | RandomForest | FEATURE_SET_A | 0.935 | **0.8684** | 0.935 | 0.8684 |
| **`D02`** | RandomForest | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | RandomForest | FEATURE_SET_A | 0.8943 | **0.7532** | 0.8852 | 0.7194 |
| **`D04`** | RandomForest | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D05`** | RandomForest | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | RandomForest | FEATURE_SET_A | 0.9756 | **0.7794** | 0.9705 | 0.7 |
| **`D07`** | RandomForest | FEATURE_SET_A | 0.9959 | **0.499** | 0.9939 | 0.5 |
| **`D01`** | RandomForest | FEATURE_SET_B | 0.9431 | **0.8654** | 0.9383 | 0.8385 |
| **`D02`** | RandomForest | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | RandomForest | FEATURE_SET_B | 0.878 | **0.744** | 0.875 | 0.7329 |
| **`D04`** | RandomForest | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D05`** | RandomForest | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | RandomForest | FEATURE_SET_B | 0.9878 | **0.9086** | 0.9868 | 0.85 |
| **`D07`** | RandomForest | FEATURE_SET_B | 0.9959 | **0.499** | 0.9939 | 0.5 |
| **`D01`** | XGBoost | FEATURE_SET_A | 0.9309 | **0.8468** | 0.9275 | 0.8318 |
| **`D02`** | XGBoost | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | XGBoost | FEATURE_SET_A | 0.878 | **0.6661** | 0.8543 | 0.6294 |
| **`D04`** | XGBoost | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D05`** | XGBoost | FEATURE_SET_A | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | XGBoost | FEATURE_SET_A | 0.9715 | **0.7235** | 0.9641 | 0.65 |
| **`D07`** | XGBoost | FEATURE_SET_A | 0.9959 | **0.499** | 0.9939 | 0.5 |
| **`D01`** | XGBoost | FEATURE_SET_B | 0.9431 | **0.8659** | 0.9382 | 0.8395 |
| **`D02`** | XGBoost | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | XGBoost | FEATURE_SET_B | 0.8618 | **0.5757** | 0.8221 | 0.5623 |
| **`D04`** | XGBoost | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D05`** | XGBoost | FEATURE_SET_B | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | XGBoost | FEATURE_SET_B | 0.9837 | **0.8708** | 0.9818 | 0.8 |
| **`D07`** | XGBoost | FEATURE_SET_B | 0.9959 | **0.499** | 0.9939 | 0.5 |

