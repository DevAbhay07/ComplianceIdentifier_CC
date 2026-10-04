# Complete ML Performance Metrics Report (Phase 9.5)

> [!NOTE]
> Comprehensive evaluation of frozen Phase 8 models across all four dataset partitions.

## Train Partition Metrics ($N=1159$)

| Declaration | Model Architecture | Feature Set | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | ROC-AUC (OvR) | Mean Conf | Uncertainty Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | RandomForest | `FEATURE_SET_A` | 99.74% | 99.85% | **99.59%** | 99.74% | 0.9953 | 0.9953 | 1.0 | 90.36% | 1.12% |
| **`D02`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D03`** | RandomForest | `FEATURE_SET_A` | 98.53% | 99.12% | **97.44%** | 98.56% | 0.95 | 0.9488 | 1.0 | 82.15% | 5.52% |
| **`D04`** | BASELINE | `NONE` | 99.65% | 50.00% | **49.91%** | 99.48% | NOT APPLICABLE (Single Class) | 0.0 | NOT APPLICABLE | 100.00% | 0.00% |
| **`D05`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D06`** | RandomForest | `FEATURE_SET_B` | 100.00% | 100.00% | **100.00%** | 100.00% | 1.0 | 1.0 | 1.0 | 92.18% | 0.17% |
| **`D07`** | BASELINE | `NONE` | 99.57% | 50.00% | **49.89%** | 99.35% | NOT APPLICABLE (Single Class) | 0.0 | NOT APPLICABLE | 100.00% | 0.00% |

## Validation Partition Metrics ($N=246$)

| Declaration | Model Architecture | Feature Set | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | ROC-AUC (OvR) | Mean Conf | Uncertainty Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | RandomForest | `FEATURE_SET_A` | 93.50% | 86.84% | **86.84%** | 93.50% | 0.8862 | 0.8862 | 0.9612 | 79.96% | 14.63% |
| **`D02`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D03`** | RandomForest | `FEATURE_SET_A` | 89.43% | 71.94% | **75.32%** | 88.52% | 0.5228 | 0.5092 | 0.8296 | 77.62% | 15.45% |
| **`D04`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D05`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D06`** | RandomForest | `FEATURE_SET_B` | 98.78% | 85.00% | **90.86%** | 98.68% | 0.8314 | 0.8174 | 0.8968 | 90.35% | 1.22% |
| **`D07`** | BASELINE | `NONE` | 99.59% | 50.00% | **49.90%** | 99.39% | NOT APPLICABLE (Single Class) | 0.0 | NOT APPLICABLE | 100.00% | 0.00% |

## Test Partition Metrics ($N=258$)

| Declaration | Model Architecture | Feature Set | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | ROC-AUC (OvR) | Mean Conf | Uncertainty Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | RandomForest | `FEATURE_SET_A` | 90.70% | 82.15% | **82.56%** | 90.57% | 0.8415 | 0.8414 | 0.9511 | 78.33% | 16.67% |
| **`D02`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D03`** | RandomForest | `FEATURE_SET_A` | 79.46% | 67.40% | **66.36%** | 80.03% | 0.3292 | 0.3279 | 0.8165 | 76.45% | 18.22% |
| **`D04`** | BASELINE | `NONE` | 99.61% | 50.00% | **49.90%** | 99.42% | NOT APPLICABLE (Single Class) | 0.0 | NOT APPLICABLE | 100.00% | 0.00% |
| **`D05`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D06`** | RandomForest | `FEATURE_SET_B` | 98.84% | 83.33% | **89.70%** | 98.72% | 0.8116 | 0.7943 | 0.938 | 91.07% | 1.55% |
| **`D07`** | BASELINE | `NONE` | 99.22% | 50.00% | **49.81%** | 98.84% | NOT APPLICABLE (Single Class) | 0.0 | NOT APPLICABLE | 100.00% | 0.00% |

## Human Reference Partition Metrics ($N=75$)

| Declaration | Model Architecture | Feature Set | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | MCC | Cohen's Kappa | ROC-AUC (OvR) | Mean Conf | Uncertainty Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | RandomForest | `FEATURE_SET_A` | 40.00% | 47.28% | **30.13%** | 39.87% | 0.1205 | 0.1 | 0.5907 | 77.41% | 13.33% |
| **`D02`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D03`** | RandomForest | `FEATURE_SET_A` | 77.33% | 34.77% | **33.17%** | 70.62% | 0.0684 | 0.0471 | 0.7309 | 82.30% | 8.00% |
| **`D04`** | BASELINE | `NONE` | 57.33% | 50.00% | **36.44%** | 41.79% | NOT APPLICABLE (Single Class) | 0.0 | NOT APPLICABLE | 100.00% | 0.00% |
| **`D05`** | BASELINE | `NONE` | 100.00% | 100.00% | **100.00%** | 100.00% | NOT APPLICABLE (Single Class) | NOT APPLICABLE (Single Class) | NOT APPLICABLE | 100.00% | 0.00% |
| **`D06`** | RandomForest | `FEATURE_SET_B` | 82.67% | 25.00% | **22.63%** | 74.82% | NOT APPLICABLE (Single Class) | 0.0 | 0.5154 | 92.92% | 0.00% |
| **`D07`** | BASELINE | `NONE` | 57.33% | 50.00% | **36.44%** | 41.79% | NOT APPLICABLE (Single Class) | 0.0 | NOT APPLICABLE | 100.00% | 0.00% |

