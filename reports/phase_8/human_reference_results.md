# Locked Human Ground Truth Reference Benchmark Results (Phase 8)

> [!IMPORTANT]
> Evaluated on the locked 75-product Human Ground Truth dataset (`pilot_ground_truth_75_v1.jsonl`).
> This measures real-world transfer from weak-label training to human-verified examples.

| Declaration | Selected Model | Feature Set | Human Accuracy | Human Macro F1 | Human Weighted F1 | Balanced Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | RandomForest | `FEATURE_SET_A` | 0.4 | **0.3013** | 0.3987 | 0.4728 |
| **`D02`** | MajorityBaseline | `NONE` | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | RandomForest | `FEATURE_SET_A` | 0.7733 | **0.3317** | 0.7062 | 0.3477 |
| **`D04`** | MajorityBaseline | `NONE` | 0.5733 | **0.3644** | 0.4179 | 0.5 |
| **`D05`** | MajorityBaseline | `NONE` | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | RandomForest | `FEATURE_SET_B` | 0.8267 | **0.2263** | 0.7482 | 0.25 |
| **`D07`** | MajorityBaseline | `NONE` | 0.5733 | **0.3644** | 0.4179 | 0.5 |

## Per-Class Breakdown (Human Reference)

### `D01` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `ABSENT` | 0.1212 | 0.8 | 0.2105 | 5 |
| `PARTIAL` | 0.6944 | 0.5814 | 0.6329 | 43 |
| `PRESENT` | 0.1667 | 0.037 | 0.0606 | 27 |

### `D02` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `PRESENT` | 1.0 | 1.0 | 1.0 | 75 |

### `D03` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `ABSENT` | 0.3333 | 0.0769 | 0.125 | 13 |
| `PARTIAL` | 0.0 | 0.0 | 0.0 | 3 |
| `PRESENT` | 0.7917 | 0.9661 | 0.8702 | 59 |

### `D04` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `CANNOT_DETERMINE` | 0.5733 | 1.0 | 0.7288 | 43 |
| `PRESENT` | 0.0 | 0.0 | 0.0 | 32 |

### `D05` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `PRESENT` | 1.0 | 1.0 | 1.0 | 75 |

### `D06` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `ABSENT` | 0.8267 | 1.0 | 0.9051 | 62 |
| `CANNOT_DETERMINE` | 0.0 | 0.0 | 0.0 | 2 |
| `PARTIAL` | 0.0 | 0.0 | 0.0 | 3 |
| `PRESENT` | 0.0 | 0.0 | 0.0 | 8 |

### `D07` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `CANNOT_DETERMINE` | 0.5733 | 1.0 | 0.7288 | 43 |
| `PRESENT` | 0.0 | 0.0 | 0.0 | 32 |

