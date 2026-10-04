# Locked Weak-Label Test Set Evaluation Results (Phase 8)

> [!NOTE]
> Evaluated on the locked Weak-Label Test Partition ($N=258$) after candidate model freezing.

| Declaration | Selected Model | Feature Set | Accuracy | Macro F1 | Weighted F1 | Balanced Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | RandomForest | `FEATURE_SET_A` | 0.907 | **0.8256** | 0.9057 | 0.8215 |
| **`D02`** | MajorityBaseline | `NONE` | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D03`** | RandomForest | `FEATURE_SET_A` | 0.7946 | **0.6636** | 0.8003 | 0.674 |
| **`D04`** | MajorityBaseline | `NONE` | 0.9961 | **0.499** | 0.9942 | 0.5 |
| **`D05`** | MajorityBaseline | `NONE` | 1.0 | **1.0** | 1.0 | 1.0 |
| **`D06`** | RandomForest | `FEATURE_SET_B` | 0.9884 | **0.897** | 0.9872 | 0.8333 |
| **`D07`** | MajorityBaseline | `NONE` | 0.9922 | **0.4981** | 0.9884 | 0.5 |

## Per-Class Breakdown (Test Split)

### `D01` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `ABSENT` | 1.0 | 1.0 | 1.0 | 103 |
| `PARTIAL` | 0.8984 | 0.9127 | 0.9055 | 126 |
| `PRESENT` | 0.5926 | 0.5517 | 0.5714 | 29 |

### `D02` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `PRESENT` | 1.0 | 1.0 | 1.0 | 258 |

### `D03` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `ABSENT` | 0.4231 | 0.4889 | 0.4536 | 45 |
| `PRESENT` | 0.8883 | 0.8592 | 0.8735 | 213 |

### `D04` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `CANNOT_DETERMINE` | 0.9961 | 1.0 | 0.9981 | 257 |
| `PRESENT` | 0.0 | 0.0 | 0.0 | 1 |

### `D05` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `PRESENT` | 1.0 | 1.0 | 1.0 | 258 |

### `D06` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `ABSENT` | 0.9881 | 1.0 | 0.994 | 249 |
| `PRESENT` | 1.0 | 0.6667 | 0.8 | 9 |

### `D07` Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `CANNOT_DETERMINE` | 0.9922 | 1.0 | 0.9961 | 256 |
| `PRESENT` | 0.0 | 0.0 | 0.0 | 2 |

