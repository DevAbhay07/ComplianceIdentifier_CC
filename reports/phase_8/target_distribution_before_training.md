# Target Distribution Before Training (Phase 8)

## 1. Overview & Dataset Profile
This audit inspects the exact class distributions of Legal Metrology declaration targets ($D01$ through $D07$) across the Train ($N=1,159$), Validation ($N=246$), Test ($N=258$), and Human Reference ($N=75$) splits.

---

## 2. Declaration-Level Class Distributions

### `D01`: Manufacturer / Packer / Importer Name & Address
| Class | Train Count (%) | Val Count (%) | Test Count (%) | Human Ref Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| `PARTIAL` | 671 (57.9%) | 130 (52.8%) | 126 (48.8%) | 43 (57.3%) |
| `ABSENT` | 376 (32.4%) | 92 (37.4%) | 103 (39.9%) | 5 (6.7%) |
| `PRESENT` | 112 (9.7%) | 24 (9.8%) | 29 (11.2%) | 27 (36.0%) |

---

### `D02`: Common / Generic Commodity Name
| Class | Train Count (%) | Val Count (%) | Test Count (%) | Human Ref Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| `PRESENT` | 1,159 (100.0%) | 246 (100.0%) | 258 (100.0%) | 75 (100.0%) |

---

### `D03`: Net Quantity
| Class | Train Count (%) | Val Count (%) | Test Count (%) | Human Ref Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| `PRESENT` | 967 (83.4%) | 210 (85.4%) | 213 (82.6%) | 59 (78.7%) |
| `ABSENT` | 192 (16.6%) | 36 (14.6%) | 45 (17.4%) | 13 (17.3%) |
| `PARTIAL` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 3 (4.0%) |

---

### `D04`: Month & Year of Manufacture / Packing / Import
| Class | Train Count (%) | Val Count (%) | Test Count (%) | Human Ref Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| `CANNOT_DETERMINE` | 1,155 (99.7%) | 246 (100.0%) | 257 (99.6%) | 43 (57.3%) |
| `PRESENT` | 4 (0.3%) | 0 (0.0%) | 1 (0.4%) | 32 (42.7%) |

---

### `D05`: Maximum Retail Price (MRP)
| Class | Train Count (%) | Val Count (%) | Test Count (%) | Human Ref Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| `PRESENT` | 1,159 (100.0%) | 246 (100.0%) | 258 (100.0%) | 75 (100.0%) |

---

### `D06`: Consumer Care / Complaint Contact Details
| Class | Train Count (%) | Val Count (%) | Test Count (%) | Human Ref Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| `ABSENT` | 1,114 (96.1%) | 236 (95.9%) | 249 (96.5%) | 62 (82.7%) |
| `PRESENT` | 43 (3.7%) | 10 (4.1%) | 9 (3.5%) | 8 (10.7%) |
| `PARTIAL` | 2 (0.2%) | 0 (0.0%) | 0 (0.0%) | 3 (4.0%) |
| `CANNOT_DETERMINE` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 2 (2.7%) |

---

### `D07`: Country of Origin
| Class | Train Count (%) | Val Count (%) | Test Count (%) | Human Ref Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| `CANNOT_DETERMINE` | 1,154 (99.6%) | 245 (99.6%) | 256 (99.2%) | 43 (57.3%) |
| `PRESENT` | 5 (0.4%) | 1 (0.4%) | 2 (0.8%) | 32 (42.7%) |

---

## 3. Modeling Implications & Class Handling Strategy

1. **Multiclass Declarations (`D01`, `D03`):** Active multi-class variance suitable for standard multi-class training and evaluation.
2. **Single-Class Majority Declarations (`D02`, `D05`):** 100% `PRESENT` in listing metadata. Majority baseline achieves 100% accuracy; models must be evaluated against majority baseline.
3. **Extreme Imbalance Declarations (`D04`, `D06`, `D07`):** Rare positive classes in weak labels (<4%). Models require balanced class weights or threshold adjustments. In evaluation, Macro F1 is the essential metric rather than raw accuracy.
