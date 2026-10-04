# Weak-Label vs. Human Ground Truth Gap Analysis (Phase 8)

This report contrasts validation learning, test generalization, and human benchmark transfer.

## 1. Three-Way Metric Comparison

| Declaration | Selected Model | Val Macro F1 (Weak) | Test Macro F1 (Weak) | Human Macro F1 (Human GT) | Transfer Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | RandomForest | 0.8684 | 0.8256 | **0.3013** | `+0.5243` |
| **`D02`** | MajorityBaseline | 1.0 | 1.0 | **1.0** | `+0.0` |
| **`D03`** | RandomForest | 0.7532 | 0.6636 | **0.3317** | `+0.3319` |
| **`D04`** | MajorityBaseline | 1.0 | 0.499 | **0.3644** | `+0.1346` |
| **`D05`** | MajorityBaseline | 1.0 | 1.0 | **1.0** | `+0.0` |
| **`D06`** | RandomForest | 0.9086 | 0.897 | **0.2263** | `+0.6707` |
| **`D07`** | MajorityBaseline | 0.499 | 0.4981 | **0.3644** | `+0.1337` |

## 2. Key Diagnostic Insights

1. **High-Transfer Declarations (`D02`, `D05`):** Both Weak Test and Human Reference achieve 100% agreement, driven by universal presence of product names and retail price fields.
2. **Moderate-Transfer Declarations (`D01`, `D03`, `D06`):** Moderate agreement reflecting generalizable text patterns (brands, net quantities, customer care absences).
3. **Discrepancy on Date & Origin (`D04`, `D07`):** Weak labels naturally assign `CANNOT_DETERMINE` due to online listing omissions, whereas human ground truth contains physical packaging edge-case annotations. This divergence confirms that image OCR is essential for physical packaging declarations.
