# Frozen Phase 8 Model Audit Report (Phase 9A)

**Overall Audit Status:** **FAILED**  

| Declaration | Model Family | Feature Set | Exp Feats | Act Feats | Artifact Exists | Loads | Artifact SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | `RandomForest` | `FEATURE_SET_A` | 219 | 719 | Yes | Yes | `d7f7f51feedbe492...` | **❌ FAIL** |
| **`D02`** | `MajorityBaseline` | `NONE` | 0 | N/A (Constant) | Yes | Yes | `N/A (Deterministic Rule)` | **✅ PASS** |
| **`D03`** | `RandomForest` | `FEATURE_SET_A` | 219 | 719 | Yes | Yes | `7d5f71a46ce7f291...` | **❌ FAIL** |
| **`D04`** | `MajorityBaseline` | `NONE` | 0 | N/A (Constant) | Yes | Yes | `N/A (Deterministic Rule)` | **✅ PASS** |
| **`D05`** | `MajorityBaseline` | `NONE` | 0 | N/A (Constant) | Yes | Yes | `N/A (Deterministic Rule)` | **✅ PASS** |
| **`D06`** | `RandomForest` | `FEATURE_SET_B` | 719 | 719 | Yes | Yes | `8b4f7126d17e86ca...` | **✅ PASS** |
| **`D07`** | `MajorityBaseline` | `NONE` | 0 | N/A (Constant) | Yes | Yes | `N/A (Deterministic Rule)` | **✅ PASS** |

## Verified Target Classes per Learned Model

- **`D01` (RandomForest):** `[ABSENT, PARTIAL, PRESENT]`
- **`D03` (RandomForest):** `[ABSENT, PRESENT]`
- **`D06` (RandomForest):** `[ABSENT, PARTIAL, PRESENT]`

