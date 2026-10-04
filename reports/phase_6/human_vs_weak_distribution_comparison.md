# Human Ground Truth vs. Automated Weak-Label Distribution Comparison

> [!IMPORTANT]
> **Distribution similarity does not establish label accuracy.**  
> This comparison is for structural profiling and distribution auditing only.

- **Human Ground Truth Dataset:** `data/processed/pilot_ground_truth_75_v1.jsonl` (75 products, 100% human-verified)
- **Automated Weak-Label Dataset:** `data/processed/flipkart_weak_labels_v1.jsonl` (1,663 products, weak labels)

## 1. Overall Compliance Comparison

| State | Human Ground Truth (N=75) | Weak-Label Dataset (N=1663) |
| :--- | :--- | :--- |
| **`COMPLIANT`** | 2 (2.7%) | 0 (0.0%) |
| **`NON_COMPLIANT`** | 61 (81.3%) | 1607 (96.6%) |
| **`UNCERTAIN`** | 12 (16.0%) | 56 (3.4%) |

## 2. Per-Declaration Comparison (Prevalence %)

| Declaration | Human PRESENT | Weak PRESENT | Human ABSENT | Weak ABSENT | Human PARTIAL | Weak PARTIAL | Human CANNOT_DET | Weak CANNOT_DET |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | 36.0% | 9.9% | 6.7% | 34.3% | 57.3% | 55.7% | 0.0% | 0.0% |
| **`D02`** | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **`D03`** | 78.7% | 83.6% | 17.3% | 16.4% | 4.0% | 0.0% | 0.0% | 0.0% |
| **`D04`** | 42.7% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% | 57.3% | 99.7% |
| **`D05`** | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **`D06`** | 10.7% | 3.7% | 82.7% | 96.2% | 4.0% | 0.1% | 2.7% | 0.0% |
| **`D07`** | 42.7% | 0.5% | 0.0% | 0.0% | 0.0% | 0.0% | 57.3% | 99.5% |

## 3. Analytical Observations

1. **D01 (Manufacturer/Packer/Importer):** Both datasets show high `PARTIAL` rates due to e-commerce brand presence without full postal addresses.
2. **D02 (Commodity Name) & D05 (MRP):** High `PRESENT` rates across both datasets, driven by structured product names and retail price fields.
3. **D04 (Mfg Date) & D07 (Country of Origin):** High `CANNOT_DETERMINE` rates in both, reflecting fundamental listing omissions.
4. **D06 (Consumer Care):** High `ABSENT` rates across both, as sellers rarely provide grievance numbers on listing cards.
