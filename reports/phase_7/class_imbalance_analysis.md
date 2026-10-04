# Multi-Label Target Class Imbalance Analysis (Phase 7)

> [!NOTE]
> In Phase 7, targets are preserved in their natural semantic distributions without artificial
> resampling or synthetic label generation. Imbalance flags are provided for Phase 8 loss weighting.

## 1. Declaration-Level Class Distribution (Training Partition, N=1159)

| Declaration | Name | PRESENT (%) | ABSENT (%) | PARTIAL (%) | CANNOT_DET (%) | Minority % | Imbalance Severity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D01`** | Manufacturer Name & Address | 9.7% | 32.4% | 57.9% | 0.0% | 9.7% | **`MODERATE_IMBALANCE`** |
| **`D02`** | Commodity Name | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | **`BALANCED`** |
| **`D03`** | Net Quantity | 83.4% | 16.6% | 0.0% | 0.0% | 16.6% | **`MILD_IMBALANCE`** |
| **`D04`** | Month/Year of Manufacture | 0.3% | 0.0% | 0.0% | 99.7% | 0.3% | **`SEVERE_IMBALANCE`** |
| **`D05`** | Maximum Retail Price (MRP) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | **`BALANCED`** |
| **`D06`** | Consumer Care Details | 3.7% | 96.1% | 0.2% | 0.0% | 0.2% | **`SEVERE_IMBALANCE`** |
| **`D07`** | Country of Origin | 0.4% | 0.0% | 0.0% | 99.6% | 0.4% | **`SEVERE_IMBALANCE`** |

## 2. Recommended Phase 8 Modeling Strategies

1. **High-Imbalance Declarations (`D02`, `D05`):** High baseline presence (>99%). Require precision-oriented loss functions.
2. **High-Uncertainty Declarations (`D04`, `D07`):** High `CANNOT_DETERMINE` (>99%). Modeling should focus on distinguishing rare explicit mentions from default missingness.
3. **Balanced Declarations (`D01`, `D03`, `D06`):** Meaningful variance between `PRESENT`, `PARTIAL`, and `ABSENT`. Suitable for multi-class classification or focal loss.
