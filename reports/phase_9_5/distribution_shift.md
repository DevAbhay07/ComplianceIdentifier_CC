# Phase 9.5 Feature & Label Distribution Shift Analysis

> **Analysis Scope**: Quantifying distribution divergence between Flipkart Weak-Label Training Set ($N=1,159$), Human Reference Set ($N=75$), and Real Physical Packaging Validation Set ($N=20$).

## 1. The Multi-Modal Domain Shift

A critical finding in Phase 9.5 is the presence of **Domain and Source Asymmetry**:

1. **Training Domain (Flipkart Web Listings)**:
   - Input source: HTML titles, product bullet points, and specification tables.
   - Characteristics: Title and price are universally present (100%), whereas manufacturing dates and explicit physical manufacturer addresses are rarely transcribed into listing text.

2. **Real Validation Domain (Physical Product Packaging Images)**:
   - Input source: OCR text extracted from product box, bottle, or carton labels.
   - Characteristics: Under Legal Metrology Rules, packaging MUST physically bear date of manufacture (`D04`), origin (`D07`), full address (`D01`), and customer care (`D06`).

## 2. Label Prevalence Comparison Across Splits

| Declaration | Weak Train ($N=1159$) | Human Ref ($N=75$) | Real Packaging ($N=20$) | Shift Severity | Key Driver |
|:---|:---:|:---:|:---:|:---:|:---|
| `D01` (Manufacturer / Packer / Importer) | 41.2% PRESENT | 38.7% PRESENT | 75.0% PRESENT | **MODERATE** | Physical packaging almost always lists factory address. |
| `D02` (Generic or Common Name) | 100.0% PRESENT | 100.0% PRESENT | 100.0% PRESENT | **NONE** | Universal generic name in both listing title and packaging. |
| `D03` (Net Quantity / Units) | 68.4% PRESENT | 72.0% PRESENT | 90.0% PRESENT | **LOW** | Net quantity standardized across web and packaging. |
| `D04` (Date of Manufacture / Expiry) | 0.0% PRESENT | 0.0% PRESENT | 80.0% PRESENT | **HIGH** | Physical pack stamps date code; web metadata omits batch dates. |
| `D05` (Maximum Retail Price (MRP)) | 100.0% PRESENT | 100.0% PRESENT | 100.0% PRESENT | **NONE** | Price is universally declared across both modalities. |
| `D06` (Consumer Care Details) | 34.5% PRESENT | 32.0% PRESENT | 85.0% PRESENT | **MODERATE** | Customer care lines appear on all retail packaging. |
| `D07` (Country of Origin) | 0.0% PRESENT | 0.0% PRESENT | 85.0% PRESENT | **HIGH** | Country of Origin mandatory on packaging, omitted on legacy Flipkart listings. |

## 3. Impact on ML Model Transferability

1. **Zero-Degradation Declarations (`D02`, `D05`)**:
   - Perfect alignment across training and deployment domains. High confidence and 100% accuracy.

2. **Robust Feature Transfer (`D01`, `D03`, `D06`)**:
   - Random Forest model generalizes well because regex evidence features (e.g. pin codes, units, emails, phones) activate strongly on OCR text.

3. **Domain Shift Bottlenecks (`D04`, `D07`)**:
   - Baseline models trained on weak web data predict majority class (`ABSENT` / `NOT_APPLICABLE`), failing on physical packaging.
   - **Conclusion**: A model trained solely on e-commerce catalog metadata cannot learn physical date-code and origin stamp patterns without packaging training samples.
