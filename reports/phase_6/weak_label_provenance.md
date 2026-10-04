# Phase 6: Weak-Label Dataset Provenance Report

## 1. Lineage & Source Dataset
- **Primary Source:** `data/raw/flipkart/flipkart_com-ecommerce_sample.csv` (20,002 raw listings).
- **Candidate Selection:** Diversity-aware stratified sampling across Legal Metrology relevance tiers (`HIGH_RELEVANCE`, `MEDIUM_RELEVANCE`, `LOW_RELEVANCE`, `UNKNOWN`), specification richness, description length, and price tiers.
- **Reference Dataset Isolation:** 75 human-verified ground-truth products (`data/processed/pilot_ground_truth_75_v1.jsonl`) were strictly excluded from the candidate pool to preserve an untouched, uncorrupted evaluation benchmark.
- **Processed Weak-Label Dataset:**
  - JSONL: `data/processed/flipkart_weak_labels_v1.jsonl` (1,663 records)
  - CSV: `data/processed/flipkart_weak_labels_v1.csv` (1,663 records)
  - Manifest: `reports/phase_6/weak_label_manifest.json`

---

## 2. Multi-Source Evidence Extraction Pipeline
Evidence is parsed, normalized, and logged from multiple observable fields per listing:

1. **`PRODUCT_TITLE`**: Normalized via Unicode NFKD and whitespace collapsing. Evaluated for generic commodity terms and embedded unit quantities.
2. **`PRODUCT_DESCRIPTION`**: Normalized text searched for manufacturer keywords, address tokens, consumer care contact details, helpline numbers (`1800...`, `+91...`), and country of origin phrases.
3. **`PRODUCT_SPECIFICATIONS`**: Ruby-hash specification strings parsed into key-value dictionaries. Cleaned for exact specification attributes (e.g., `Net Quantity`, `Type`, `Country of Origin`, `Customer Care Phone`).
4. **`PRICE_FIELD`**: Structured `retail_price` and `discounted_price` evaluated per Legal Metrology MRP declaration rules.
5. **`BRAND_FIELD`**: Structured brand metadata evaluated as partial manufacturer identity.
6. **`IMAGE_METADATA`**: Structured image URLs validated for syntactic integrity and domain structure.

---

## 3. Heuristic Rules & Declaration Mapping

| Declaration | Name | Primary Rules Applied | Confidence Policy | Default If Missing |
| :--- | :--- | :--- | :--- | :--- |
| **`D01`** | Manufacturer Name & Address | `D01_SPEC_MFG_FOUND`, `D01_BRAND_FIELD_FOUND`, `D01_SPEC_ADDR_FOUND` | `0.90` (mfg+address), `0.70` (brand only), `0.85` (absent) | `ABSENT` |
| **`D02`** | Commodity Name | `D02_GENERIC_IN_TITLE`, `D02_SPEC_TYPE_FOUND`, `D02_CATEGORY_LEAF` | `0.95` (title/spec generic), `0.80` (category leaf) | `ABSENT` |
| **`D03`** | Net Quantity | `D03_SPEC_QUANTITY_FOUND`, `D03_TITLE_QUANTITY_FOUND`, `D03_DESC_QUANTITY_FOUND` | `0.95` (SI unit in spec), `0.85` (title unit), `0.85` (absent) | `ABSENT` |
| **`D04`** | Month/Year of Manufacture | `D04_SPEC_DATE_FOUND`, `D04_DESC_DATE_FOUND`, `D04_NO_DATE_IN_LISTING` | `0.85` (spec date), `0.60` (desc date), `0.90` (missing) | `CANNOT_DETERMINE` |
| **`D05`** | Maximum Retail Price | `D05_PRICE_FIELD_PRESENT`, `D05_NO_PRICE_FOUND` | `0.95` (valid retail price), `0.90` (no price) | `ABSENT` |
| **`D06`** | Consumer Care Details | `D06_SPEC_CARE_FOUND`, `D06_DESC_CARE_FOUND`, `D06_NO_CARE_FOUND` | `0.90` (spec care), `0.75` (desc phone/email), `0.90` (absent) | `ABSENT` |
| **`D07`** | Country of Origin | `D07_SPEC_ORIGIN_FOUND`, `D07_DESC_ORIGIN_FOUND`, `D07_NO_ORIGIN_IN_LISTING` | `0.90` (spec origin), `0.75` (desc origin), `0.85` (missing) | `CANNOT_DETERMINE` |

---

## 4. Overall Compliance Derivation Policy
Overall compliance is derived deterministically from declaration states:
- **`NON_COMPLIANT`**: If any mandatory declaration is evidenced as `ABSENT`.
- **`COMPLIANT`**: If all mandatory declarations are evidenced as `PRESENT` (or `NOT_APPLICABLE`).
- **`UNCERTAIN`**: If no declaration is `ABSENT`, but unresolved `PARTIAL` or `CANNOT_DETERMINE` states exist.

---

## 5. Quality Tiering & Filtering
- **`TIER_A_HIGH_CONFIDENCE`**: Mean confidence $\ge 0.70$, $\le 2$ `CANNOT_DETERMINE` declarations, zero conflicting evidence.
- **`TIER_B_MEDIUM_CONFIDENCE`**: Mean confidence $\ge 0.45$, $\le 4$ `CANNOT_DETERMINE` declarations, zero conflicting evidence.
- **`TIER_C_LOW_CONFIDENCE`**: Mean confidence $\ge 0.20$.
- **`REJECTED`**: Missing essential fields, corrupted specifications, or unresolvable contradictions.

---

## 6. Versioning Metadata
- `dataset_version`: `1.0.0`
- `release_tag`: `flipkart_weak_labels_v1`
- `rule_engine_version`: `v1`
- `evidence_version`: `v1`
- `extraction_version`: `v1`
