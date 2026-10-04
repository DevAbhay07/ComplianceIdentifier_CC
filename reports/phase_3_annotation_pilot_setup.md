# Phase 3 — Annotation Pilot Setup Report

**Date**: 2026-09-01
**Status**: Setup COMPLETED. Awaiting human annotation.

---

## 1. Files Inspected

| File | Status | Findings |
|---|---|---|
| `configs/legal_requirements.yaml` | ✅ Inspected | 7 mandatory declarations (D01-D07), verification feasibility mapped |
| `docs/annotation_guidelines.md` | ✅ Inspected | Per-declaration labels: PRESENT/ABSENT/PARTIAL/NOT_APPLICABLE/CANNOT_DETERMINE. Overall: COMPLIANT/NON_COMPLIANT/UNCERTAIN |
| `data/raw/flipkart/flipkart_com-ecommerce_sample.csv` | ✅ Inspected | 20,002 rows, 15 columns, no compliance labels |

---

## 2. Scripts Created

| Script | Location | Purpose |
|---|---|---|
| `select_annotation_candidates.py` | `src/data/` | Filters Flipkart data to priority categories |
| `parse_specifications.py` | `src/data/` | Parses Ruby-hash product specifications to JSON |
| `create_pilot_sample.py` | `src/data/` | Creates stratified 75-product pilot sample |
| `annotation_tool.py` | `src/` | Streamlit annotation interface |
| `validate_annotations.py` | `src/data/` | Validates annotation JSONL for schema compliance |

---

## 3. Dataset Filtering Results

| Metric | Value |
|---|---|
| Total Flipkart products | 20,002 |
| Priority categories | Beauty and Personal Care, Baby Care, Kitchen & Dining |
| Candidates after filtering | **1,840** |

### Per-Category Breakdown

| Category | Count | Brand Missing (%) |
|---|---|---|
| Beauty and Personal Care | 710 | ~46.8% across all |
| Kitchen & Dining | 647 | |
| Baby Care | 483 | |

### Data Quality (Candidates)

| Field | Missing |
|---|---|
| Product name | 0 (0.0%) |
| Description | 0 (0.0%) |
| Product specifications | 1 (0.1%) |
| Brand | 862 (46.8%) |
| Retail price | 5 (0.3%) |
| Image URLs | 0 (0.0%) |

### Specification Parsing

| Metric | Value |
|---|---|
| Parse success rate | **99.9%** (1,838 / 1,840) |
| Parse failures | 2 (1 empty, 1 unexpected format) |
| Mean spec keys per product | 8.4 |
| Most common keys | Type, Ideal For, Sales Package, Material, Brand, Weight |

---

## 4. Pilot Sampling Methodology

| Parameter | Value |
|---|---|
| Target size | 75 products |
| Random seed | 42 |
| Strategy | Proportional category allocation + stratification by brand presence and spec richness |

### Category Allocation

| Category | Allocated | Available |
|---|---|---|
| Beauty and Personal Care | 29 | 710 |
| Kitchen & Dining | 26 | 647 |
| Baby Care | 20 | 483 |

### Diversity Achieved

| Feature | With | Without |
|---|---|---|
| Brand | 42 (56%) | 33 (44%) |
| Rich specs (≥5 keys) | 42 (56%) | 33 (44%) |
| Price | 75 (100%) | 0 (0%) |
| Unique product IDs | 75 | — |

---

## 5. Annotation Tool Capabilities

### What the Tool Displays

**Available Evidence section** (read-only):
- Product ID, name, category, full category tree
- Brand (with missing indicator)
- Retail price, discounted price
- Parsed product specifications (key-value pairs)
- Full description (expandable)

**Machine Guidance section** (clearly separated, read-only):
- Automated keyword-based hints for each declaration
- Explicitly marked as "NOT ground truth"
- Stored separately from human annotations

### What the Tool Collects

**Annotator Decision section** (human input):
- Per-declaration label (PRESENT/ABSENT/PARTIAL/NOT_APPLICABLE/CANNOT_DETERMINE)
- Evidence sources (LISTING_TITLE/DESCRIPTION/SPECIFICATION/PRICE_FIELD/BRAND_FIELD/IMAGE/OTHER/NONE)
- Evidence text / notes
- Overall compliance label (COMPLIANT/NON_COMPLIANT/UNCERTAIN)
- Overall notes with reasoning
- Confidence level (LOW/MEDIUM/HIGH)

### Label Leakage Prevention

- Machine guidance stored in `machine_guidance` field — **separate from** `declarations` and `overall_assessment`
- No auto-labeling occurs
- No rule engine is invoked to generate labels
- Human annotator makes all final decisions

---

## 6. Annotation Storage Format

**File**: `annotations/pilot_annotations.jsonl`
**Format**: JSON Lines (one record per line, append-only)

Each record contains:

```json
{
  "annotation_id": "product_id_timestamp",
  "schema_version": "1.0.0",
  "product_id": "...",
  "annotator_id": "...",
  "timestamp": "ISO-8601 UTC",
  "product_snapshot": {
    "name": "...",
    "category": "...",
    "brand": "...",
    "retail_price": ...,
    "spec_key_count": ...,
    "description_length": ...
  },
  "declarations": {
    "D01": {"label": "...", "evidence_sources": [...], "evidence_text": "..."},
    ...
  },
  "overall_assessment": {
    "label": "...",
    "notes": "...",
    "confidence": "..."
  },
  "machine_guidance": { ... }
}
```

Product snapshot is frozen at annotation time so future dataset changes don't invalidate context.

---

## 7. Validation Checks Implemented

| Check | Type |
|---|---|
| Missing required fields (annotation_id, product_id, etc.) | Error |
| Empty product_id | Error |
| Missing product_snapshot | Error |
| Missing declarations section | Error |
| Missing individual declarations (D01-D07) | Error |
| Unexpected declaration IDs | Error |
| Invalid declaration labels | Error |
| Invalid overall label | Error |
| Invalid evidence sources | Error |
| Invalid timestamp format | Error |
| Machine guidance containing compliance labels | Error |
| Duplicate product annotations | Warning |
| JSON parse errors in JSONL | Error |

---

## 8. Exact Commands to Start Annotation

```bash
# 1. Navigate to project
cd /home/devabhay07/Desktop/MLmodelV2

# 2. Activate environment
source venv/bin/activate

# 3. Start annotation tool
streamlit run src/annotation_tool.py

# 4. Open browser at http://localhost:8501

# 5. After annotation, validate:
python src/data/validate_annotations.py
```

---

## 9. Image Availability Experiment

| Metric | Value |
|---|---|
| URLs tested | 10 |
| Accessible (HTTP 200) | **10 (100%)** |
| Inaccessible | **0** |
| Content type | image/jpeg |
| CDN host | flixcart.com |

> **Significant finding**: Contrary to the Phase 1 expectation that Flipkart CDN URLs would be dead, all 10 tested URLs returned valid JPEG images. This means OCR/Vision features could potentially be extracted from product images in a future phase.

Results saved to: `reports/image_availability_test.json`

---

## 10. Current Blockers

| Blocker | Status |
|---|---|
| No ground-truth labels exist | 🔴 Awaiting human annotation |
| Human annotation not yet performed | 🔴 Tool ready, annotator must begin |
| Image-based annotation not yet tested | 🟡 Images accessible, but not yet integrated into tool |

---

## 11. Limitations

1. **Listing-only annotation**: Initial annotations are based on e-commerce listing metadata. D04 (manufacture date) and D06 (consumer care) will almost always be `CANNOT_DETERMINE` or `ABSENT`. This is a known and expected limitation.

2. **No inter-annotator agreement**: With a single annotator, we cannot measure inter-rater reliability. Annotations represent one person's judgment.

3. **Category filtering is approximate**: The classification of which Flipkart categories count as "packaged commodities" involves judgment. Some edge cases exist within each category.

4. **Brand ≠ Manufacturer**: The brand field is not the legal manufacturer name and address. Having a brand present does not mean D01 is fully satisfied.

5. **MRP ambiguity**: The `retail_price` field may not correspond to the legally required "MRP inclusive of all taxes" printed on the package.

---

## 12. What Must Happen Before Phase 4

### COMPLETED ✅
- [x] Clean candidate dataset (1,840 products)
- [x] Reproducible pilot sample (75 products)
- [x] Parsed product specifications (99.9% success)
- [x] Working annotation tool (Streamlit)
- [x] Annotation storage format (JSONL, append-only)
- [x] Annotation validation system
- [x] Annotation workflow documentation
- [x] Image availability test (10/10 accessible)
- [x] No fabricated ground truth
- [x] No circular rule-generated labels
- [x] No premature ML training

### NOT YET COMPLETED ❌
- [ ] **Human annotation of pilot products** (0 / 75 annotated)
- [ ] Pilot annotation analysis (depends on annotations)
- [ ] Assessment of listing-only annotation signal quality
- [ ] Decision on image integration for annotation

### Phase 4 Prerequisites
1. Complete pilot annotation (minimum ~30-50 products)
2. Run validation and review annotation statistics
3. Assess whether listing-only data provides enough signal
4. Decide whether to incorporate images (now known to be accessible)
5. Approval to proceed to feature engineering and dataset preparation
