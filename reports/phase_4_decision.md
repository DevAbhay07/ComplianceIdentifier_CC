# Phase 4 Decision — Are We Ready for ML Training?

**Date**: 2026-09-05
**Decision**: **NO — NOT YET READY**

---

## Answer: NO

We are **not ready** to begin ML training. Three critical issues must be resolved first.

---

## Critical Blocker 1: 36 Inconsistent Annotations

**Problem**: 48% of annotations (36/75) have ALL declarations marked PRESENT but the overall label is NON_COMPLIANT. This is an internal contradiction caused by the human accepting the machine-draft overall label without updating it after modifying individual declarations.

**Impact**: If used for training, these labels would teach the model that a product with all declarations present is still non-compliant — the exact opposite of the annotation guidelines.

**Resolution Required**:
1. Re-review these 36 products (listed in `annotations/review_required.csv`)
2. For each product, verify: if all 7 declarations are genuinely PRESENT, the overall label should be COMPLIANT
3. This re-review will likely increase the COMPLIANT count from 2 to approximately 38 products

**Estimated Effort**: 1-2 hours (quick re-review per product since declarations are already labeled)

---

## Critical Blocker 2: Extreme Class Imbalance (Current State)

**Problem**: Current distribution is 2.7% COMPLIANT / 97.3% NON_COMPLIANT / 0% UNCERTAIN. A classifier would achieve 97% accuracy by always predicting NON_COMPLIANT.

**After Blocker 1 is resolved**: Distribution may shift to approximately 50% COMPLIANT / 50% NON_COMPLIANT — a much healthier balance. This is the optimistic scenario.

**Impact**: If class balance improves after re-review, this blocker may be resolved automatically.

---

## Critical Blocker 3: Evidence Source Data Quality

**Problem**: 420 instances of PRESENT labels with no evidence sources recorded. This prevents meaningful source importance analysis and limits feature engineering.

**Impact**: We cannot reliably determine which listing features (title, specs, description) contain the evidence for each declaration. Future annotation must enforce source selection.

**Resolution**: This is not a full blocker for initial training but reduces the quality of feature engineering insights.

---

## Recommended ML Problem: OPTION D — Hybrid Multi-Label Architecture

After analyzing all four options:

### Option A: Binary Classification (COMPLIANT / NON_COMPLIANT)

- **Rejected** as primary approach
- With only 75 pilot samples and severe class imbalance, binary classification is premature
- The UNCERTAIN class (0% in pilot) may emerge in expanded datasets
- Simplistic — does not capture which specific declarations are problematic

### Option B: Three-Class Classification

- **Not supported by data** — zero UNCERTAIN annotations in the pilot
- May work with a larger dataset but the pilot provides no evidence for this class

### Option C: Multi-Label Declaration Prediction

- **Strong candidate**
- Predict each declaration (D01-D07) independently as PRESENT/ABSENT/PARTIAL
- D02 and D05 are constants (always PRESENT) — 2 of 7 declarations have no predictive value
- D06 (Consumer Care) has the strongest signal (59% PRESENT, 39% ABSENT)
- Allows the rule engine to handle the overall compliance logic
- **Most interpretable** — model explains WHY a product may be non-compliant

### Option D: Hybrid Multi-Label + Overall (RECOMMENDED)

- **Primary**: Multi-label declaration prediction (Option C)
- **Secondary**: The rule engine derives overall compliance from ML declaration predictions
- **Uncertainty**: Model provides per-declaration confidence scores
- **Interpretability**: Each prediction is traceable to specific declarations

**Why this is the right choice**:
1. Aligns with the hybrid architecture (ML + Rules + RAG)
2. ML predicts declaration presence probabilities
3. Rule engine applies legal logic to declaration predictions
4. RAG retrieves supporting legal evidence
5. Each component does what it does best

---

## Recommended Dataset Expansion Strategy

### Stage 1: Pilot Repair + Annotation Quality (IMMEDIATE)

- Re-review 36 inconsistent annotations (~1-2 hours)
- Verify D04 and D07 evidence for a sample of annotations
- Enforce evidence source selection in future annotations

### Stage 2: Listing-Only Expansion (300-500 products)

| Parameter | Value |
|---|---|
| Target size | 400 products |
| Categories | Beauty (120), Baby Care (100), Kitchen (100), + 2 new categories (80) |
| New categories | Home Improvement, Tools & Hardware |
| Annotation method | Draft-assisted + human review (enforced source selection) |
| Estimated annotation time | 2-3 minutes/product with draft acceleration |
| Total estimated effort | ~15-20 hours |

**Justification**: 400 products with ~5 variable declarations provides approximately 2,000 declaration-level training examples.

### Stage 3: Listing + Image Expansion (after Stage 2 model baseline)

| Parameter | Value |
|---|---|
| Target size | 800-1,000 products |
| Images | Downloaded for all products (96% available) |
| Annotation method | Listing + image side-by-side review |
| Focus declarations | D01, D04, D06, D07 (benefit most from images) |
| Estimated image download | ~100-200 MB |

**Justification**: Only proceed if Stage 2 model identifies specific declarations where image evidence would improve prediction.

### Stage 4: Scale (only if justified)

- Expand beyond 1,000 only if model performance and annotation quality warrant it
- Consider active learning to select the most informative products for annotation

### UNCERTAIN Samples

- **Retain** declaration-level UNCERTAIN/CANNOT_DETERMINE labels
- Do **not** force into binary PRESENT/ABSENT
- Use as a third class in declaration-level predictions if sufficient examples emerge

### Declaration-Level Labels

- **Preserve** all 7 per-declaration labels in the annotation schema
- These are more valuable than the overall label for the multi-label approach
- The overall label becomes a derived quantity from the rule engine

---

## What Must Happen Next

### Before ML Training Can Begin

1. ✅ Complete pilot annotations (75/75 done)
2. ❌ **Re-review 36 inconsistent annotations** → estimate new class distribution
3. ❌ **Verify D04/D07 evidence quality** for a sample
4. ❌ **Expand dataset to ~400 products** (Stage 2)
5. ❌ **Add evidence source enforcement** to annotation tool
6. ❌ **Add 2 new product categories** to annotation pool
7. ❌ **Begin feature engineering** based on verified declaration labels

### Phase 5 Proposal: Dataset Expansion + Feature Engineering

Only after Blockers 1-2 are resolved:

1. Expand the annotation candidate pool to 5 categories
2. Create a 400-product annotation sample with stratified sampling
3. Annotate with draft assistance + enforced evidence sources
4. Design the multi-label feature matrix (listing features → declaration predictions)
5. Establish a baseline model for declaration-level prediction

---

## Summary

| Question | Answer |
|---|---|
| Are all 75 products annotated? | ✅ YES |
| Are annotations valid? | ⚠️ PARTIALLY — 36 need re-review |
| Is class balance acceptable? | ❌ NO (97/3 split) — may improve after re-review |
| Is the pilot sufficient for ML training? | ❌ NO — 75 products is too small |
| Are images available? | ✅ YES (96% accessible) |
| What ML problem should we solve? | Multi-label declaration prediction (Option D) |
| What's the next step? | Re-review 36 annotations, then expand to 400 products |
| Can we start training now? | **NO** |
