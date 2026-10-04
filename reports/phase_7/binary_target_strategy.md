# Binary Target Strategy & Multi-Label Modeling Options (Phase 7)

## 1. Overview
The canonical NitiMark ML formulation is **Multi-Label Declaration Prediction** across $D01$ to $D07$, where each declaration retains its multi-class semantic states:
- `PRESENT`
- `ABSENT`
- `PARTIAL`
- `NOT_APPLICABLE`
- `CANNOT_DETERMINE`

However, some downstream applications or baseline models may benefit from simplified binary formulations. This report outlines the trade-offs and mapping strategies.

---

## 2. Comparison of Modeling Strategies

### Strategy 1: Canonical Multi-Class Multi-Label (Recommended for Phase 8)
- **Target Representation:** 7 independent categorical targets, each with 5 semantic states.
- **Advantages:**
  - Preserves legal nuance (e.g., `PARTIAL` brand presence vs full `PRESENT` manufacturer address).
  - Preserves epistemic uncertainty (`CANNOT_DETERMINE` due to listing omissions vs explicit `ABSENT`).
- **Disadvantages:**
  - Requires handling severe class imbalance on rare states (`PARTIAL` on D06, `PRESENT` on D04).

---

### Strategy 2: Binary Presence Modeling (`PRESENT` vs `NOT_PRESENT`)
- **Mapping:**
  - `PRESENT` $\rightarrow 1$
  - `ABSENT` $\rightarrow 0$
  - `PARTIAL` $\rightarrow 0$ (or excluded)
  - `CANNOT_DETERMINE` $\rightarrow$ Excluded from training loss (masked target)
  - `NOT_APPLICABLE` $\rightarrow$ Excluded
- **Advantages:**
  - Simpler binary classification problem per declaration.
  - Standard ROC-AUC and Binary Cross-Entropy metrics.
- **Disadvantages:**
  - Conflates explicit absence with missing listing evidence.
  - Loses partial declaration information.

---

### Strategy 3: Hierarchical Multi-Stage Architecture
1. **Stage 1 (Evidence Detectability):** Binary classifier predicting whether listing contains observable evidence for $D_i$ (`CANNOT_DETERMINE` vs `OBSERVABLE`).
2. **Stage 2 (Compliance State):** Multi-class classifier predicting `PRESENT`, `PARTIAL`, or `ABSENT` given that observable evidence exists.
3. **Stage 3 (Legal Rule Engine):** Deterministic legal aggregator producing `COMPLIANT`, `NON_COMPLIANT`, or `UNCERTAIN`.

---

## 3. Implementation Recommendation for Phase 8
Maintain the canonical 5-state representation in `Y_train.csv` ... `Y_test.csv` to ensure maximum experimental flexibility without forcing irreversible target collapse.
