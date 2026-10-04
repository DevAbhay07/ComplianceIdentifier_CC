# Phase 13 — Weak Test Distributional Discrepancy Audit

## Why is Weak Test Macro F1 (0.9147) so high while Human Ref & Real Packaging are ~0.29?

### 1. Weak Silver Label Origin
The Weak Test set was generated using the exact same heuristic keyword matching rules as the training partition.
In the weak dataset, products with missing catalog text fields were labeled `NON_COMPLIANT` based on keyword absence (e.g. absence of 'mfd by' or 'mrp').
Because the ML model learned to correlate text length and keyword patterns with the synthetic target, it achieves 91.47% agreement on test samples drawn from the identical synthetic distribution.

### 2. Catalog Text Sparsity vs Packaging Visuals
- In catalog metadata, >90% of products omit manufacturing date (`D04`) and consumer care phone/email (`D06`).
- Consequently, ~96% of catalog samples in Weak Train/Val/Test are synthetically labeled `NON_COMPLIANT`.
- When evaluated on real physical packaging images where products actually possess printed dates and consumer care, the model continues to predict `NON_COMPLIANT` due to its strong catalog training prior.

### 3. Conclusion
The 91.47% test Macro F1 is a **synthetic in-distribution artifact**. It does NOT reflect real-world packaging compliance accuracy.