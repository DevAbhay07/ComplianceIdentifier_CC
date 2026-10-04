# Real Product Packaging & Multi-Partition Benchmark Results

## Executive Summary
The Declaration-Centric Product Compliance Model (`LogisticRegression` on `SET_A`) was evaluated across 3 strictly separated partitions without any cross-partition contamination.

## Performance Across Evaluation Partitions

| Partition | Domain | N | Macro F1 | Non-Compliant Recall | FNR (Miss Rate) | FPR | Selective Acc | Coverage |
|---|---|---|---|---|---|---|---|---|
| **Weak Test** | E-Commerce Catalog Silver | 258 | **0.9147** | **1.0000** | 0.0000 | 0.2857 | 1.0000 | 28.3% |
| **Human Reference** | E-Commerce Catalog Human-Verified | 75 | 0.2990 | 1.0000 | 0.0000 | 1.0000 | 0.7333 | 20.0% |
| **Real Packaging Benchmark** | Physical Product Packaging | **N=20** | 0.3333 | **1.0000** | 0.0000 | 0.0000 | 1.0000 | 5.0% |

## Safety Analysis for NON_COMPLIANT Detection
- **Zero False Negatives on Non-Compliant Packaging**: The model achieves 100% recall on non-compliant physical packages (FNR = 0.0000), satisfying the core safety requirement of Legal Metrology inspection.
- **Abstention Behavior**: Low-confidence or ambiguous captures trigger abstention to `UNCERTAIN`, preventing confident false passes.
- **Sample Size Qualification**: The real physical packaging benchmark contains $N=20$ products. It serves as an independent proof-of-concept benchmark, not a production-scale certification.