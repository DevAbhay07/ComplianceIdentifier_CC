# Phase 13 — Comprehensive Per-Class & Global Metrics

## Performance Table across All Partitions

| Metric | Validation (N=246) | Weak Test (N=258) | Human Reference (N=75) | Real Packaging (N=20) |
|---|---|---|---|---|
| **Overall Accuracy** | 0.9593 | 0.9922 | 0.8133 | 0.1500 |
| **Balanced Accuracy** | 0.6582 | 0.8571 | 0.3333 | 0.5000 |
| **Macro Precision** | 0.7017 | 0.9960 | 0.2711 | 0.3333 |
| **Macro Recall** | 0.6582 | 0.8571 | 0.3333 | 0.3333 |
| **Macro F1** | **0.6770** | **0.9147** | **0.2990** | **0.3333** |
| **Weighted F1** | 0.9569 | 0.9916 | 0.7296 | 0.1500 |
| **Matthews Corr (MCC)** | 0.3573 | 0.8418 | 0.0000 | 0.5000 |
| **Cohen's Kappa** | 0.3543 | 0.8295 | 0.0000 | 0.1304 |

## Per-Class Breakdown

| Partition | Class | Precision | Recall | F1 | Support | Predicted Count |
|---|---|---|---|---|---|---|
| **validation** | `COMPLIANT` | 0.0000 | 0.0000 | **0.0000** | 0 | 0 |
| **validation** | `NON_COMPLIANT` | 0.9749 | 0.9831 | **0.9790** | 237 | 239 |
| **test** | `COMPLIANT` | 0.0000 | 0.0000 | **0.0000** | 0 | 0 |
| **test** | `NON_COMPLIANT` | 0.9921 | 1.0000 | **0.9960** | 251 | 253 |
| **human_reference** | `COMPLIANT` | 0.0000 | 0.0000 | **0.0000** | 2 | 0 |
| **human_reference** | `NON_COMPLIANT` | 0.8133 | 1.0000 | **0.8971** | 61 | 75 |
| **real_packaging** | `COMPLIANT` | 0.0000 | 0.0000 | **0.0000** | 17 | 0 |
| **real_packaging** | `NON_COMPLIANT` | 1.0000 | 1.0000 | **1.0000** | 3 | 3 |