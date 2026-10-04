# Product Compliance ML — Controlled Feature Set Ablation Study

## Objective
Measure the incremental contribution of declaration semantic values, OCR quality diagnostics, and YOLO physical packaging evidence across Sets A through E.

## Ablation Results (Combined Training Corpus)

| Feature Set | Features | Val Macro F1 | Val Balanced Acc | Val Compliant F1 | Test Macro F1 |
|---|---|---|---|---|---|
| **SET_A** | 16 features | 0.4991 | 0.5000 | 0.0000 | 0.4991 |
| **SET_B** | 40 features | 0.4991 | 0.5000 | 0.0000 | 0.4991 |
| **SET_C** | 46 features | 0.4991 | 0.5000 | 0.0000 | 0.4991 |
| **SET_D** | 56 features | 0.4991 | 0.5000 | 0.0000 | 0.4991 |
| **SET_E** | 61 features | 0.4991 | 0.5000 | 0.0000 | 0.4991 |