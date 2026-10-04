# Real-Product Validation Dataset & Experimentation Suite (Phase 9.5)

## 1. Directory Structure

- `images/`: High-resolution packaging label images for 20 real products.
- `inputs/`: Structured product input JSON files (`real_prod_01.json` to `real_prod_20.json`).
- `manual_reference/`: Ground-truth human reference declarations (`real_prod_01_ref.json` to `real_prod_20_ref.json`).
- `predictions/`: Machine-generated standalone inference results (`real_prod_01_pred.json` to `real_prod_20_pred.json`).
- `reports/`: Machine-readable summary JSON metrics.

## 2. Validation Metrics (N=20)

- **Declaration-Level Accuracy**: **50.0%**
- **Product-Level Perfect Agreement**: **0.0%**
- **Total Discrepancies**: **70** / 140 judgments

## 3. How to Run Validation

```bash
venv/bin/python real_product_validation/run_real_product_validation.py
```

## 4. Disclaimer

This validation is an empirical benchmark on 20 products to assess model transfer from web listings to physical packaging. It does not constitute statutory certification.
