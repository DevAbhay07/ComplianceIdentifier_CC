# Phase 13 — Confidence Calibration & Uncertainty Audit

## Executive Summary
Audits model probability calibration and reliability across partitions.

## Key Findings
1. **Overconfident Predictions**: The model outputs probabilities >0.85 for `NON_COMPLIANT` on almost all inputs.
2. **Abstention Policy**: When the abstention threshold is calibrated on validation data (0.65), it filters low-margin predictions on ambiguous inputs.
3. **Legal Disclaimer**: Confidence scores represent model statistical posterior estimates under its training prior, NOT statutory judicial certainty.