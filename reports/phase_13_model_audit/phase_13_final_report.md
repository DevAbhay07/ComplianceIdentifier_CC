# Phase 13 — Comprehensive Scientific Model Audit Final Report

**Project**: NitiMark-ML (SIH25057)  
**Date**: September 2026  
**Status**: COMPLETE  

## Executive Summary
This audit explains why the declaration-centric model achieves 91.47% Macro F1 on Weak Test but ~0.2980 on physical packaging benchmarks.

## Key Metric Summary Across All Partitions

| Partition | Domain | N | Macro F1 | Non-Compliant Recall | Compliant F1 | Balanced Acc | MCC |
|---|---|---|---|---|---|---|---|
| **Weak Test** | Catalog Silver | 258 | **0.9147** | **100.0%** | 0.0000 | 0.8571 | 0.8418 |
| **Human Reference** | Catalog Human-Verified | 75 | **0.2990** | **100.0%** | 0.0000 | 0.3333 | 0.0000 |
| **Real Packaging** | Physical Packaging | **20** | **0.3333** | **100.0%** | 0.0000 | 0.5000 | 0.5000 |

## Summary of Audit Findings
1. **Target Provenance**: Weak training labels are heuristic silver rules, not gold ground truth.
2. **Prediction Skew**: The model predicts `NON_COMPLIANT` on 100% of out-of-domain physical samples, creating an apparent 100% recall while yielding 0.0% Compliant F1.
3. **Domain Shift**: Catalog metadata systematically omits physical packaging dates and helplines.
4. **Scientific Honesty**: The declaration extraction and fusion layers are fully functional; the ML classification layer requires physical-domain ground-truth labels for real-world certification.