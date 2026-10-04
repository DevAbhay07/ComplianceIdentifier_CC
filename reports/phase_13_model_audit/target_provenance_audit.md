# Phase 13 — Target Provenance & Label Integrity Audit

## Executive Summary
A rigorous scientific audit of target label origins reveals the exact cause of the divergence between Weak Test and physical benchmarks.

## Dataset Provenance Classification Matrix

| Dataset Split | N | Domain | Target Classification | Generation Method | Independent Human Verification | Suitability for Supervised Learning |
|---|---|---|---|---|---|---|
| **Train Split** | 1,159 | E-Commerce Catalog Metadata | `WEAKLY_DERIVED_SILVER` | Rule-based keyword matching on catalog text & specs | ❌ NO (Silver/Synthetic) | Suitable **ONLY** for weak prototype prototyping |
| **Validation Split** | 246 | E-Commerce Catalog Metadata | `WEAKLY_DERIVED_SILVER` | Same keyword heuristic rules as Train | ❌ NO (Silver/Synthetic) | Suitable **ONLY** for internal hyperparameter selection |
| **Weak Test Split** | 258 | E-Commerce Catalog Metadata | `WEAKLY_DERIVED_SILVER` | Same keyword heuristic rules as Train | ❌ NO (Silver/Synthetic) | **Evaluates agreement with weak keyword rules, NOT physical compliance** |
| **Human Reference** | 75 | E-Commerce Catalog Metadata | `HUMAN_VERIFIED_CATALOG` | Dual-adjudicated human expert review (Hash: `37dc62a308a6...`) | ✅ YES | Valid **ONLY** for catalog-domain evaluation |
| **Real Packaging Benchmark** | 20 | Physical Product Packaging Images | `HUMAN_VERIFIED_PHYSICAL_BENCHMARK` | Physical inspection of pre-packaged commodities | ✅ YES | Independent physical benchmark (**Evaluation Only, N=20**) |

## Critical Scientific Finding
> **The Weak Test target is NOT ground truth.**
> The 91.47% Macro F1 on Weak Test measures how closely the ML model approximates the **weak synthetic rule engine that created the dataset**, not whether the model accurately detects physical packaging compliance.
> When evaluated against human-verified packaging evidence, performance drops to ~0.2980 because the catalog rules do not represent real physical packaging distributions.