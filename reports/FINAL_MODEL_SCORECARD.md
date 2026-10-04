# NitiMark-ML — Final Production Model Scorecard

**Evaluation Date**: September 7, 2026  
**Problem Statement**: SIH25057 — Automated Compliance Checker for Legal Metrology Declarations  
**Model Identifier**: `RandomForest_SET_A` (`models/product_compliance/best_product_compliance_model.joblib`)  
**Evaluation Script**: `src/evaluation/generate_ml_evaluation.py`  
**Metric Source**: `reports/ml_metrics.json` & `reports/ml_metrics.csv`  

---

## 1. Model Configuration & Artifact Details

| Property | Value / Specification |
|---|---|
| **Base Estimator** | `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)` |
| **Wrapper Class** | `src.models.product_compliance_model.ProductComplianceModel` |
| **Feature Set** | `SET_A` (16 core statutory declaration presence & confidence features) |
| **Training Scheme** | Multi-Domain Unified Training (`EXPERIMENT_F_UNIFIED_ALL` / `EXPERIMENT_J`) |
| **Training Sample Count** | $N = 1,506$ (Flipkart: 1,164, YOLO: 174, Synthetic: 168) |
| **Validation Sample Count** | $N = 325$ (Held-out stratified multi-domain validation split) |
| **Real Gold Benchmark** | $N = 20$ (`real_product_validation/`, 17 COMPLIANT, 3 NON_COMPLIANT) |
| **Abstention Policy** | Rejection if $P(\text{class}) < 0.65$, margin $< 0.20$, or OCR quality $< 0.60$ with completeness $< 0.35$ |

---

## 2. Benchmark Performance Scorecard

### 2.1 Full Real Gold Packaging Benchmark ($N=20$)

> [!IMPORTANT]
> The Real Gold benchmark represents multi-view physical consumer packaging curated with human-inspected manual references. It consists of 17 fully compliant packages and 3 intentional statutory violation packages (`real_prod_11`, `real_prod_12`, `real_prod_17`).

| Evaluation Metric | Trained NitiMark Model | Always-NON_COMPLIANT Baseline | Always-COMPLIANT Baseline | Majority Class Baseline | Outperforms Baseline? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Macro F1-Score** | **1.0000** | 0.1304 | 0.4595 | 0.4595 | **YES (+0.8696)** |
| **Balanced Accuracy** | **1.0000** | 0.5000 | 0.5000 | 0.5000 | **YES (+0.5000)** |
| **Matthews Corr Coef (MCC)** | **1.0000** | 0.0000 | 0.0000 | 0.0000 | **YES (+1.0000)** |
| **COMPLIANT Recall** | **1.0000 (17/17)** | 0.0000 (0/17) | 1.0000 (17/17) | 1.0000 (17/17) | **YES (+1.0000 vs Always-NC)** |
| **COMPLIANT Precision** | **1.0000** | 0.0000 | 0.8500 | 0.8500 | **YES (+0.1500)** |
| **NON_COMPLIANT Recall** | **1.0000 (3/3)** | 1.0000 (3/3) | 0.0000 (0/3) | 0.0000 (0/3) | **YES (+1.0000 vs Always-C)** |
| **NON_COMPLIANT Precision** | **1.0000** | 0.1500 | 0.0000 | 0.0000 | **YES (+0.8500)** |
| **NC False Negative Rate** | **0.0000 (0/3)** | 0.0000 | 1.0000 | 1.0000 | **Zero Safety Breaches** |
| **ROC-AUC** | **1.0000** | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE | **Optimal Discrimination** |
| **Average Precision (PR-AUC)** | **1.0000** | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE | **Optimal Precision** |
| **Abstention Rate** | **0.0000** | 0.0000 | 0.0000 | 0.0000 | **Full Confidence** |

---

### 2.2 Multi-Domain Held-Out Validation Split ($N=325$)

| Metric | Score | Note |
|---|:---:|---|
| **Validation Accuracy** | **100.0%** | All 325 multi-domain validation samples correctly classified |
| **Validation Balanced Accuracy** | **1.0000** | Equal sensitivity across both classes |
| **Validation Macro F1** | **1.0000** | Harmonic mean of per-class F1 scores |
| **Validation MCC** | **1.0000** | High correlation without class-imbalance bias |
| **Brier Calibration Score** | **0.0003** | Close to 0.0 (near-perfect probability calibration) |
| **System Abstention Rate** | **0.0031 (1/325)** | 1 borderline low-quality sample abstained to `UNCERTAIN` |
| **System Coverage** | **99.69%** | 324/325 samples passed confidence gating |

---

## 3. Critical Scientific Findings

### 3.1 Does the model outperform simple baselines on real data?
**YES, definitively.**
- On the Real Gold 20 packaging benchmark, the **Always-NON_COMPLIANT baseline fails completely on compliant packages**, achieving an accuracy of only 15.0% (3/20), a COMPLIANT recall of 0.0%, and a Macro F1 of 0.1304.
- The **Always-COMPLIANT baseline fails on safety violations**, suffering a 100% Non-Compliant False Negative Rate (missing all 3 violations).
- The **NitiMark trained model correctly identifies all 17 compliant products and all 3 non-compliant products**, achieving 100% Balanced Accuracy and 1.0000 MCC.

### 3.2 Is COMPLIANT recall measurable?
**YES.**
- On Flipkart catalog text alone, COMPLIANT recall is unmeasurable because 100% of catalog samples lack on-pack declarations.
- However, on the **Real Gold Packaging Benchmark ($N=20$)** and the **Synthetic Physical Packaging Test Split ($N=36$)**, both classes are well represented (17 C in Real, 18 C in Synthetic), and COMPLIANT recall is empirically verified at **100.0%**.

---

## 4. Primary Visual Evidence (Stored in `reports/figures/`)

1. **Confusion Matrix (Real Packaging)**: `reports/figures/confusion/confusion_matrix_real.png`
2. **ROC Curve Comparison**: `reports/figures/roc/roc_curve_comparison.png`
3. **Precision-Recall Curve**: `reports/figures/precision_recall/precision_recall_curves.png`
4. **Model Architecture Comparison**: `reports/figures/experiments/model_comparison_lr_rf_xgb.png`
5. **Baseline Comparison**: `reports/figures/experiments/baseline_comparison.png`
6. **Domain Shift Performance**: `reports/figures/experiments/domain_shift_performance.png`
7. **Feature Importance (Top 15)**: `reports/figures/feature_importance/feature_importance_top20.png`
8. **Feature Ablation (Sets A–E)**: `reports/figures/ablation/feature_ablation_sets_a_through_e.png`
9. **Confidence Calibration**: `reports/figures/calibration/reliability_diagram.png`
10. **Evidence Intervention Response**: `reports/figures/intervention/evidence_intervention_response.png`
