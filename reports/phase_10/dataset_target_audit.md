# Phase 10 Dataset & Compliance Target Provenance Audit

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Audit Scope**: Comprehensive provenance audit of product-level compliance targets across Flipkart Weak-Label splits, Human Reference Ground Truth, and Real-Product Validation datasets.

---

## 1. Executive Summary of Dataset Targets

Prior to training or evaluating any product-level compliance model, all candidate target labels were systematically audited for mathematical validity, provenance, derivation rules, and leakage risk.

```
+---------------------------------------------------------------------------------------------------+
|                                TARGET DISTRIBUTION AUDIT SUMMARY                                  |
+----------------------+---------+-------------------+----------------+---------------+-------------+
| Dataset Split        | Size (N)| NON_COMPLIANT (%) | UNCERTAIN (%)  | COMPLIANT (%) | Label Type  |
+----------------------+---------+-------------------+----------------+---------------+-------------+
| Weak-Label Train     | 1,159   | 1,119 (96.55%)    | 40 (3.45%)     | 0 (0.00%)     | WEAK_RULE   |
| Weak-Label Val       | 246     | 237 (96.34%)      | 9 (3.66%)      | 0 (0.00%)     | WEAK_RULE   |
| Weak-Label Test      | 258     | 251 (97.29%)      | 7 (2.71%)      | 0 (0.00%)     | WEAK_RULE   |
| Human Reference (P5) | 75      | 61 (81.33%)       | 12 (16.00%)    | 2 (2.67%)     | HUMAN_REF   |
| Real Packaging (P9.5)| 20      | 3 (15.00%)        | 0 (0.00%)      | 17 (85.00%)   | REAL_MANUAL |
+----------------------+---------+-------------------+----------------+---------------+-------------+
```

---

## 2. Nine-Point Target Provenance Analysis

### 1. Where does the overall compliance label come from?
- **In Weak-Label Dataset ($N=1,663$)**: Derived by the Phase 6 automated rule engine applied to regex evidence extracted from Flipkart catalog title, description, and specifications.
- **In Human Reference Dataset ($N=75$)**: Derived from the Phase 4.5/5 human verification workflow, where human annotators verified D01–D07 and confirmed overall compliance.
- **In Real Product Dataset ($N=20$)**: Derived by manual audit of physical packaging labels under the Legal Metrology (Packaged Commodities) Rules, 2011.

### 2. Is it human-derived?
- **Weak-Label Splits**: **NO**. Generated 100% automatically without human annotation.
- **Human Reference ($N=75$)**: **YES**. Dual-verified by human annotators.
- **Real Product Validation ($N=20$)**: **YES**. Manually inspected on physical retail packages.

### 3. Is it rule-derived?
- **Weak-Label Splits**: **YES**. Computed deterministically from D01–D07 weak labels.
- **Human Reference ($N=75$)**: Human-confirmed based on legal standards.
- **Real Product Validation ($N=20$)**: Audited against statutory rules.

### 4. Is it weak/silver?
- **Weak-Label Splits**: **YES**. Officially designated as `WEAK / SILVER PRODUCT COMPLIANCE LABEL`.
- **Human Reference**: **NO** (Gold Reference).
- **Real Product Validation**: **NO** (Empirical Physical Ground Truth).

### 5. Is it directly observed?
- **Weak-Label Splits**: **NO**. Observed only indirectly through web listing text.
- **Human Reference & Real Product**: **YES**. Directly observed on packaging/product cards.

### 6. Is it derived from D01–D07?
- **Weak-Label Splits**: **YES**. The rule engine marked a product `NON_COMPLIANT` if any mandatory declaration was `ABSENT` or `PARTIAL`, and `UNCERTAIN` if critical declarations were `CANNOT_DETERMINE`.
- **Significance**: If `D01–D07` ground-truth labels were used as features to predict `overall_target`, it would constitute **direct circular leakage**. Therefore, D01–D07 target labels are **strictly excluded** from feature matrices.

### 7. Does it contain label leakage?
- **Audited**: **NO LEAKAGE** into predictive feature sets ($X$).
- Verified by [`src/data/product_level_leakage_audit.py`](file:///home/devabhay07/Desktop/MLmodelV2/src/data/product_level_leakage_audit.py). Feature matrices $X$ contain only raw text TF-IDF, lexical length metrics, and parsed entity counts.

### 8. Is it suitable for training?
- **Direct ML Supervised Training on Weak Labels**: **LIMITED SCIENTIFIC UTILITY**.
  - Because `COMPLIANT` has **0 samples** in `train` ($0/1,159$), any direct classifier trained on `Y_train` can only learn a binary `NON_COMPLIANT` vs `UNCERTAIN` discriminator. It cannot learn the decision boundary for `COMPLIANT`.
- **Evidence-Aware Decision Model**: **HIGHLY SUITABLE**.
  - Evaluates individual declaration signals and confidence bounds dynamically.

### 9. Is it suitable only for evaluation?
- **Weak-Label Test ($N=258$)**: Evaluates weak in-distribution transfer.
- **Human Reference ($N=75$)**: Evaluates locked human benchmark transfer.
- **Real Product ($N=20$)**: Evaluates true physical deployment transfer.

---

## 3. Label Provenance Taxonomy

Every target record in the Phase 10 evaluation suite retains explicit provenance metadata:

| Label Source Code | Provenance Description | Permitted Usage |
|:---|:---|:---|
| `WEAK_RULE_DERIVED` | Automatically generated by rule engine on Flipkart catalog text. | In-distribution baseline training & validation. |
| `HUMAN_REFERENCE` | Verified by human annotators in Phase 5 ($N=75$). | Locked external benchmark evaluation ONLY. |
| `REAL_PRODUCT_MANUAL` | Audited on physical retail product packaging ($N=20$). | Out-of-distribution real-world validation ONLY. |

---

## 4. Conclusion & Modeling Strategy

Due to the complete absence of `COMPLIANT` samples in weak catalog listings ($0/1,159$), Phase 10 deploys and benchmarks:
1. **Direct ML Classifier (Approach A)**: Demonstrates the mathematical limitations of catalog-only weak supervision.
2. **Evidence-Aware Decision Policy (Approach B)**: Evaluates structured internal signals (D01–D07) under statutory compliance logic to reliably output `COMPLIANT`, `NON_COMPLIANT`, or `UNCERTAIN`.
3. **Direct ML + Uncertainty Abstention (Approach C)**: Abstains on out-of-distribution or low-confidence patterns.
