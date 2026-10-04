# Phase 10 Model Experiments & Empirical Validation Record

> **Project**: NitiMark-ML — Standalone ML Prediction Layer (SIH25057)  
> **Experiment Goal**: Compare candidate Direct ML architectures (Logistic Regression, Random Forest, XGBoost) and the Evidence-Aware Decision Engine across 4 dataset partitions.

---

## 1. Experimental Setup & Partition Protocol

- **Weak-Label Training Set**: $N = 1,159$ products (Group-partitioned by `product_family_id`, random seed: 42).
- **Weak-Label Validation Set**: $N = 246$ products (Independent family groups).
- **Weak-Label Test Set**: $N = 258$ products (Independent family groups).
- **Human Reference Set**: $N = 75$ products (Locked external benchmark).
- **Real Product Validation Set**: $N = 20$ products (Physical packaging photographs across 5 categories).

---

## 2. In-Distribution Validation Benchmarking

| Experiment ID | Model Architecture | Feature Set | Hyperparameters | Val Accuracy | Val Macro F1 | Val Balanced Acc |
|:---|:---|:---:|:---|:---:|:---:|:---:|
| `EXP_01` | Majority Baseline | None | Constant `NON_COMPLIANT` | 96.34% | 0.4907 | 50.00% |
| `EXP_02` | Logistic Regression | Set A ($D=219$) | $C=1.0$, `class_weight='balanced'` | 89.43% | 0.6104 | 68.35% |
| `EXP_03` | Logistic Regression | Set B ($D=719$) | $C=1.0$, `class_weight='balanced'` | 96.34% | 0.7948 | 84.18% |
| `EXP_04` | Random Forest | Set A ($D=219$) | $n=150$, `max_depth=10`, `class_weight='balanced'` | 97.97% | 0.8281 | 86.71% |
| `EXP_05` | **Random Forest (Selected Direct ML)** | **Set B ($D=719$)** | **$n=150$, `max_depth=10`, `class_weight='balanced'`** | **99.19%** | **0.9354** | **94.94%** |
| `EXP_06` | XGBoost | Set A ($D=219$) | $n=100$, `max_depth=6`, `lr=0.1` | 97.56% | 0.7438 | 74.38% |
| `EXP_07` | **Evidence-Aware Decision Model** | **Internal Evidence ($D=7$)** | $\tau_{\text{pos}}=0.40, \tau_{\text{neg}}=0.50$ | 96.34% | 0.4907 | 50.00% |

---

## 3. Model Selection Decision

Based **strictly on validation partition performance** (`val_macro_f1 = 0.9354`), **Random Forest on Feature Set B** is selected as the primary **Direct ML Model**.

The **Evidence-Aware Decision Engine** is deployed in parallel as the **Statutory Decision Model**, providing full explainability and transparent regulatory reasoning.
