"""
generate_figures.py
===================
Generates publication-quality charts and visualizations for Phase 9.5 ML performance reporting.
Saves all figures in reports/phase_9_5/figures/.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DECLARATIONS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]
FIGURES_DIR = "reports/phase_9_5/figures"
METRICS_JSON_PATH = "reports/phase_9_5/all_metrics.json"
EXP_REGISTRY_PATH = "experiments/phase_8/experiment_registry.csv"
FEAT_IMP_DIR = "reports/phase_8/feature_importance"


def setup_plot_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 300
    })


def generate_all_figures():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    setup_plot_style()
    print("\n--- Generating Publication-Quality Figures for Phase 9.5 ---")

    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    # 1. Figure 01: Model Validation Macro F1
    plt.figure(figsize=(9, 5))
    val_f1 = [metrics_data["Validation"][d]["macro_f1"] * 100 for d in DECLARATIONS]
    colors = ["#2b5c8f" if metrics_data["Validation"][d]["is_baseline"] else "#2a9d8f" for d in DECLARATIONS]
    bars = plt.bar(DECLARATIONS, val_f1, color=colors, edgecolor="black", width=0.55)
    plt.title("Figure 01: Frozen Model Performance on Validation Partition (Macro F1 %)")
    plt.ylabel("Validation Macro F1 (%)")
    plt.xlabel("Declaration ID")
    plt.ylim(0, 110)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_model_validation_macro_f1.png"))
    plt.close()

    # 2. Figure 02: Model Test Macro F1
    plt.figure(figsize=(9, 5))
    test_f1 = [metrics_data["Test"][d]["macro_f1"] * 100 for d in DECLARATIONS]
    bars = plt.bar(DECLARATIONS, test_f1, color="#e76f51", edgecolor="black", width=0.55)
    plt.title("Figure 02: Generalization on Locked Weak-Label Test Partition (Macro F1 %)")
    plt.ylabel("Test Macro F1 (%)")
    plt.xlabel("Declaration ID")
    plt.ylim(0, 110)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_model_test_macro_f1.png"))
    plt.close()

    # 3. Figure 03: Model Human Reference Macro F1
    plt.figure(figsize=(9, 5))
    human_f1 = [metrics_data["Human_Reference"][d]["macro_f1"] * 100 for d in DECLARATIONS]
    bars = plt.bar(DECLARATIONS, human_f1, color="#9b5de5", edgecolor="black", width=0.55)
    plt.title("Figure 03: Transfer Performance on Locked Human Ground Truth Benchmark (Macro F1 %)")
    plt.ylabel("Human Reference Macro F1 (%)")
    plt.xlabel("Declaration ID")
    plt.ylim(0, 110)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_model_human_reference_macro_f1.png"))
    plt.close()

    # 4. Figure 04: Validation vs Test Generalization
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(DECLARATIONS))
    width = 0.35
    plt.bar(x - width/2, val_f1, width, label="Validation Split (Weak)", color="#2a9d8f", edgecolor="black")
    plt.bar(x + width/2, test_f1, width, label="Test Split (Weak)", color="#e76f51", edgecolor="black")
    plt.title("Figure 04: Generalization Comparison — Validation vs. Test Partitions (Macro F1 %)")
    plt.ylabel("Macro F1 (%)")
    plt.xlabel("Declaration ID")
    plt.xticks(x, DECLARATIONS)
    plt.ylim(0, 115)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_validation_vs_test.png"))
    plt.close()

    # 5. Figure 05: Test vs Human Reference Transfer Gap
    plt.figure(figsize=(10, 5.5))
    gaps = [round(test_f1[i] - human_f1[i], 2) for i in range(len(DECLARATIONS))]
    colors_gap = ["#e63946" if g > 15 else "#2a9d8f" for g in gaps]
    bars = plt.bar(DECLARATIONS, gaps, color=colors_gap, edgecolor="black", width=0.55)
    plt.axhline(0, color="black", linewidth=1.2, linestyle="--")
    plt.title("Figure 05: Transfer Gap (Test Weak-Label Macro F1 − Human Ground Truth Macro F1 %)")
    plt.ylabel("Transfer Gap Δ (%)")
    plt.xlabel("Declaration ID")
    plt.ylim(-10, 80)
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        plt.text(bar.get_x() + bar.get_width()/2., h + (1.5 if h >= 0 else -3), f"{h:+.1f}%", ha="center", va=va, fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_validation_vs_human_gap.png"))
    plt.close()

    # 6. Figure 06: Test Declaration Multi-Metric Performance
    plt.figure(figsize=(11, 5.5))
    test_acc = [metrics_data["Test"][d]["accuracy"] * 100 for d in DECLARATIONS]
    test_bal = [metrics_data["Test"][d]["balanced_accuracy"] * 100 for d in DECLARATIONS]
    x = np.arange(len(DECLARATIONS))
    width = 0.25
    plt.bar(x - width, test_acc, width, label="Raw Accuracy", color="#457b9d", edgecolor="black")
    plt.bar(x, test_bal, width, label="Balanced Accuracy", color="#f4a261", edgecolor="black")
    plt.bar(x + width, test_f1, width, label="Macro F1", color="#2a9d8f", edgecolor="black")
    plt.title("Figure 06: Comprehensive Multi-Metric Comparison on Test Partition")
    plt.ylabel("Metric Score (%)")
    plt.xlabel("Declaration ID")
    plt.xticks(x, DECLARATIONS)
    plt.ylim(0, 115)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_declaration_performance.png"))
    plt.close()

    # 7. Figure 07: Target Class Distribution Across Splits
    plt.figure(figsize=(11, 6))
    data_dir = "data/processed/ml_dataset_v1"
    train_y = pd.read_csv(os.path.join(data_dir, "train/Y_train.csv"))
    val_y = pd.read_csv(os.path.join(data_dir, "validation/Y_val.csv"))
    test_y = pd.read_csv(os.path.join(data_dir, "test/Y_test.csv"))
    human_y = pd.read_csv(os.path.join(data_dir, "human_reference/Y_human_ref.csv"))

    splits_dist = {
        "Train": train_y["D01_target"].value_counts(normalize=True) * 100,
        "Validation": val_y["D01_target"].value_counts(normalize=True) * 100,
        "Test": test_y["D01_target"].value_counts(normalize=True) * 100,
        "Human_Ref": human_y["D01_target"].value_counts(normalize=True) * 100
    }
    df_dist = pd.DataFrame(splits_dist).fillna(0).T
    df_dist.plot(kind="bar", stacked=True, figsize=(10, 5.5), colormap="Set2", edgecolor="black")
    plt.title("Figure 07: Class Distribution Shift Across Partitions (D01 Target Example)")
    plt.ylabel("Proportion (%)")
    plt.xlabel("Dataset Partition")
    plt.ylim(0, 100)
    plt.legend(title="Class Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "07_class_distribution.png"))
    plt.close()

    # 8-14. Figures 08-14: Confusion Matrices (D01-D07 on Test Split)
    for fig_num, did in enumerate(DECLARATIONS, start=8):
        plt.figure(figsize=(6, 5))
        cm_dict = metrics_data["Test"][did]["confusion_matrix"]
        cm_df = pd.DataFrame(cm_dict)
        labels = [col.replace("Pred_", "") for col in cm_df.columns]
        
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            linewidths=1.0,
            linecolor="gray"
        )
        plt.title(f"Figure {fig_num:02d}: Confusion Matrix for `{did}` (Test Split, N=258)")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{fig_num:02d}_confusion_matrix_{did}.png"))
        plt.close()

    # 15. Figure 15: Confidence Distribution
    plt.figure(figsize=(10, 5))
    confs = [metrics_data["Test"][d]["mean_confidence"] * 100 for d in DECLARATIONS]
    bars = plt.bar(DECLARATIONS, confs, color="#457b9d", edgecolor="black", width=0.55)
    plt.title("Figure 15: Mean Prediction Confidence per Declaration (Test Split)")
    plt.ylabel("Mean Confidence (%)")
    plt.xlabel("Declaration ID")
    plt.ylim(0, 110)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "15_confidence_distribution.png"))
    plt.close()

    # 16. Figure 16: Uncertainty Rate Distribution
    plt.figure(figsize=(10, 5))
    unc_rates = [metrics_data["Test"][d]["uncertainty_rate_pct"] for d in DECLARATIONS]
    bars = plt.bar(DECLARATIONS, unc_rates, color="#e63946", edgecolor="black", width=0.55)
    plt.title("Figure 16: Automated Uncertainty Flag Rate (%) per Declaration (Test Split)")
    plt.ylabel("Uncertainty Rate (%)")
    plt.xlabel("Declaration ID")
    plt.ylim(0, max(max(unc_rates) + 15, 30))
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "16_uncertainty_distribution.png"))
    plt.close()

    # 17-19. Figures 17-19: Top Feature Importances (D01, D03, D06)
    for did, fig_num in [("D01", 17), ("D03", 18), ("D06", 19)]:
        csv_p = os.path.join(FEAT_IMP_DIR, f"{did}_feature_importance.csv")
        if os.path.exists(csv_p):
            df_fi = pd.read_csv(csv_p).head(12)
            plt.figure(figsize=(9, 5.5))
            y_pos = np.arange(len(df_fi))
            plt.barh(y_pos, df_fi["importance"][::-1], color="#2a9d8f", edgecolor="black")
            plt.yticks(y_pos, df_fi["feature"][::-1])
            plt.title(f"Figure {fig_num:02d}: Top Predictive Features for `{did}` ({metrics_data['Validation'][did]['model_family']})")
            plt.xlabel("Feature Importance Score")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f"{fig_num:02d}_feature_importance_{did}.png"))
            plt.close()

    # 20. Figure 20: Model Family Comparison on Validation Split
    if os.path.exists(EXP_REGISTRY_PATH):
        df_exp = pd.read_csv(EXP_REGISTRY_PATH)
        # Aggregate best validation macro F1 per family
        fam_agg = df_exp.groupby(["declaration", "model_family"])["validation_macro_f1"].max().unstack() * 100
        plt.figure(figsize=(11, 6))
        fam_agg.plot(kind="bar", figsize=(11, 5.5), colormap="Spectral", edgecolor="black")
        plt.title("Figure 20: Comprehensive Model Family Validation Comparison (Macro F1 %)")
        plt.ylabel("Validation Macro F1 (%)")
        plt.xlabel("Declaration ID")
        plt.ylim(0, 115)
        plt.legend(title="Model Family", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "20_model_comparison.png"))
        plt.close()

    print(f"All 20 publication figures generated successfully in {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all_figures()
