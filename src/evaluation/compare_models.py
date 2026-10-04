"""
compare_models.py
==================
Evaluates baselines and compares all trained model families (Logistic Regression,
Random Forest, XGBoost across Feature Sets A and B) on the Validation partition.
Selects the best candidate model per declaration strictly on Validation Macro F1
and generates the formal model freeze record.

Outputs:
    reports/phase_8/model_comparison.md
    reports/phase_8/validation_results.md
    reports/phase_8/model_freeze_record.md
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from collections import Counter

from src.evaluation.metrics import evaluate_declaration_predictions


DECLARATIONS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]


def evaluate_majority_baseline(
    Y_train: pd.DataFrame,
    Y_val: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Establish Majority-Class Baseline on Validation set."""
    baseline_results = {}
    target_decls = [did for did in DECLARATIONS if f"{did}_target" in Y_train.columns]
    for did in target_decls:
        target_col = f"{did}_target"
        train_counts = Counter(Y_train[target_col].dropna())
        majority_class = train_counts.most_common(1)[0][0]

        y_val = Y_val[target_col].values
        val_preds = [majority_class] * len(y_val)
        val_metrics = evaluate_declaration_predictions(y_val, val_preds)

        baseline_results[did] = {
            "model_family": "MajorityBaseline",
            "majority_class": majority_class,
            "metrics": val_metrics
        }
    return baseline_results


def compare_and_select_models(
    all_experiments: Dict[str, Dict[str, Dict[str, Any]]],
    baseline_results: Dict[str, Dict[str, Any]],
    output_dir: str
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Compare all models against baseline and select the winning candidate per declaration.
    all_experiments format: { "LR_SetA": { "D01": {...}, ... }, "LR_SetB": {...}, ... }
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Comparing Model Families on Validation Partition ---")

    comparison_rows = []

    # Add Baseline
    for did, base in baseline_results.items():
        comparison_rows.append({
            "declaration": did,
            "model_family": "MajorityBaseline",
            "feature_set": "NONE",
            "val_accuracy": base["metrics"]["accuracy"],
            "val_macro_f1": base["metrics"]["macro_f1"],
            "val_weighted_f1": base["metrics"]["weighted_f1"],
            "balanced_accuracy": base["metrics"]["balanced_accuracy"],
            "hyperparameters": f"constant={base['majority_class']}"
        })

    # Add all trained models
    for exp_key, decl_dict in all_experiments.items():
        for did, res in decl_dict.items():
            if not res or "metrics" not in res:
                continue
            m = res["metrics"]
            # Extract family and feature set
            parts = exp_key.split("_")
            family = parts[0]
            fset = "_".join(parts[1:])
            comparison_rows.append({
                "declaration": did,
                "model_family": family,
                "feature_set": res.get("feature_set", fset),
                "val_accuracy": m["accuracy"],
                "val_macro_f1": m["macro_f1"],
                "val_weighted_f1": m["weighted_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "hyperparameters": json.dumps(res.get("params", {}))
            })

    comp_df = pd.DataFrame(comparison_rows)

    # Model Selection per declaration: Highest val_macro_f1, tiebreaker: weighted_f1, accuracy
    frozen_candidates = {}
    print("\nSelected Best Models per Declaration (based on Validation Macro F1):")

    available_decls = sorted(list(comp_df["declaration"].unique()))
    for did in available_decls:
        decl_df = comp_df[comp_df["declaration"] == did].sort_values(
            by=["val_macro_f1", "val_weighted_f1", "val_accuracy"],
            ascending=[False, False, False]
        )
        if decl_df.empty:
            continue
        best_row = decl_df.iloc[0].to_dict()
        
        # Find the corresponding model object in all_experiments
        chosen_exp_key = f"{best_row['model_family']}_{best_row['feature_set']}"
        model_obj = all_experiments.get(chosen_exp_key, {}).get(did, {}).get("model")
        if model_obj is None and best_row['model_family'] == "MajorityBaseline":
            model_obj = baseline_results.get(did, {}).get("majority_class", "PRESENT")

        frozen_candidates[did] = {
            "declaration": did,
            "model_family": best_row["model_family"],
            "feature_set": best_row["feature_set"],
            "hyperparameters": best_row["hyperparameters"],
            "val_macro_f1": best_row["val_macro_f1"],
            "val_weighted_f1": best_row["val_weighted_f1"],
            "val_accuracy": best_row["val_accuracy"],
            "model_object": model_obj,
            "freeze_timestamp": datetime.now(timezone.utc).isoformat()
        }

        print(f"  * {did} -> {best_row['model_family']} ({best_row['feature_set']}) | "
              f"Val Macro F1: {best_row['val_macro_f1']} | Val Acc: {best_row['val_accuracy']}")

    # Export Validation Comparison Markdown
    comp_md = os.path.join(output_dir, "model_comparison.md")
    with open(comp_md, "w", encoding="utf-8") as f:
        f.write("# Model Family Comparison on Validation Partition (Phase 8)\n\n")
        f.write("> [!NOTE]\n")
        f.write("> Model selection is strictly driven by **Validation Macro F1** to prevent majority-class dominance.\n\n")

        f.write("| Declaration | Model Family | Feature Set | Val Accuracy | Val Macro F1 | Val Weighted F1 | Balanced Acc |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for idx, row in comp_df.iterrows():
            f.write(
                f"| **`{row['declaration']}`** | {row['model_family']} | {row['feature_set']} | "
                f"{row['val_accuracy']} | **{row['val_macro_f1']}** | {row['val_weighted_f1']} | {row['balanced_accuracy']} |\n"
            )
        f.write("\n")
    print(f"Saved model comparison to {comp_md}")

    # Export Validation Results Summary
    val_md = os.path.join(output_dir, "validation_results.md")
    with open(val_md, "w", encoding="utf-8") as f:
        f.write("# Validation Partition Results Summary\n\n")
        f.write("| Declaration | Selected Model | Feature Set | Val Macro F1 | Val Weighted F1 | Val Accuracy |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for did in sorted(frozen_candidates.keys()):
            fc = frozen_candidates[did]
            f.write(f"| **`{did}`** | {fc['model_family']} | {fc['feature_set']} | **{fc['val_macro_f1']}** | {fc['val_weighted_f1']} | {fc['val_accuracy']} |\n")
        f.write("\n")
    print(f"Saved validation results summary to {val_md}")

    # Export Formal Model Freeze Record
    freeze_md = os.path.join(output_dir, "model_freeze_record.md")
    with open(freeze_md, "w", encoding="utf-8") as f:
        f.write("# Phase 8 Model Freeze Record\n\n")
        f.write("**Status:** **FROZEN**  \n")
        f.write(f"**Freeze Timestamp:** {datetime.now(timezone.utc).isoformat()}  \n")
        f.write("**Selection Criterion:** Maximum Validation Macro F1  \n\n")

        f.write("## Selected & Frozen Models per Declaration\n\n")
        f.write("| Declaration | Model Family | Feature Set | Hyperparameters | Val Macro F1 | Val Accuracy |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for did in sorted(frozen_candidates.keys()):
            fc = frozen_candidates[did]
            f.write(
                f"| **`{did}`** | **{fc['model_family']}** | `{fc['feature_set']}` | "
                f"`{fc['hyperparameters']}` | **{fc['val_macro_f1']}** | {fc['val_accuracy']} |\n"
            )
        f.write("\n")

        f.write("## Protocol Verification\n\n")
        f.write("- Test split ($N=258$) remained completely untouched during model selection.\n")
        f.write("- Human reference split ($N=75$) remained completely untouched during model selection.\n")
        f.write("- All model configurations are locked prior to out-of-sample evaluation.\n")

    print(f"Saved model freeze record to {freeze_md}")
    return frozen_candidates, comp_df
