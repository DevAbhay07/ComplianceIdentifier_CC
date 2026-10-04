"""
feature_importance.py
=====================
Extracts, ranks, and analyzes feature importance metrics for each frozen model
across declarations D01 through D07. Conducts leakage screening on top predictive features.

Outputs:
    reports/phase_8/feature_importance/D01_feature_importance.md (.csv)
    ...
    reports/phase_8/feature_importance/D07_feature_importance.md (.csv)
    reports/phase_8/feature_importance/feature_importance_summary.md
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import numpy as np
from typing import Dict, Any, List

DECLARATIONS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]

FORBIDDEN_LEAKAGE_SUBSTRINGS = [
    "target",
    "rule_confidence",
    "overall_compliance",
    "provenance",
    "verification_timestamp",
    "annotation_status"
]


def extract_feature_importances(
    frozen_models: Dict[str, Dict[str, Any]],
    feature_names_a: List[str],
    feature_names_b: List[str],
    output_dir: str
) -> Dict[str, pd.DataFrame]:
    """
    Extract top features per declaration and generate reports.
    """
    fi_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(fi_dir, exist_ok=True)
    print("\n--- Extracting Feature Importances & Screening Leakage ---")

    fi_summary_rows = []
    all_importances = {}

    for did in DECLARATIONS:
        fc = frozen_models[did]
        model_obj = fc["model_object"]
        model_family = fc["model_family"]
        fset = fc["feature_set"]
        feature_names = feature_names_b if fset == "FEATURE_SET_B" else feature_names_a

        if isinstance(model_obj, str) or model_family == "MajorityBaseline":
            # Baseline predictor has no learned feature weights
            df_fi = pd.DataFrame([{"feature": "CONSTANT_BASELINE_NO_FEATURES", "importance": 1.0, "rank": 1}])
            all_importances[did] = df_fi
            fi_summary_rows.append({
                "declaration": did,
                "model_family": model_family,
                "top_1_feature": "N/A (Constant Baseline)",
                "top_1_score": 1.0,
                "leakage_passed": True
            })
            continue

        # Extract importances based on family
        raw_importances = None
        if hasattr(model_obj, "feature_importances_"):
            raw_importances = model_obj.feature_importances_
        elif hasattr(model_obj, "model") and hasattr(model_obj.model, "feature_importances_"):
            # XGBoost wrapper
            raw_importances = model_obj.model.feature_importances_
        elif hasattr(model_obj, "coef_"):
            # Logistic Regression: aggregate across multi-class using mean absolute magnitude
            coef = model_obj.coef_
            if coef.ndim == 1:
                raw_importances = np.abs(coef)
            else:
                raw_importances = np.mean(np.abs(coef), axis=0)

        if raw_importances is None or len(raw_importances) != len(feature_names):
            df_fi = pd.DataFrame([{"feature": "EXTRACTION_UNAVAILABLE", "importance": 0.0, "rank": 1}])
            all_importances[did] = df_fi
            continue

        df_fi = pd.DataFrame({
            "feature": feature_names,
            "importance": raw_importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        df_fi["rank"] = df_fi.index + 1
        all_importances[did] = df_fi

        # Leakage screening: check top 50 features for forbidden substrings
        top_50_features = df_fi.head(50)["feature"].tolist()
        leaks = []
        for feat in top_50_features:
            for sub in FORBIDDEN_LEAKAGE_SUBSTRINGS:
                if sub in feat.lower():
                    leaks.append((feat, sub))

        leakage_passed = len(leaks) == 0
        if not leakage_passed:
            print(f"  [WARNING] Potential leakage flagged in {did}: {leaks}")

        top_1_feat = df_fi.iloc[0]["feature"]
        top_1_score = round(float(df_fi.iloc[0]["importance"]), 5)

        fi_summary_rows.append({
            "declaration": did,
            "model_family": model_family,
            "top_1_feature": top_1_feat,
            "top_1_score": top_1_score,
            "leakage_passed": leakage_passed
        })

        # Save individual declaration CSV
        csv_path = os.path.join(fi_dir, f"{did}_feature_importance.csv")
        df_fi.head(50).to_csv(csv_path, index=False)

        # Save individual declaration Markdown
        md_path = os.path.join(fi_dir, f"{did}_feature_importance.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# `{did}` Top Predictive Features ({model_family})\n\n")
            f.write(f"- **Model Family:** {model_family}\n")
            f.write(f"- **Feature Set:** `{fset}` ({len(feature_names)} features)\n")
            f.write(f"- **Leakage Screen:** {'PASSED (Zero leakage detected)' if leakage_passed else 'FAILED'}\n\n")
            f.write("| Rank | Feature Name | Importance Score |\n")
            f.write("| :--- | :--- | :--- |\n")
            for idx, row in df_fi.head(20).iterrows():
                f.write(f"| {row['rank']} | `{row['feature']}` | {round(row['importance'], 5)} |\n")
            f.write("\n")

    # Save summary report
    summary_md = os.path.join(fi_dir, "feature_importance_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Feature Importance & Interpretability Summary (Phase 8)\n\n")
        f.write("| Declaration | Selected Model | Top 1 Predictive Feature | Importance Score | Leakage Screen |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for r in fi_summary_rows:
            f.write(f"| **`{r['declaration']}`** | {r['model_family']} | `{r['top_1_feature']}` | {r['top_1_score']} | {'✅ PASS' if r['leakage_passed'] else '❌ FAIL'} |\n")
        f.write("\n")

    print(f"Saved feature importance reports to {fi_dir}")
    return all_importances
