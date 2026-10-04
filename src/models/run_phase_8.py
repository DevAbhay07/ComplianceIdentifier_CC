"""
run_phase_8.py
==============
Master orchestration pipeline for Phase 8: Multi-Label ML Model Training & Experimentation.

Executes:
1. Data loading and partition slicing (Train, Val, Test, Human Reference).
2. Majority-Class baseline evaluation.
3. Multi-family model training & hyperparameter search (LR, RF, XGBoost on Set A & Set B).
4. Validation-driven model selection & freezing.
5. Locked evaluation on weak-label Test partition (N=258).
6. Out-of-sample transfer evaluation on Human Ground Truth Reference (N=75).
7. Feature importance extraction & anti-leakage verification.
8. Error analysis, category stratification, and probability calibration.
9. Execution of all 20 Quality Gates (GATE_A - GATE_T).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import json
import pandas as pd
import numpy as np

from src.models.train_logistic import train_logistic_grid
from src.models.train_random_forest import train_random_forest_grid
from src.models.train_xgboost import train_xgboost_grid
from src.evaluation.compare_models import evaluate_majority_baseline, compare_and_select_models
from src.evaluation.evaluate_human_reference import run_locked_evaluations
from src.evaluation.feature_importance import extract_feature_importances
from src.evaluation.error_analysis import run_error_analysis
from src.models.phase_8_quality_gate import evaluate_phase_8_quality_gates


def main():
    start_time = time.time()
    print("\n" + "=" * 70)
    print("      NITIMARK ML PREDICTION LAYER - PHASE 8 TRAINING PIPELINE")
    print("=" * 70)

    base_dir = "/home/devabhay07/Desktop/MLmodelV2"
    data_dir = os.path.join(base_dir, "data/processed/ml_dataset_v1")
    reports_dir = os.path.join(base_dir, "reports/phase_8")
    models_dir = os.path.join(base_dir, "models/phase_8")
    exp_dir = os.path.join(base_dir, "experiments/phase_8")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Load Data Splits
    print("\n[Step 1/8] Loading ML Partitions...")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")
    human_dir = os.path.join(data_dir, "human_reference")

    X_train_b = pd.read_csv(os.path.join(train_dir, "X_train.csv"))
    X_train_a = pd.read_csv(os.path.join(train_dir, "X_train_set_a.csv"))
    Y_train = pd.read_csv(os.path.join(train_dir, "Y_train.csv"))
    meta_train = pd.read_csv(os.path.join(train_dir, "metadata_train.csv"))

    X_val_b = pd.read_csv(os.path.join(val_dir, "X_val.csv"))
    X_val_a = pd.read_csv(os.path.join(val_dir, "X_val_set_a.csv"))
    Y_val = pd.read_csv(os.path.join(val_dir, "Y_val.csv"))
    meta_val = pd.read_csv(os.path.join(val_dir, "metadata_val.csv"))

    X_test_b = pd.read_csv(os.path.join(test_dir, "X_test.csv"))
    X_test_a = pd.read_csv(os.path.join(test_dir, "X_test_set_a.csv"))
    Y_test = pd.read_csv(os.path.join(test_dir, "Y_test.csv"))
    meta_test = pd.read_csv(os.path.join(test_dir, "metadata_test.csv"))

    X_human_b = pd.read_csv(os.path.join(human_dir, "X_human_ref.csv"))
    X_human_a = pd.read_csv(os.path.join(human_dir, "X_human_ref_set_a.csv"))
    Y_human = pd.read_csv(os.path.join(human_dir, "Y_human_ref.csv"))
    meta_human = pd.read_csv(os.path.join(human_dir, "metadata_human_ref.csv"))

    feature_names_a = list(X_train_a.columns)
    feature_names_b = list(X_train_b.columns)

    print(f"  * Train records: {len(X_train_b)} (Set A: {len(feature_names_a)} cols, Set B: {len(feature_names_b)} cols)")
    print(f"  * Validation records: {len(X_val_b)}")
    print(f"  * Test records: {len(X_test_b)}")
    print(f"  * Human Reference records: {len(X_human_b)}")

    # 2. Majority Baseline Evaluation
    print("\n[Step 2/8] Establishing Validation Baselines...")
    baseline_results = evaluate_majority_baseline(Y_train, Y_val)

    # 3. Model Training & Grid Searches
    print("\n[Step 3/8] Training Model Families across Feature Sets A & B...")
    all_experiments = {}

    # Logistic Regression
    all_experiments["LogisticRegression_FEATURE_SET_A"] = train_logistic_grid(
        X_train_a, Y_train, X_val_a, Y_val, feature_set_name="FEATURE_SET_A"
    )
    all_experiments["LogisticRegression_FEATURE_SET_B"] = train_logistic_grid(
        X_train_b, Y_train, X_val_b, Y_val, feature_set_name="FEATURE_SET_B"
    )

    # Random Forest
    all_experiments["RandomForest_FEATURE_SET_A"] = train_random_forest_grid(
        X_train_a, Y_train, X_val_a, Y_val, feature_set_name="FEATURE_SET_A"
    )
    all_experiments["RandomForest_FEATURE_SET_B"] = train_random_forest_grid(
        X_train_b, Y_train, X_val_b, Y_val, feature_set_name="FEATURE_SET_B"
    )

    # XGBoost
    all_experiments["XGBoost_FEATURE_SET_A"] = train_xgboost_grid(
        X_train_a, Y_train, X_val_a, Y_val, feature_set_name="FEATURE_SET_A"
    )
    all_experiments["XGBoost_FEATURE_SET_B"] = train_xgboost_grid(
        X_train_b, Y_train, X_val_b, Y_val, feature_set_name="FEATURE_SET_B"
    )

    # 4. Model Selection & Freezing
    print("\n[Step 4/8] Model Selection & Freezing strictly on Validation Macro F1...")
    frozen_models, comp_df = compare_and_select_models(
        all_experiments=all_experiments,
        baseline_results=baseline_results,
        output_dir=reports_dir
    )

    # 5. Out-of-sample Locked Evaluations (Test & Human Reference)
    print("\n[Step 5/8] Running Locked Evaluations on Test Split and Human Reference Split...")
    test_results, human_results = run_locked_evaluations(
        frozen_models=frozen_models,
        X_val_b=X_val_b,
        X_val_a=X_val_a,
        Y_val=Y_val,
        X_test_b=X_test_b,
        X_test_a=X_test_a,
        Y_test=Y_test,
        X_human_b=X_human_b,
        X_human_a=X_human_a,
        Y_human=Y_human,
        output_dir=reports_dir
    )

    # 6. Feature Importance Extraction & Leakage Screening
    print("\n[Step 6/8] Extracting Feature Importances & Screening for Leakage...")
    all_importances = extract_feature_importances(
        frozen_models=frozen_models,
        feature_names_a=feature_names_a,
        feature_names_b=feature_names_b,
        output_dir=reports_dir
    )

    # 7. Error Analysis, Calibration & Category Stratification
    print("\n[Step 7/8] Conducting Comprehensive Failure & Calibration Analysis...")
    splits_data = {
        "validation": {"X_b": X_val_b, "X_a": X_val_a, "Y": Y_val, "meta": meta_val},
        "test": {"X_b": X_test_b, "X_a": X_test_a, "Y": Y_test, "meta": meta_test},
        "human_reference": {"X_b": X_human_b, "X_a": X_human_a, "Y": Y_human, "meta": meta_human}
    }
    run_error_analysis(
        frozen_models=frozen_models,
        splits_data=splits_data,
        output_dir=reports_dir
    )

    # 8. Run Quality Gates (GATE_A through GATE_T)
    print("\n[Step 8/8] Evaluating Phase 8 Quality Gates...")
    all_passed, gates = evaluate_phase_8_quality_gates(base_dir=base_dir)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"      PHASE 8 PIPELINE COMPLETED IN {elapsed} SECONDS")
    print("=" * 70)


if __name__ == "__main__":
    main()
