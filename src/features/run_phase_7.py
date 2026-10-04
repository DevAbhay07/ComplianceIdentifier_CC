"""
run_phase_7.py
===============
Master execution script for Phase 7: Leakage-Safe Feature Engineering & Dataset Construction.
Runs all pipeline stages end-to-end and outputs complete statistical reports and quality gates.

Usage:
    python src/features/run_phase_7.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import pandas as pd
from datetime import datetime, timezone

from src.features.audit_feature_sources import run_feature_source_audit
from src.features.build_ml_dataset import build_and_export_ml_dataset
from src.features.analyze_ml_dataset import analyze_dataset_partitions
from src.features.feature_leakage_audit import run_feature_leakage_audit
from src.features.phase_7_quality_gate import run_phase_7_quality_gates


def main():
    start_time = time.time()
    print("=" * 70)
    print("NITIMARK ML — PHASE 7: LEAKAGE-SAFE FEATURE ENGINEERING PIPELINE")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Configuration Paths
    WEAK_JSONL = "data/processed/flipkart_weak_labels_v1.jsonl"
    HUMAN_JSONL = "data/processed/pilot_ground_truth_75_v1.jsonl"
    CONFIG_YAML = "configs/feature_config.yaml"
    ML_DATA_DIR = "data/processed/ml_dataset_v1"
    MODEL_PATH = "models/feature_pipeline_v1.joblib"
    REPORTS_DIR = "reports/phase_7"

    # Step 1: Feature Source Audit
    print("\n>>> STAGE 1: Auditing and Classifying Feature Sources...")
    run_feature_source_audit(REPORTS_DIR)

    # Step 2: Build ML Dataset Partitions
    print("\n>>> STAGE 2: Constructing Group-Aware ML Partitions (Train, Val, Test, Human Ref)...")
    build_and_export_ml_dataset(WEAK_JSONL, HUMAN_JSONL, CONFIG_YAML, ML_DATA_DIR, MODEL_PATH)

    # Step 3: Analyze Dataset & Imbalance
    print("\n>>> STAGE 3: Profiling Dataset Dimensions, Missingness & Class Imbalance...")
    analyze_dataset_partitions(ML_DATA_DIR, REPORTS_DIR)

    # Step 4: Run Feature Leakage Audit
    print("\n>>> STAGE 4: Running 10-Point Feature Leakage Audit...")
    X_train = pd.read_csv(os.path.join(ML_DATA_DIR, "train/X_train.csv"))
    X_val = pd.read_csv(os.path.join(ML_DATA_DIR, "validation/X_val.csv"))
    X_test = pd.read_csv(os.path.join(ML_DATA_DIR, "test/X_test.csv"))
    X_human = pd.read_csv(os.path.join(ML_DATA_DIR, "human_reference/X_human_ref.csv"))

    meta_train = pd.read_csv(os.path.join(ML_DATA_DIR, "train/metadata_train.csv"))
    meta_val = pd.read_csv(os.path.join(ML_DATA_DIR, "validation/metadata_val.csv"))
    meta_test = pd.read_csv(os.path.join(ML_DATA_DIR, "test/metadata_test.csv"))
    meta_human = pd.read_csv(os.path.join(ML_DATA_DIR, "human_reference/metadata_human_ref.csv"))

    leakage_status, _ = run_feature_leakage_audit(
        X_train, X_val, X_test, X_human,
        meta_train, meta_val, meta_test, meta_human,
        os.path.join(REPORTS_DIR, "feature_leakage_audit.md")
    )

    # Step 5: Master Phase 7 Quality Gate
    print("\n>>> STAGE 5: Evaluating Master Phase 7 Quality Gate...")
    qgate_status, _ = run_phase_7_quality_gates(
        ML_DATA_DIR,
        os.path.join(REPORTS_DIR, "PHASE_7_QUALITY_GATE.md")
    )

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"PHASE 7 PIPELINE COMPLETED IN {elapsed}s")
    print(f"Leakage Audit Status: {leakage_status}")
    print(f"Quality Gate Status:  {qgate_status}")
    print(f"Phase 8 Readiness:    {'PASS' if qgate_status == 'PASS' else 'BLOCKED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
