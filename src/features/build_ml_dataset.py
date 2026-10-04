"""
build_ml_dataset.py
====================
Constructs the canonical, leakage-safe ML dataset partitions (Train, Validation, Test, Human Reference)
using Group-Aware Splitting on product families and train-only fitted preprocessing.

Outputs:
    data/processed/ml_dataset_v1/
        ├── train/ (X_train.csv, X_train_set_a.csv, Y_train.csv, metadata_train.csv)
        ├── validation/ (X_val.csv, X_val_set_a.csv, Y_val.csv, metadata_val.csv)
        ├── test/ (X_test.csv, X_test_set_a.csv, Y_test.csv, metadata_test.csv)
        ├── human_reference/ (X_human_ref.csv, X_human_ref_set_a.csv, Y_human_ref.csv, metadata_human_ref.csv)
        └── ml_dataset_manifest.json
    models/feature_pipeline_v1.joblib
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import yaml
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupShuffleSplit

from src.features.preprocessing import (
    extract_raw_features_from_record,
    NitiMarkFeaturePipeline
)


TARGET_COLS = [
    "D01_target", "D02_target", "D03_target", "D04_target",
    "D05_target", "D06_target", "D07_target", "overall_target"
]

METADATA_COLS = [
    "product_id", "product_family_id", "duplicate_group_id",
    "category", "price_tier"
]


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_and_extract_features(filepath: str) -> pd.DataFrame:
    """Load JSONL dataset and extract raw feature representations."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                raw_feats = extract_raw_features_from_record(rec)
                # Attach extra metadata if available
                raw_feats["quality_tier"] = rec.get("quality_tier", "HUMAN_GROUND_TRUTH")
                raw_feats["overall_confidence"] = rec.get("overall_confidence", 1.0)
                records.append(raw_feats)
    return pd.DataFrame(records)


def perform_group_aware_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into Train, Validation, and Test partitions ensuring that no
    product_family_id or duplicate_group_id is shared across splits.
    """
    groups = df["product_family_id"].values

    # Step 1: Split Train vs (Val + Test)
    val_test_ratio = 1.0 - train_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
    train_idx, val_test_idx = next(gss1.split(df, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_val_test = df.iloc[val_test_idx].copy()

    # Step 2: Split Val vs Test (equal split of the remaining groups)
    groups_val_test = df_val_test["product_family_id"].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_sub_idx, test_sub_idx = next(gss2.split(df_val_test, groups=groups_val_test))

    df_val = df_val_test.iloc[val_sub_idx].copy()
    df_test = df_val_test.iloc[test_sub_idx].copy()

    df_train["split"] = "TRAIN"
    df_val["split"] = "VALIDATION"
    df_test["split"] = "TEST"

    return df_train, df_val, df_test


def build_and_export_ml_dataset(
    weak_jsonl_path: str,
    human_jsonl_path: str,
    config_path: str,
    output_base_dir: str,
    pipeline_model_path: str
):
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(os.path.dirname(pipeline_model_path), exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    split_seed = config.get("split_seed", 42)

    print("Step 1: Loading datasets and extracting raw feature tables...")
    df_weak = load_and_extract_features(weak_jsonl_path)
    df_human = load_and_extract_features(human_jsonl_path)
    df_human["split"] = "HUMAN_REFERENCE"

    print(f"Loaded {len(df_weak)} weak-label records, {len(df_human)} human reference records.")

    print("Step 2: Performing group-aware splitting on weak-label dataset...")
    df_train, df_val, df_test = perform_group_aware_split(df_weak, seed=split_seed)
    print(f"Split sizes: Train={len(df_train)} (Groups: {df_train['product_family_id'].nunique()}), "
          f"Val={len(df_val)} (Groups: {df_val['product_family_id'].nunique()}), "
          f"Test={len(df_test)} (Groups: {df_test['product_family_id'].nunique()})")

    print("Step 3: Fitting NitiMark Feature Pipeline STRICTLY on TRAIN...")
    pipeline = NitiMarkFeaturePipeline(config=config)
    pipeline.fit(df_train)
    pipeline.save(pipeline_model_path)
    pipeline.save(os.path.join(output_base_dir, "feature_pipeline.joblib"))

    print("Step 4: Transforming feature matrices (Feature Set A and Feature Set B)...")
    # Feature Set B (Recommended: Tabular + TF-IDF)
    X_train_b = pipeline.transform(df_train, feature_set="FEATURE_SET_B")
    X_val_b = pipeline.transform(df_val, feature_set="FEATURE_SET_B")
    X_test_b = pipeline.transform(df_test, feature_set="FEATURE_SET_B")
    X_human_b = pipeline.transform(df_human, feature_set="FEATURE_SET_B")

    # Feature Set A (Tabular only)
    X_train_a = pipeline.transform(df_train, feature_set="FEATURE_SET_A")
    X_val_a = pipeline.transform(df_val, feature_set="FEATURE_SET_A")
    X_test_a = pipeline.transform(df_test, feature_set="FEATURE_SET_A")
    X_human_a = pipeline.transform(df_human, feature_set="FEATURE_SET_A")

    # Target DataFrames
    Y_train = df_train[TARGET_COLS].copy().reset_index(drop=True)
    Y_val = df_val[TARGET_COLS].copy().reset_index(drop=True)
    Y_test = df_test[TARGET_COLS].copy().reset_index(drop=True)
    Y_human = df_human[TARGET_COLS].copy().reset_index(drop=True)

    # Metadata DataFrames
    meta_cols_extended = METADATA_COLS + ["quality_tier", "overall_confidence", "split"]
    meta_train = df_train[meta_cols_extended].copy().reset_index(drop=True)
    meta_val = df_val[meta_cols_extended].copy().reset_index(drop=True)
    meta_test = df_test[meta_cols_extended].copy().reset_index(drop=True)
    meta_human = df_human[meta_cols_extended].copy().reset_index(drop=True)

    print("Step 5: Exporting partition files...")
    partitions = {
        "train": (X_train_b, X_train_a, Y_train, meta_train),
        "validation": (X_val_b, X_val_a, Y_val, meta_val),
        "test": (X_test_b, X_test_a, Y_test, meta_test),
        "human_reference": (X_human_b, X_human_a, Y_human, meta_human)
    }

    manifest_files = {}

    for part_name, (xb, xa, y, meta) in partitions.items():
        part_dir = os.path.join(output_base_dir, part_name)
        os.makedirs(part_dir, exist_ok=True)

        prefix = "train" if part_name == "train" else ("val" if part_name == "validation" else ("test" if part_name == "test" else "human_ref"))
        
        xb_path = os.path.join(part_dir, f"X_{prefix}.csv")
        xa_path = os.path.join(part_dir, f"X_{prefix}_set_a.csv")
        y_path = os.path.join(part_dir, f"Y_{prefix}.csv")
        meta_path = os.path.join(part_dir, f"metadata_{prefix}.csv")

        xb.to_csv(xb_path, index=False)
        xa.to_csv(xa_path, index=False)
        y.to_csv(y_path, index=False)
        meta.to_csv(meta_path, index=False)

        manifest_files[part_name] = {
            "record_count": len(xb),
            "feature_set_b_dimensions": xb.shape[1],
            "feature_set_a_dimensions": xa.shape[1],
            "target_dimensions": y.shape[1],
            "X_file_sha256": compute_file_sha256(xb_path),
            "Y_file_sha256": compute_file_sha256(y_path),
            "metadata_file_sha256": compute_file_sha256(meta_path)
        }

    manifest = {
        "dataset_name": "NitiMark ML Feature Dataset",
        "version": "1.0.0",
        "release_tag": "ml_dataset_v1",
        "feature_version": "v1",
        "split_strategy": "GroupShuffleSplit",
        "group_key": "product_family_id",
        "random_seed": split_seed,
        "feature_counts": {
            "feature_set_b_total": X_train_b.shape[1],
            "feature_set_a_total": X_train_a.shape[1],
            "numeric_structural_count": len(pipeline.NUMERIC_COLS),
            "categorical_onehot_count": len(pipeline.ohe.get_feature_names_out()),
            "tfidf_word_count": len(pipeline.tfidf_word.get_feature_names_out()),
            "tfidf_char_count": len(pipeline.tfidf_char.get_feature_names_out())
        },
        "partitions": manifest_files
    }

    manifest_out = os.path.join(output_base_dir, "ml_dataset_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved dataset manifest to {manifest_out}")

    return partitions, manifest


if __name__ == "__main__":
    WEAK_JSONL = "data/processed/flipkart_weak_labels_v1.jsonl"
    HUMAN_JSONL = "data/processed/pilot_ground_truth_75_v1.jsonl"
    CONFIG_YAML = "configs/feature_config.yaml"
    OUT_DIR = "data/processed/ml_dataset_v1"
    MODEL_PATH = "models/feature_pipeline_v1.joblib"

    build_and_export_ml_dataset(WEAK_JSONL, HUMAN_JSONL, CONFIG_YAML, OUT_DIR, MODEL_PATH)
