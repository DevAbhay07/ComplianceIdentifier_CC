"""
product_level_leakage_audit.py
==============================
Performs comprehensive data leakage, target contamination, and split isolation audits
for Phase 10 product-level compliance modeling.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ml_dataset_v1")
FORBIDDEN_FEATURE_SUBSTRINGS = [
    "target", "overall_target", "label_source", "label_confidence",
    "rule_derived", "compliance_score", "ground_truth", "human_verified",
    "d01_target", "d02_target", "d03_target", "d04_target", "d05_target",
    "d06_target", "d07_target"
]


def check_feature_matrix_leakage() -> Tuple[bool, List[str]]:
    """Check that X matrices contain zero target or metadata leakage columns."""
    errors = []
    splits = ["train", "validation", "test", "human_reference"]
    
    for split in splits:
        prefix = "human_ref" if split == "human_reference" else ("val" if split == "validation" else split)
        x_path = os.path.join(DATA_DIR, split, f"X_{prefix}.csv")
        x_a_path = os.path.join(DATA_DIR, split, f"X_{prefix}_set_a.csv")

        for p in [x_path, x_a_path]:
            if not os.path.exists(p):
                errors.append(f"Missing feature file: {p}")
                continue
            
            # Read first row for header check
            df_head = pd.read_csv(p, nrows=2)
            cols = [c.lower() for c in df_head.columns]

            for c in cols:
                for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
                    if forbidden == c or forbidden in c:
                        errors.append(f"Leakage in {p}: column '{c}' matches forbidden pattern '{forbidden}'")

    return len(errors) == 0, errors


def check_group_disjointness() -> Tuple[bool, List[str]]:
    """Verify group-level (product_family_id) strict partition across splits."""
    errors = []
    meta_train = pd.read_csv(os.path.join(DATA_DIR, "train", "metadata_train.csv"))
    meta_val = pd.read_csv(os.path.join(DATA_DIR, "validation", "metadata_val.csv"))
    meta_test = pd.read_csv(os.path.join(DATA_DIR, "test", "metadata_test.csv"))
    meta_human = pd.read_csv(os.path.join(DATA_DIR, "human_reference", "metadata_human_ref.csv"))

    fams_train = set(meta_train["product_family_id"].dropna())
    fams_val = set(meta_val["product_family_id"].dropna())
    fams_test = set(meta_test["product_family_id"].dropna())
    fams_human = set(meta_human["product_family_id"].dropna())

    train_val_overlap = fams_train.intersection(fams_val)
    train_test_overlap = fams_train.intersection(fams_test)
    val_test_overlap = fams_val.intersection(fams_test)

    if train_val_overlap:
        errors.append(f"Train/Val family overlap: {len(train_val_overlap)} groups ({list(train_val_overlap)[:3]})")
    if train_test_overlap:
        errors.append(f"Train/Test family overlap: {len(train_test_overlap)} groups ({list(train_test_overlap)[:3]})")
    if val_test_overlap:
        errors.append(f"Val/Test family overlap: {len(val_test_overlap)} groups ({list(val_test_overlap)[:3]})")

    return len(errors) == 0, errors


def check_external_dataset_isolation() -> Tuple[bool, List[str]]:
    """Verify human reference (N=75) and real-product (N=20) IDs are strictly absent in train."""
    errors = []
    meta_train = pd.read_csv(os.path.join(DATA_DIR, "train", "metadata_train.csv"))
    meta_human = pd.read_csv(os.path.join(DATA_DIR, "human_reference", "metadata_human_ref.csv"))

    train_ids = set(meta_train["product_id"])
    human_ids = set(meta_human["product_id"])

    human_train_overlap = train_ids.intersection(human_ids)
    if human_train_overlap:
        errors.append(f"Human Reference product IDs leaked into training set: {human_train_overlap}")

    # Check 20 real product IDs
    real_ids = {f"real_prod_{i:02d}" for i in range(1, 21)}
    real_train_overlap = train_ids.intersection(real_ids)
    if real_train_overlap:
        errors.append(f"Real product validation IDs found in training set: {real_train_overlap}")

    return len(errors) == 0, errors


def run_full_leakage_audit() -> Dict[str, Any]:
    print("=================================================================")
    print("   NITIMARK-ML — PRODUCT-LEVEL DATA LEAKAGE AUDIT (PHASE 10)    ")
    print("=================================================================\n")

    feat_pass, feat_errs = check_feature_matrix_leakage()
    grp_pass, grp_errs = check_group_disjointness()
    iso_pass, iso_errs = check_external_dataset_isolation()

    all_passed = feat_pass and grp_pass and iso_pass

    print(f"1. Feature Column Leakage Check       : {'PASS' if feat_pass else 'FAIL'}")
    if feat_errs:
        for e in feat_errs:
            print(f"   ! {e}")

    print(f"2. Group Partition Disjointness Check : {'PASS' if grp_pass else 'FAIL'}")
    if grp_errs:
        for e in grp_errs:
            print(f"   ! {e}")

    print(f"3. External Set Isolation Check       : {'PASS' if iso_pass else 'FAIL'}")
    if iso_errs:
        for e in iso_errs:
            print(f"   ! {e}")

    print("-" * 65)
    print(f"OVERALL LEAKAGE AUDIT RESULT         : {'PASS' if all_passed else 'FAIL'}")
    print("=================================================================\n")

    return {
        "passed": all_passed,
        "feature_leakage_passed": feat_pass,
        "group_disjointness_passed": grp_pass,
        "external_isolation_passed": iso_pass,
        "errors": feat_errs + grp_errs + iso_errs
    }


if __name__ == "__main__":
    res = run_full_leakage_audit()
    if not res["passed"]:
        sys.exit(1)
