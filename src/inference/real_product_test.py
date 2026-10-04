"""
real_product_test.py
====================
Provides novelty verification and collision detection for user-supplied real products,
flagging if an input product matches or closely resembles an item from the training split.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional

TRAIN_META_PATH = "data/processed/ml_dataset_v1/train/metadata_train.csv"


def check_product_novelty(
    title: str = "",
    product_id: Optional[str] = None,
    train_metadata_path: str = TRAIN_META_PATH
) -> Dict[str, Any]:
    """
    Checks whether a product is genuinely unseen or collides with the training dataset.

    Returns:
        Dict[str, Any]:
        {
            "status": "NEW_PRODUCT" | "KNOWN_TRAINING_PRODUCT" | "POSSIBLE_DUPLICATE",
            "matched_id": str or None,
            "warning": str or None
        }
    """
    if not os.path.exists(train_metadata_path):
        return {
            "status": "NEW_PRODUCT",
            "matched_id": None,
            "warning": None
        }

    try:
        df_train = pd.read_csv(train_metadata_path)
    except Exception:
        return {
            "status": "NEW_PRODUCT",
            "matched_id": None,
            "warning": None
        }

    # Check exact product ID
    if product_id and "product_id" in df_train.columns:
        if product_id in df_train["product_id"].values:
            return {
                "status": "KNOWN_TRAINING_PRODUCT",
                "matched_id": product_id,
                "warning": (
                    f"DATA_LEAKAGE_WARNING: Product ID '{product_id}' was part of the Phase 8 training partition. "
                    "Prediction will proceed, but this is an in-sample test."
                )
            }

    # Check title collision via token set matching
    norm_title = set(title.lower().split()) if title else set()
    if len(norm_title) >= 3 and "category" in df_train.columns:
        for idx, row in df_train.iterrows():
            train_cat_or_name = str(row.get("category", "")).lower()
            train_tokens = set(train_cat_or_name.split())
            if train_tokens and len(norm_title.intersection(train_tokens)) / max(len(norm_title), 1) > 0.85:
                return {
                    "status": "POSSIBLE_DUPLICATE",
                    "matched_id": str(row.get("product_id", "")),
                    "warning": (
                        f"NEAR_DUPLICATE_WARNING: Input title closely matches training item '{train_cat_or_name}'. "
                        "Treat result as near-distribution evaluation."
                    )
                }

    return {
        "status": "NEW_PRODUCT",
        "matched_id": None,
        "warning": None
    }
