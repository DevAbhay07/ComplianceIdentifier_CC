"""
schema_validator.py
===================
Validates that generated inference feature vectors strictly conform to
the frozen Phase 7/8 training feature schemas (dimension, names, ordering, dtypes, missingness).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

DEFAULT_PIPELINE_PATH = "models/feature_pipeline_v1.joblib"


def validate_inference_feature_schema(
    X_set_a: pd.DataFrame,
    X_set_b: pd.DataFrame,
    pipeline_path: str = DEFAULT_PIPELINE_PATH
) -> Dict[str, Any]:
    """
    Validates feature matrices against the frozen training pipeline schema.

    Returns:
        Dict[str, Any] with validation diagnostics and error messages.
    """
    errors: List[str] = []

    if not os.path.exists(pipeline_path):
        return {
            "valid": False,
            "errors": [f"Feature pipeline not found at {pipeline_path}"],
            "feature_set_a_count": len(X_set_a.columns) if isinstance(X_set_a, pd.DataFrame) else 0,
            "feature_set_b_count": len(X_set_b.columns) if isinstance(X_set_b, pd.DataFrame) else 0
        }

    pipeline = joblib.load(pipeline_path)
    expected_names_a = pipeline.feature_names_a_
    expected_names_b = pipeline.feature_names_b_

    # 1. Dimension checks
    if len(X_set_a.columns) != len(expected_names_a):
        errors.append(
            f"DIMENSION_MISMATCH (Set A): Expected {len(expected_names_a)} features, got {len(X_set_a.columns)}"
        )

    if len(X_set_b.columns) != len(expected_names_b):
        errors.append(
            f"DIMENSION_MISMATCH (Set B): Expected {len(expected_names_b)} features, got {len(X_set_b.columns)}"
        )

    # 2. Column names & ordering checks
    if list(X_set_a.columns) != list(expected_names_a):
        errors.append("COLUMN_MISMATCH (Set A): Column names or ordering do not match frozen schema.")

    if list(X_set_b.columns) != list(expected_names_b):
        errors.append("COLUMN_MISMATCH (Set B): Column names or ordering do not match frozen schema.")

    # 3. Missingness & Non-finite checks
    if X_set_a.isna().any().any():
        errors.append("MISSING_VALUES_FOUND: NaN values detected in Feature Set A vector.")

    if X_set_b.isna().any().any():
        errors.append("MISSING_VALUES_FOUND: NaN values detected in Feature Set B vector.")

    if np.isinf(X_set_a.values).any():
        errors.append("INFINITE_VALUES_FOUND: Infinite values detected in Feature Set A vector.")

    if np.isinf(X_set_b.values).any():
        errors.append("INFINITE_VALUES_FOUND: Infinite values detected in Feature Set B vector.")

    # 4. Dtype checks
    if not np.issubdtype(X_set_a.values.dtype, np.number):
        errors.append(f"INVALID_DTYPE (Set A): Non-numeric dtype {X_set_a.values.dtype}")

    if not np.issubdtype(X_set_b.values.dtype, np.number):
        errors.append(f"INVALID_DTYPE (Set B): Non-numeric dtype {X_set_b.values.dtype}")

    is_valid = len(errors) == 0

    return {
        "valid": is_valid,
        "errors": errors,
        "feature_set_a_dims": len(X_set_a.columns),
        "feature_set_b_dims": len(X_set_b.columns),
        "sample_count": len(X_set_a),
        "pipeline_version": "v1"
    }
