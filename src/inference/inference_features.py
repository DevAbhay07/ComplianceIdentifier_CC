"""
inference_features.py
=====================
Transforms input product data (images, OCR text, product title, description,
specifications) into the exact feature representation expected by the frozen
Phase 8 ML models using models/feature_pipeline_v1.joblib.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import joblib
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from src.features.preprocessing import extract_raw_features_from_record, NitiMarkFeaturePipeline

DEFAULT_PIPELINE_PATH = "models/feature_pipeline_v1.joblib"


def build_inference_feature_vectors(
    title: str = "",
    description: str = "",
    raw_ocr_text: str = "",
    specifications: Optional[Dict[str, Any]] = None,
    retail_price: Optional[float] = None,
    discounted_price: Optional[float] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    image_count: int = 1,
    pipeline_path: str = DEFAULT_PIPELINE_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Constructs exact Feature Set A (219 features) and Feature Set B (719 features)
    feature matrices from raw inputs using the frozen Phase 7/8 feature pipeline.

    Returns:
        Tuple[X_set_a, X_set_b, raw_features_dict]
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"FROZEN_PIPELINE_MISSING: Feature pipeline artifact not found at {pipeline_path}. "
            "Ensure Phase 7 models/feature_pipeline_v1.joblib exists."
        )

    # Combine description and OCR text
    combined_desc = description.strip()
    if raw_ocr_text.strip():
        if combined_desc:
            combined_desc = f"{combined_desc}\n[Visible Packaging OCR]:\n{raw_ocr_text.strip()}"
        else:
            combined_desc = raw_ocr_text.strip()

    # If title is empty but OCR text exists, derive candidate title from first line of OCR
    final_title = title.strip()
    if not final_title and raw_ocr_text.strip():
        first_line = raw_ocr_text.strip().split("\n")[0][:120].strip()
        final_title = first_line

    # Infer brand if not provided
    final_brand = (brand or "").strip()
    if not final_brand and final_title:
        first_word = final_title.split()[0]
        if len(first_word) > 2 and first_word.isalpha():
            final_brand = first_word

    # Construct standard product record
    record = {
        "product_id": "INFERENCE_SAMPLE",
        "title": final_title,
        "product_name": final_title,
        "description": combined_desc,
        "specifications": specifications or {},
        "retail_price": retail_price,
        "discounted_price": discounted_price,
        "brand": final_brand,
        "category": category or "General Commodity",
        "image_urls": ["sample_image"] if image_count > 0 else []
    }

    # Extract raw features via Phase 7 logic
    raw_feats = extract_raw_features_from_record(record)
    df_raw = pd.DataFrame([raw_feats])

    # Load frozen feature pipeline
    pipeline = joblib.load(pipeline_path)

    # Transform through fitted transformers
    X_set_a = pipeline.transform(df_raw, feature_set="FEATURE_SET_A")
    X_set_b = pipeline.transform(df_raw, feature_set="FEATURE_SET_B")

    return X_set_a, X_set_b, raw_feats
