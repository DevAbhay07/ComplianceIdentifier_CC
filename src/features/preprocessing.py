"""
preprocessing.py
=================
Leakage-safe feature extraction and transformation pipeline for NitiMark ML.
Extracts independent numeric, structural, categorical, and text TF-IDF features.

Strict Guarantee:
All transformers (Imputer, StandardScaler, OneHotEncoder, TfidfVectorizer)
are fitted ONLY on the TRAIN partition. Transform is applied to Validation,
Test, and Human Reference splits without re-fitting.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import re
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


# Patterns for token count features
MEASUREMENT_REGEX = re.compile(r'\b\d+(?:\.\d+)?\s*(?:ml|l|litre|liter|g|gm|kg|oz|cm|mm|m|pcs|pc|pack|set)\b', re.IGNORECASE)
CURRENCY_REGEX = re.compile(r'(?:₹|rs\.?|inr)\s*\d+', re.IGNORECASE)
CONTACT_REGEX = re.compile(r'\b1800\s*\d+|\+91[\s-]*\d{10}|support@|care@|help@|helpline|customer care', re.IGNORECASE)
ORIGIN_REGEX = re.compile(r'\b(?:india|china|imported|made in|manufactured in|origin)\b', re.IGNORECASE)


def extract_raw_features_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract raw independent features from a single product record."""
    title = str(rec.get("product_name", "") or rec.get("title", "")).strip()
    desc = str(rec.get("description", "") or "").strip()
    
    # Specs serialization
    specs = rec.get("specifications", {})
    if isinstance(specs, str):
        try:
            specs = json.loads(specs)
        except Exception:
            specs = {}
    if not isinstance(specs, dict):
        specs = {}

    spec_items = [f"{k}: {v}" for k, v in specs.items()]
    spec_text = " | ".join(spec_items)

    # Price handling
    retail_price = None
    discounted_price = None
    try:
        if rec.get("retail_price") is not None and str(rec.get("retail_price")).strip() not in ("", "nan", "None"):
            retail_price = float(rec.get("retail_price"))
    except Exception:
        retail_price = None

    try:
        if rec.get("discounted_price") is not None and str(rec.get("discounted_price")).strip() not in ("", "nan", "None"):
            discounted_price = float(rec.get("discounted_price"))
    except Exception:
        discounted_price = None

    discount_amount = 0.0
    discount_pct = 0.0
    if retail_price is not None and discounted_price is not None and retail_price > 0:
        discount_amount = max(0.0, retail_price - discounted_price)
        discount_pct = (discount_amount / retail_price) * 100.0

    # Price tier
    price_tier = "UNKNOWN"
    if retail_price is not None:
        if retail_price < 300:
            price_tier = "BUDGET"
        elif retail_price <= 1000:
            price_tier = "MID_RANGE"
        elif retail_price <= 3000:
            price_tier = "PREMIUM"
        else:
            price_tier = "LUXURY"

    # Image counts
    img_urls = rec.get("image_urls", [])
    if isinstance(img_urls, str):
        try:
            img_urls = json.loads(img_urls)
        except Exception:
            img_urls = []
    img_count = len(img_urls) if isinstance(img_urls, list) else 0

    # Combined text
    all_text = f"{title} {desc} {spec_text}".strip()

    # Raw features dictionary
    raw_feats = {
        # Identifiers / Metadata (for splitting only, NOT for X)
        "product_id": str(rec.get("product_id", "") or rec.get("uniq_id", "")).strip(),
        "product_family_id": str(rec.get("product_family_id", "") or "fam_unknown").strip(),
        "duplicate_group_id": str(rec.get("duplicate_group_id", "") or "dup_unknown").strip(),

        # Numeric structural features
        "title_length": len(title),
        "title_word_count": len(title.split()),
        "desc_length": len(desc),
        "desc_word_count": len(desc.split()),
        "spec_count": len(specs),
        "spec_text_length": len(spec_text),
        "numeric_token_count": len(re.findall(r'\b\d+\b', all_text)),
        "measurement_token_count": len(MEASUREMENT_REGEX.findall(all_text)),
        "currency_token_count": len(CURRENCY_REGEX.findall(all_text)),
        "contact_token_count": len(CONTACT_REGEX.findall(all_text)),
        "origin_token_count": len(ORIGIN_REGEX.findall(all_text)),
        "retail_price": retail_price,
        "discounted_price": discounted_price,
        "discount_amount": discount_amount,
        "discount_percentage": discount_pct,
        "image_count": img_count,

        # Binary structural flags
        "has_description": 1 if len(desc) > 0 else 0,
        "has_specifications": 1 if len(specs) > 0 else 0,
        "has_brand": 1 if bool(str(rec.get("brand", "")).strip()) else 0,
        "has_image": 1 if img_count > 0 else 0,
        "has_price": 1 if retail_price is not None else 0,

        # Categorical features
        "category": str(rec.get("category", "") or "Unknown").strip(),
        "price_tier": price_tier,

        # Raw text fields
        "combined_text": all_text
    }

    # Targets (extracted separately, NEVER in X)
    decls = rec.get("declarations", {})
    for i in range(1, 8):
        did = f"D0{i}"
        d_val = decls.get(did, {})
        if isinstance(d_val, dict):
            raw_feats[f"{did}_target"] = d_val.get("label", "CANNOT_DETERMINE")
        else:
            raw_feats[f"{did}_target"] = str(rec.get(f"{did}_label", "CANNOT_DETERMINE"))

    raw_feats["overall_target"] = rec.get("overall_label", "UNCERTAIN")

    return raw_feats


class NitiMarkFeaturePipeline:
    """
    Leakage-safe multi-modal preprocessing transformer.
    Fitted exclusively on X_train.
    """

    NUMERIC_COLS = [
        "title_length", "title_word_count", "desc_length", "desc_word_count",
        "spec_count", "spec_text_length", "numeric_token_count", "measurement_token_count",
        "currency_token_count", "contact_token_count", "origin_token_count",
        "retail_price", "discounted_price", "discount_amount", "discount_percentage", "image_count",
        "has_description", "has_specifications", "has_brand", "has_image", "has_price"
    ]

    CATEGORICAL_COLS = ["category", "price_tier"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        prep_cfg = self.config.get("preprocessing", {})
        tfidf_cfg = prep_cfg.get("tfidf", {})

        # Transformers
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        self.tfidf_word = TfidfVectorizer(
            max_features=tfidf_cfg.get("word_max_features", 300),
            ngram_range=tuple(tfidf_cfg.get("word_ngram_range", [1, 2])),
            min_df=tfidf_cfg.get("word_min_df", 3)
        )
        self.tfidf_char = TfidfVectorizer(
            max_features=tfidf_cfg.get("char_max_features", 200),
            analyzer="char",
            ngram_range=tuple(tfidf_cfg.get("char_ngram_range", [3, 5])),
            min_df=tfidf_cfg.get("char_min_df", 5)
        )

        self.fitted = False
        self.feature_names_a_: List[str] = []
        self.feature_names_b_: List[str] = []

    def fit(self, df_train: pd.DataFrame):
        """Fit all preprocessing models strictly on training DataFrame."""
        n_samples = len(df_train)
        prep_cfg = self.config.get("preprocessing", {})
        tfidf_cfg = prep_cfg.get("tfidf", {})

        # Adjust min_df dynamically if training sample size is very small (e.g. unit tests)
        word_min_df = min(n_samples, tfidf_cfg.get("word_min_df", 3))
        char_min_df = min(n_samples, tfidf_cfg.get("char_min_df", 5))

        self.tfidf_word.set_params(min_df=word_min_df)
        self.tfidf_char.set_params(min_df=char_min_df)

        # 1. Fit Numeric Imputer & Scaler
        num_vals = df_train[self.NUMERIC_COLS].values
        imputed_num = self.imputer.fit_transform(num_vals)
        self.scaler.fit(imputed_num)

        # 2. Fit Categorical OneHotEncoder
        cat_vals = df_train[self.CATEGORICAL_COLS].astype(str).values
        self.ohe.fit(cat_vals)
        cat_feature_names = list(self.ohe.get_feature_names_out(self.CATEGORICAL_COLS))

        # Build Feature Set A Names
        self.feature_names_a_ = list(self.NUMERIC_COLS) + cat_feature_names

        # 3. Fit TF-IDF Vectorizers
        texts = df_train["combined_text"].fillna("").astype(str).values
        self.tfidf_word.fit(texts)
        self.tfidf_char.fit(texts)

        word_names = [f"tfidf_word_{w}" for w in self.tfidf_word.get_feature_names_out()]
        char_names = [f"tfidf_char_{c}" for c in self.tfidf_char.get_feature_names_out()]

        # Build Feature Set B Names
        self.feature_names_b_ = list(self.feature_names_a_) + word_names + char_names

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, feature_set: str = "FEATURE_SET_B") -> pd.DataFrame:
        """Transform DataFrame using fitted transformers."""
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted on TRAIN before transform can be called.")

        # Numeric transform
        num_vals = df[self.NUMERIC_COLS].values
        imputed_num = self.imputer.transform(num_vals)
        scaled_num = self.scaler.transform(imputed_num)

        # Categorical transform
        cat_vals = df[self.CATEGORICAL_COLS].astype(str).values
        ohe_cat = self.ohe.transform(cat_vals)

        # Concatenate tabular features (Feature Set A)
        X_a = np.hstack([scaled_num, ohe_cat])

        if feature_set == "FEATURE_SET_A":
            return pd.DataFrame(X_a, columns=self.feature_names_a_)

        # Feature Set B: Add TF-IDF
        texts = df["combined_text"].fillna("").astype(str).values
        tfidf_w = self.tfidf_word.transform(texts).toarray()
        tfidf_c = self.tfidf_char.transform(texts).toarray()

        X_b = np.hstack([X_a, tfidf_w, tfidf_c])
        return pd.DataFrame(X_b, columns=self.feature_names_b_)

    def get_feature_names(self, feature_set: str = "FEATURE_SET_B") -> List[str]:
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted.")
        return self.feature_names_b_ if feature_set == "FEATURE_SET_B" else self.feature_names_a_

    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        print(f"Saved fitted preprocessing pipeline to {output_path}")

    @staticmethod
    def load(model_path: str) -> 'NitiMarkFeaturePipeline':
        return joblib.load(model_path)
