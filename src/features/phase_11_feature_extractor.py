"""
Phase 11 Multimodal Feature Extractor.
Constructs leakage-safe feature representations:
- Feature Set A: Structured Catalog Metadata
- Feature Set B: Metadata + Text TF-IDF
- Feature Set C: Metadata + Text + OCR Signals & Keyword Markers
- Feature Set D: Metadata + Text + OCR + YOLO Visual Region Signals

Strict Anti-Leakage Rules:
- Raw D01-D07 target labels are NEVER used as input features.
- All transformations are fitted on training data only.
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

WORKSPACE_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
from src.features.yolo_feature_extractor import YOLOFeatureExtractor


class Phase11FeatureExtractor:
    """
    Multimodal feature extraction pipeline for Phase 11 product compliance models.
    """
    
    def __init__(self, max_word_features: int = 300, max_char_features: int = 200):
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        
        self.yolo_extractor = YOLOFeatureExtractor()
        self.word_vectorizer = TfidfVectorizer(
            max_features=self.max_word_features,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        self.char_vectorizer = TfidfVectorizer(
            max_features=self.max_char_features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True
        )
        self.scaler = StandardScaler()
        self.category_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        
        # Categorical and structural feature names
        self.categorical_columns = ["category"]
        self.known_categories = []
        
        # Keyword markers for OCR/Vision
        self.declaration_keywords = {
            "kw_mfg_date": r"\b(mfd|mfg|manufactur|packed|pkd|date|exp|batch)\b",
            "kw_mrp_price": r"\b(mrp|rs\.?|inr|₹|incl|taxes|retail\s*price)\b",
            "kw_net_quantity": r"\b(net\s*wt|net\s*vol|quantity|qty|ml|g|kg|l|gms|litres?|pcs|units?)\b",
            "kw_consumer_care": r"\b(consumer|care|helpline|toll[-\s]*free|feedback|complaint|email|contact|customercare)\b",
            "kw_origin": r"\b(country\s*of\s*origin|made\s*in|origin|india|imported)\b",
            "kw_manufacturer": r"\b(mfd\s*by|manufactured\s*by|marketed\s*by|packer|imported\s*by|ltd|pvt)\b",
            "kw_commodity_name": r"\b(shampoo|oil|cream|soap|cleaner|wash|powder|bottle|pack|organic|herbal)\b"
        }

    def _extract_structural_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """Extract numeric and boolean structural features from catalog records."""
        features = []
        for _, row in df.iterrows():
            retail_price = float(row.get("retail_price", 0.0) or 0.0)
            discounted_price = float(row.get("discounted_price", 0.0) or 0.0)
            discount_ratio = (retail_price - discounted_price) / max(retail_price, 1.0)
            
            title_len = len(str(row.get("product_name", "") or ""))
            desc_len = len(str(row.get("description", "") or ""))
            spec_len = len(str(row.get("product_specifications", "") or ""))
            
            has_brand = 1.0 if str(row.get("brand", "") or "").strip() else 0.0
            has_image = 1.0 if str(row.get("image_urls", "") or row.get("image", "") or "").strip() else 0.0
            
            features.append([
                retail_price,
                discounted_price,
                discount_ratio,
                float(title_len),
                float(desc_len),
                float(spec_len),
                has_brand,
                has_image
            ])
        return np.array(features, dtype=np.float32)

    def _extract_ocr_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Extract OCR/Vision textual signals, token densities, and keyword matches."""
        features = []
        for _, row in df.iterrows():
            # Combine available text sources (OCR text or fallback product listing text)
            ocr_text = str(row.get("ocr_text", "") or row.get("observable_packaging_text", "") or "").lower()
            if not ocr_text:
                # Fallback to description/specifications if purely metadata
                ocr_text = f"{row.get('product_name', '')} {row.get('description', '')} {row.get('product_specifications', '')}".lower()
                
            ocr_char_count = float(len(ocr_text))
            ocr_word_count = float(len(ocr_text.split()))
            ocr_confidence = float(row.get("ocr_quality", row.get("ocr_confidence", 0.85 if ocr_text else 0.0)))
            
            kw_matches = []
            for kw_name, pattern in self.declaration_keywords.items():
                match_count = len(re.findall(pattern, ocr_text))
                kw_matches.append(min(float(match_count), 5.0))
                
            features.append([ocr_char_count, ocr_word_count, ocr_confidence] + kw_matches)
        return np.array(features, dtype=np.float32)

    def _extract_yolo_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Extract structured visual region signals from YOLO detections using YOLOFeatureExtractor."""
        return self.yolo_extractor.extract_from_df(df)

    def fit(self, df_train: pd.DataFrame):
        """Fit all feature transformers on training partition only."""
        print("Fitting Phase11FeatureExtractor on training data...")
        
        # 1. Text corpora for TF-IDF
        p_name = df_train["product_name"].fillna("").astype(str) if "product_name" in df_train.columns else pd.Series([""]*len(df_train))
        p_desc = df_train["description"].fillna("").astype(str) if "description" in df_train.columns else pd.Series([""]*len(df_train))
        p_spec = df_train["product_specifications"].fillna("").astype(str) if "product_specifications" in df_train.columns else pd.Series([""]*len(df_train))
        
        text_corpus = (p_name + " " + p_desc + " " + p_spec).tolist()
        
        self.word_vectorizer.fit(text_corpus)
        self.char_vectorizer.fit(text_corpus)
        
        # 2. Structural metadata scaler
        struct_feats = self._extract_structural_metadata(df_train)
        self.scaler.fit(struct_feats)
        
        # 3. Categorical one-hot
        if "category" in df_train.columns:
            cats = df_train[["category"]].fillna("Other").astype(str)
        else:
            cats = pd.DataFrame({"category": ["Other"] * len(df_train)})
        self.category_encoder.fit(cats)
        
        self.is_fitted = True
        print(f"Phase11FeatureExtractor fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame, feature_set: str = "SET_D") -> np.ndarray:
        """
        Transform dataset into specified feature representation.
        Options: 'SET_A', 'SET_B', 'SET_C', 'SET_D'
        """
        if not self.is_fitted:
            raise RuntimeError("Feature extractor must be fitted before transform.")
            
        # 1. Structural features (scaled)
        struct_feats = self.scaler.transform(self._extract_structural_metadata(df))
        
        # 2. Categorical features (one-hot)
        if "category" in df.columns:
            cats = df[["category"]].fillna("Other").astype(str)
        else:
            cats = pd.DataFrame({"category": ["Other"] * len(df)})
        cat_feats = self.category_encoder.transform(cats)
        
        # Base Feature Set A
        feats_set_a = np.hstack([struct_feats, cat_feats])
        if feature_set.upper() == "SET_A":
            return feats_set_a
            
        # 3. Text TF-IDF features
        p_name = df["product_name"].fillna("").astype(str) if "product_name" in df.columns else pd.Series([""]*len(df))
        p_desc = df["description"].fillna("").astype(str) if "description" in df.columns else pd.Series([""]*len(df))
        p_spec = df["product_specifications"].fillna("").astype(str) if "product_specifications" in df.columns else pd.Series([""]*len(df))
        
        text_corpus = (p_name + " " + p_desc + " " + p_spec).tolist()
        word_tfidf = self.word_vectorizer.transform(text_corpus).toarray()
        char_tfidf = self.char_vectorizer.transform(text_corpus).toarray()
        
        # Base Feature Set B
        feats_set_b = np.hstack([feats_set_a, word_tfidf, char_tfidf])
        if feature_set.upper() == "SET_B":
            return feats_set_b
            
        # 4. OCR / Vision signals
        ocr_feats = self._extract_ocr_signals(df)
        
        # Feature Set C
        feats_set_c = np.hstack([feats_set_b, ocr_feats])
        if feature_set.upper() == "SET_C":
            return feats_set_c
            
        # 5. YOLO visual region signals
        yolo_feats = self._extract_yolo_signals(df)
        
        # Feature Set D (Multimodal)
        feats_set_d = np.hstack([feats_set_c, yolo_feats])
        return feats_set_d

    def save(self, output_path: Union[str, Path]):
        """Persist fitted feature extractor."""
        joblib.dump(self, str(output_path))
        print(f"Saved Phase11FeatureExtractor to: {output_path}")

    @classmethod
    def load(cls, input_path: Union[str, Path]) -> "Phase11FeatureExtractor":
        """Load persisted feature extractor."""
        return joblib.load(str(input_path))
