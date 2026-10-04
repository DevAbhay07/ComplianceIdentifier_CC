"""
multimodal_feature_adapter.py
=============================
Adapts single-image and multi-image packaging evidence records into unified
multimodal feature matrices for downstream declaration-centric product models.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from src.extraction.declaration_schema import ProductEvidenceRecord
from src.features.declaration_feature_extractor import DeclarationFeatureExtractor


class MultimodalFeatureAdapter:
    """
    Seamlessly adapts ProductEvidenceRecord objects into feature arrays.
    """

    def __init__(self, feature_set: str = "SET_E"):
        self.feature_set = feature_set
        self.feature_extractor = DeclarationFeatureExtractor(default_feature_set=feature_set)

    def fit(self, X, y=None):
        return self

    def transform_record(
        self,
        record: ProductEvidenceRecord,
        feature_set: Optional[str] = None
    ) -> np.ndarray:
        """
        Transforms a single ProductEvidenceRecord into a 1xD numpy array.
        """
        fset = feature_set or self.feature_set
        return self.feature_extractor.transform([record], feature_set=fset)

    def transform_records(
        self,
        records: List[ProductEvidenceRecord],
        feature_set: Optional[str] = None
    ) -> np.ndarray:
        """
        Transforms a list of ProductEvidenceRecords into an NxD numpy array.
        """
        fset = feature_set or self.feature_set
        return self.feature_extractor.transform(records, feature_set=fset)

    def get_feature_names(self, feature_set: Optional[str] = None) -> List[str]:
        fset = feature_set or self.feature_set
        return self.feature_extractor.get_feature_names(fset)
