"""
declaration_feature_extractor.py
================================
Transforms structured D01-D07 ProductEvidenceRecord objects and text/visual evidence
into controlled, leakage-safe declaration feature vectors (Sets A, B, C, D, E).

CRITICAL ARCHITECTURAL RULES:
1. Features are extracted from verified declaration semantic evidence, actual values, and OCR quality.
2. ZERO target leakage: Never derive features from the product target or rule-derived ground truth.
3. Controlled feature sets (A-E) allow rigorous model ablation.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin

from src.extraction.declaration_schema import (
    DeclarationEvidence,
    DeclarationStatus,
    ProductEvidenceRecord,
    DECLARATION_DEFINITIONS
)
from src.extraction.declaration_extractor import DeclarationExtractor
from src.extraction.evidence_fusion import EvidenceFusionEngine


class DeclarationFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts structured declaration-centric ML features across controlled ablation sets:
    - SET_A: Declaration Presence & Confidence (16 features)
    - SET_B: Set A + Actual Numerical & Categorical Values (38 features)
    - SET_C: Set B + OCR Quality & Ambiguity Diagnostics (44 features)
    - SET_D: Set C + Supplementary YOLO Visual Support (54 features)
    - SET_E: Set D + Multi-Image & Cross-View Fusion (59 features)
    """

    FEATURE_SETS = ["SET_A", "SET_B", "SET_C", "SET_D", "SET_E"]

    def __init__(self, default_feature_set: str = "SET_E"):
        self.default_feature_set = default_feature_set
        self.extractor = DeclarationExtractor()
        self.fusion_engine = EvidenceFusionEngine()
        self.feature_names_by_set: Dict[str, List[str]] = {}
        self._init_feature_names()

    def _init_feature_names(self):
        # Set A: Presence (16)
        set_a = []
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            set_a.append(f"{did.lower()}_present")
            set_a.append(f"{did.lower()}_confidence")
        set_a.extend([
            "mandatory_declarations_present_count",
            "all_mandatory_present"
        ])
        self.feature_names_by_set["SET_A"] = set_a

        # Set B: Actual Values (Set A + 22 = 38)
        set_b = list(set_a) + [
            # D01 Values
            "d01_name_length", "d01_address_present", "d01_has_pincode",
            "d01_is_packer", "d01_is_importer", "d01_is_marketer",
            # D02 Values
            "d02_name_length", "d02_identity_confidence",
            # D03 Values
            "d03_quantity_numeric_log", "d03_unit_is_weight", "d03_unit_is_volume", "d03_unit_is_count",
            # D04 Values
            "d04_manufacture_month", "d04_manufacture_year", "d04_format_valid",
            # D05 Values
            "d05_mrp_numeric_log", "d05_tax_inclusive_present", "d05_has_price",
            # D06 Values
            "d06_phone_present", "d06_email_present", "d06_has_toll_free", "d06_address_present",
            # D07 Values
            "d07_country_origin_present", "d07_is_domestic"
        ]
        self.feature_names_by_set["SET_B"] = set_b

        # Set C: OCR Quality & Completeness (Set B + 6 = 44)
        set_c = list(set_b) + [
            "overall_ocr_confidence",
            "ocr_text_length_log",
            "declaration_completeness_ratio",
            "number_of_ambiguous_declarations",
            "number_of_cannot_determine",
            "number_of_conflicting_fields"
        ]
        self.feature_names_by_set["SET_C"] = set_c

        # Set D: YOLO Visual Support (Set C + 10 = 54)
        set_d = list(set_c) + [
            "yolo_count_mrp_panel",
            "yolo_count_mfg_date",
            "yolo_count_net_quantity",
            "yolo_count_consumer_care",
            "yolo_count_manufacturer_info",
            "yolo_count_country_of_origin",
            "yolo_total_detections",
            "yolo_unique_classes",
            "yolo_mean_confidence",
            "yolo_max_confidence"
        ]
        self.feature_names_by_set["SET_D"] = set_d

        # Set E: Full Multimodal & Multi-Image (Set D + 5 = 59)
        set_e = list(set_d) + [
            "image_count",
            "images_with_evidence",
            "cross_image_consistency",
            "duplicate_evidence_count",
            "has_conflicts"
        ]
        self.feature_names_by_set["SET_E"] = set_e

    def fit(self, X, y=None):
        return self

    def transform(
        self,
        records_or_df: Union[List[ProductEvidenceRecord], pd.DataFrame, List[Dict[str, Any]]],
        feature_set: Optional[str] = None
    ) -> np.ndarray:
        """
        Transforms input records or DataFrame into a 2D numpy feature array.
        """
        selected_set = feature_set or self.default_feature_set
        if selected_set not in self.FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {selected_set}. Choose from {self.FEATURE_SETS}")

        feature_names = self.feature_names_by_set[selected_set]
        rows = []

        if isinstance(records_or_df, pd.DataFrame):
            for _, row in records_or_df.iterrows():
                rows.append(self._extract_features_from_dict_or_row(row, selected_set))
        elif isinstance(records_or_df, list):
            for item in records_or_df:
                if isinstance(item, ProductEvidenceRecord):
                    rows.append(self._extract_features_from_record(item, selected_set))
                elif isinstance(item, dict):
                    rows.append(self._extract_features_from_dict_or_row(item, selected_set))
        else:
            raise TypeError(f"Unsupported input type: {type(records_or_df)}")

        df_feats = pd.DataFrame(rows, columns=feature_names).fillna(0.0)
        return df_feats.values.astype(np.float32)

    def get_feature_names(self, feature_set: Optional[str] = None) -> List[str]:
        selected_set = feature_set or self.default_feature_set
        return self.feature_names_by_set.get(selected_set, self.feature_names_by_set["SET_E"])

    def _extract_features_from_record(
        self,
        record: ProductEvidenceRecord,
        feature_set: str
    ) -> Dict[str, float]:
        """
        Extracts feature values directly from a ProductEvidenceRecord.
        """
        feats: Dict[str, float] = {}
        decls = record.declarations

        # Set A: Presence & Confidence
        mand_present = 0
        total_present = 0
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            ev = decls.get(did)
            is_pres = 1.0 if (ev and ev.status == DeclarationStatus.PRESENT.value) else 0.0
            conf = float(ev.extraction_confidence) if ev else 0.0
            
            feats[f"{did.lower()}_present"] = is_pres
            feats[f"{did.lower()}_confidence"] = conf
            
            if is_pres > 0.5:
                total_present += 1
                if did != "D07":  # D01-D06 are strictly mandatory
                    mand_present += 1

        feats["mandatory_declarations_present_count"] = float(mand_present)
        feats["all_mandatory_present"] = 1.0 if mand_present >= 6 else 0.0

        if feature_set == "SET_A":
            return feats

        # Set B: Actual Values
        # D01
        d01 = decls.get("D01")
        norm_d01 = d01.normalized_value if (d01 and d01.status == DeclarationStatus.PRESENT.value) else {}
        feats["d01_name_length"] = float(len(norm_d01.get("entity_name", "")))
        feats["d01_address_present"] = 1.0 if norm_d01.get("address_markers") else 0.0
        feats["d01_has_pincode"] = 1.0 if norm_d01.get("pincode") else 0.0
        feats["d01_is_packer"] = 1.0 if norm_d01.get("is_packer") else 0.0
        feats["d01_is_importer"] = 1.0 if norm_d01.get("is_importer") else 0.0
        feats["d01_is_marketer"] = 1.0 if norm_d01.get("is_marketer") else 0.0

        # D02
        d02 = decls.get("D02")
        norm_d02 = d02.normalized_value if (d02 and d02.status == DeclarationStatus.PRESENT.value) else {}
        feats["d02_name_length"] = float(norm_d02.get("name_length", 0))
        feats["d02_identity_confidence"] = float(d02.extraction_confidence if d02 else 0.0)

        # D03
        d03 = decls.get("D03")
        norm_d03 = d03.normalized_value if (d03 and d03.status == DeclarationStatus.PRESENT.value) else {}
        qty_num = float(norm_d03.get("quantity", 0.0))
        feats["d03_quantity_numeric_log"] = float(math.log1p(qty_num))
        feats["d03_unit_is_weight"] = 1.0 if norm_d03.get("unit_category") == "WEIGHT" else 0.0
        feats["d03_unit_is_volume"] = 1.0 if norm_d03.get("unit_category") == "VOLUME" else 0.0
        feats["d03_unit_is_count"] = 1.0 if norm_d03.get("unit_category") == "COUNT" else 0.0

        # D04
        d04 = decls.get("D04")
        norm_d04 = d04.normalized_value if (d04 and d04.status == DeclarationStatus.PRESENT.value) else {}
        feats["d04_manufacture_month"] = float(norm_d04.get("month", 0))
        y_val = norm_d04.get("year", 0)
        feats["d04_manufacture_year"] = float(y_val - 2020) if y_val >= 2020 else 0.0
        feats["d04_format_valid"] = 1.0 if norm_d04.get("format_valid") else 0.0

        # D05
        d05 = decls.get("D05")
        norm_d05 = d05.normalized_value if (d05 and d05.status == DeclarationStatus.PRESENT.value) else {}
        mrp_num = float(norm_d05.get("mrp", 0.0))
        feats["d05_mrp_numeric_log"] = float(math.log1p(mrp_num))
        feats["d05_tax_inclusive_present"] = 1.0 if norm_d05.get("tax_inclusive") else 0.0
        feats["d05_has_price"] = 1.0 if mrp_num > 0 else 0.0

        # D06
        d06 = decls.get("D06")
        norm_d06 = d06.normalized_value if (d06 and d06.status == DeclarationStatus.PRESENT.value) else {}
        feats["d06_phone_present"] = 1.0 if norm_d06.get("phone") else 0.0
        feats["d06_email_present"] = 1.0 if norm_d06.get("email") else 0.0
        feats["d06_has_toll_free"] = 1.0 if norm_d06.get("has_toll_free") else 0.0
        feats["d06_address_present"] = 1.0 if norm_d06.get("address") else 0.0

        # D07
        d07 = decls.get("D07")
        norm_d07 = d07.normalized_value if (d07 and d07.status == DeclarationStatus.PRESENT.value) else {}
        feats["d07_country_origin_present"] = 1.0 if norm_d07.get("country") else 0.0
        feats["d07_is_domestic"] = 1.0 if norm_d07.get("is_domestic") else 0.0

        if feature_set == "SET_B":
            return feats

        # Set C: OCR Quality & Completeness
        all_ocr_text = " ".join(record.raw_ocr_texts.values())
        ambig_count = sum(1 for ev in decls.values() if ev.status == DeclarationStatus.CANNOT_DETERMINE.value)
        conflict_count = sum(1 for ev in decls.values() if ev.status == DeclarationStatus.CONFLICTING_EVIDENCE.value)

        feats["overall_ocr_confidence"] = float(record.overall_ocr_confidence)
        feats["ocr_text_length_log"] = float(math.log1p(len(all_ocr_text)))
        feats["declaration_completeness_ratio"] = float(total_present / 7.0)
        feats["number_of_ambiguous_declarations"] = float(ambig_count)
        feats["number_of_cannot_determine"] = float(ambig_count)
        feats["number_of_conflicting_fields"] = float(conflict_count)

        if feature_set == "SET_C":
            return feats

        # Set D: YOLO Visual Regions
        all_detections = []
        for img_name, det_list in record.visual_detections.items():
            all_detections.extend(det_list)

        class_counts = {}
        confidences = []
        for det in all_detections:
            cname = det.get("class", det.get("class_name", "")).lower()
            class_counts[cname] = class_counts.get(cname, 0) + 1
            confidences.append(float(det.get("confidence", 0.85)))

        feats["yolo_count_mrp_panel"] = float(class_counts.get("mrp details", 0) + class_counts.get("tax", 0))
        feats["yolo_count_mfg_date"] = float(class_counts.get("details", 0))
        feats["yolo_count_net_quantity"] = float(class_counts.get("weight", 0))
        feats["yolo_count_consumer_care"] = float(class_counts.get("consumer", 0))
        feats["yolo_count_manufacturer_info"] = float(class_counts.get("address", 0) + class_counts.get("brand", 0))
        feats["yolo_count_country_of_origin"] = float(class_counts.get("origin", 0))
        feats["yolo_total_detections"] = float(len(all_detections))
        feats["yolo_unique_classes"] = float(len(class_counts))
        feats["yolo_mean_confidence"] = float(np.mean(confidences)) if confidences else 0.0
        feats["yolo_max_confidence"] = float(max(confidences)) if confidences else 0.0

        if feature_set == "SET_D":
            return feats

        # Set E: Full Multimodal & Multi-Image
        feats["image_count"] = float(record.image_count)
        feats["images_with_evidence"] = float(len([k for k, v in record.raw_ocr_texts.items() if len(v.strip()) > 10]))
        feats["cross_image_consistency"] = 1.0 if conflict_count == 0 else 0.0
        feats["duplicate_evidence_count"] = float(sum(1 for ev in decls.values() if "," in ev.source_image))
        feats["has_conflicts"] = 1.0 if conflict_count > 0 else 0.0

        return feats

    def _extract_features_from_dict_or_row(
        self,
        item: Union[Dict[str, Any], pd.Series],
        feature_set: str
    ) -> Dict[str, float]:
        """
        Parses text and metadata row into a ProductEvidenceRecord on-the-fly and extracts features.
        """
        raw_ocr = str(item.get("ocr_text", item.get("description", item.get("observable_packaging_text", ""))) or "")
        title = str(item.get("title", item.get("product_name", "")) or "")
        description = str(item.get("description", "") or "")
        specs = item.get("specifications", item.get("product_specifications", {}))
        if isinstance(specs, str):
            specs = {}

        visual_regions = item.get("visual_regions", [])
        if not isinstance(visual_regions, list):
            visual_regions = []

        # Extract declarations
        decls = self.extractor.extract_from_text_and_vision(
            raw_ocr_text=raw_ocr,
            title=title,
            description=description,
            specifications=specs,
            visual_regions=visual_regions,
            image_name="img_01"
        )

        record = ProductEvidenceRecord(
            product_id=str(item.get("product_id", "sample")),
            image_count=1,
            source_images=["img_01"],
            declarations=decls,
            overall_ocr_confidence=0.90,
            image_quality=1.0,
            raw_ocr_texts={"img_01": raw_ocr},
            visual_detections={"img_01": visual_regions}
        )

        return self._extract_features_from_record(record, feature_set)
