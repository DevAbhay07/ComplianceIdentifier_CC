"""
product_compliance_feature_extractor.py
=======================================
Transforms unified ProductComplianceTrainingRecord objects, ProductEvidenceRecord
objects, or DataFrames into controlled, leakage-free feature sets (Sets A through E).

Controlled Feature Sets:
- SET_A (16 features): Statutory Declaration Presence & Confidence flags
- SET_B (40 features): Set A + Actual Numerical & Categorical Values (MRP, quantity, dates, units, entity names)
- SET_C (46 features): Set B + OCR Quality & Evidence Completeness Diagnostics
- SET_D (56 features): Set C + Physical Packaging / YOLO-derived Evidence Features
- SET_E (61 features): Set D + Multi-Image & Cross-View Consistency Features
"""

import re
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.extraction.declaration_schema import ProductEvidenceRecord, DeclarationStatus


class ProductComplianceFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Standardized Feature Extractor for Product Compliance ML.
    Operates identically on Flipkart catalog records, YOLO physical packaging records,
    and real-time inference images.
    """

    FEATURE_SETS = ["SET_A", "SET_B", "SET_C", "SET_D", "SET_E"]

    def __init__(self, default_feature_set: str = "SET_E"):
        self.default_feature_set = default_feature_set
        self.feature_names_by_set: Dict[str, List[str]] = {}
        self._init_feature_names()

    def _init_feature_names(self):
        # Set A: Presence & Confidence (16)
        set_a = []
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            set_a.append(f"{did.lower()}_present")
            set_a.append(f"{did.lower()}_confidence")
        set_a.extend([
            "mandatory_declarations_present_count",
            "all_mandatory_present"
        ])
        self.feature_names_by_set["SET_A"] = set_a

        # Set B: Set A + Actual Values (16 + 24 = 40)
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

        # Set C: Set B + OCR Quality & Diagnostics (40 + 6 = 46)
        set_c = list(set_b) + [
            "overall_ocr_confidence",
            "ocr_text_length_log",
            "declaration_completeness_ratio",
            "number_of_ambiguous_declarations",
            "number_of_cannot_determine",
            "number_of_conflicting_fields"
        ]
        self.feature_names_by_set["SET_C"] = set_c

        # Set D: Set C + Physical Packaging / YOLO-derived Evidence (46 + 10 = 56)
        set_d = list(set_c) + [
            "yolo_count_mrp_panel",
            "yolo_count_mfg_date",
            "yolo_count_net_quantity",
            "yolo_count_consumer_care",
            "yolo_count_manufacturer_info",
            "yolo_count_country_of_origin",
            "yolo_total_detections",
            "yolo_unique_classes",
            "yolo_total_bbox_area",
            "is_physical_package_source"
        ]
        self.feature_names_by_set["SET_D"] = set_d

        # Set E: Set D + Multi-Image & Fusion Features (56 + 5 = 61)
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
        records_or_df: Union[List[ProductComplianceTrainingRecord], List[ProductEvidenceRecord], pd.DataFrame, List[Dict[str, Any]]],
        feature_set: Optional[str] = None
    ) -> np.ndarray:
        """
        Transforms records or dataframe into a 2D numpy feature array.
        """
        selected_set = feature_set or self.default_feature_set
        if selected_set not in self.FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {selected_set}. Choose from {self.FEATURE_SETS}")

        feature_names = self.feature_names_by_set[selected_set]
        rows = []

        if isinstance(records_or_df, pd.DataFrame):
            for _, row in records_or_df.iterrows():
                rows.append(self._extract_from_row(row, selected_set))
        elif isinstance(records_or_df, list):
            for item in records_or_df:
                if isinstance(item, ProductComplianceTrainingRecord):
                    rows.append(self._extract_from_training_record(item, selected_set))
                elif isinstance(item, ProductEvidenceRecord):
                    rows.append(self._extract_from_evidence_record(item, selected_set))
                elif isinstance(item, dict):
                    rows.append(self._extract_from_row(item, selected_set))
        else:
            raise TypeError(f"Unsupported input type: {type(records_or_df)}")

        df_feats = pd.DataFrame(rows, columns=feature_names).fillna(0.0)
        return df_feats.values.astype(np.float32)

    def get_feature_names(self, feature_set: Optional[str] = None) -> List[str]:
        selected_set = feature_set or self.default_feature_set
        return list(self.feature_names_by_set[selected_set])

    def _extract_from_training_record(self, rec: ProductComplianceTrainingRecord, fset: str) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        decls = rec.declarations or {}

        # 1. Presence & Confidence
        # IMPORTANT: ABSENT ≠ CANNOT_DETERMINE
        # PRESENT → is_p=1.0, confidence from extraction
        # ABSENT → is_p=0.0, confidence from extraction (evidence of absence)
        # CANNOT_DETERMINE → is_p=0.0, confidence=0.0 (no evidence)
        pres_count = 0
        for did in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            d_info = decls.get(did, {})
            status = d_info.get("status", "CANNOT_DETERMINE")
            conf = float(d_info.get("extraction_confidence", d_info.get("confidence", 0.0)))
            is_p = 1.0 if status in [DeclarationStatus.PRESENT.value, "PRESENT"] else 0.0
            feats[f"{did.lower()}_present"] = is_p
            # Distinguish ABSENT (has evidence, confidence > 0) from CANNOT_DETERMINE (no evidence)
            if is_p > 0:
                feats[f"{did.lower()}_confidence"] = conf
            elif status in [DeclarationStatus.ABSENT.value, "ABSENT"]:
                # Negative evidence: preserve confidence to distinguish from unknown
                feats[f"{did.lower()}_confidence"] = conf * 0.5  # Scale to differentiate
            else:
                # CANNOT_DETERMINE / UNKNOWN: no evidence
                feats[f"{did.lower()}_confidence"] = 0.0
            if is_p > 0:
                pres_count += 1

        feats["mandatory_declarations_present_count"] = float(pres_count)
        feats["all_mandatory_present"] = 1.0 if pres_count >= 6 else 0.0

        if fset == "SET_A":
            return feats

        # 2. Actual Values (Set B)
        # D01
        v1 = rec.d01_values or {}
        feats["d01_name_length"] = float(len(str(v1.get("entity_name", ""))))
        feats["d01_address_present"] = 1.0 if v1.get("address_markers") else 0.0
        feats["d01_has_pincode"] = 1.0 if v1.get("pincode") else 0.0
        feats["d01_is_packer"] = 1.0 if v1.get("is_packer") else 0.0
        feats["d01_is_importer"] = 1.0 if v1.get("is_importer") else 0.0
        feats["d01_is_marketer"] = 1.0 if v1.get("is_marketer") else 0.0

        def _safe_float(val: Any, default: float = 0.0) -> float:
            if val is None:
                return default
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                val_clean = val.replace(",", "")
                m = re.search(r"[-+]?\d*\.?\d+", val_clean)
                if m:
                    try:
                        return float(m.group(0))
                    except Exception:
                        return default
            return default

        # D02
        v2 = rec.d02_values or {}
        feats["d02_name_length"] = float(len(str(v2.get("common_name", ""))))
        feats["d02_identity_confidence"] = _safe_float(v2.get("identity_confidence"), 0.8)

        # D03
        v3 = rec.d03_values or {}
        qty = _safe_float(v3.get("quantity") or v3.get("net_quantity"))
        feats["d03_quantity_numeric_log"] = float(math.log1p(qty)) if qty > 0 else 0.0
        u = str(v3.get("unit", "")).lower()
        feats["d03_unit_is_weight"] = 1.0 if u in ["g", "gm", "kg", "mg", "oz"] else 0.0
        feats["d03_unit_is_volume"] = 1.0 if u in ["ml", "l", "litre", "liter"] else 0.0
        feats["d03_unit_is_count"] = 1.0 if u in ["pcs", "pc", "units", "capsules", "tablets", "count"] else 0.0

        # D04
        v4 = rec.d04_values or {}
        feats["d04_manufacture_month"] = _safe_float(v4.get("month"))
        feats["d04_manufacture_year"] = _safe_float(v4.get("year"))
        feats["d04_format_valid"] = 1.0 if _safe_float(v4.get("year")) > 0 else 0.0

        # D05
        v5 = rec.d05_values or {}
        mrp = _safe_float(v5.get("mrp") or v5.get("price"))
        feats["d05_mrp_numeric_log"] = float(math.log1p(mrp)) if mrp > 0 else 0.0
        feats["d05_tax_inclusive_present"] = 1.0 if v5.get("tax_inclusive") else 0.0
        feats["d05_has_price"] = 1.0 if mrp > 0 else 0.0

        # D06
        v6 = rec.d06_values or {}
        feats["d06_phone_present"] = 1.0 if v6.get("phone") or v6.get("has_phone") else 0.0
        feats["d06_email_present"] = 1.0 if v6.get("email") or v6.get("has_email") else 0.0
        feats["d06_has_toll_free"] = 1.0 if ("1800" in str(v6.get("phone", "")) or "1800" in str(v6.get("consumer_care", ""))) else 0.0
        feats["d06_address_present"] = 1.0 if v6.get("address") else 0.0

        # D07
        v7 = rec.d07_values or {}
        country_str = str(v7.get("country") or v7.get("country_of_origin", "")).lower()
        feats["d07_country_origin_present"] = 1.0 if country_str else 0.0
        feats["d07_is_domestic"] = 1.0 if ("india" in country_str) else 0.0

        if fset == "SET_B":
            return feats

        # 3. OCR Quality & Completeness (Set C)
        feats["overall_ocr_confidence"] = float(rec.ocr_quality)
        feats["ocr_text_length_log"] = float(math.log1p(rec.text_length))
        feats["declaration_completeness_ratio"] = float(pres_count / 7.0)
        feats["number_of_ambiguous_declarations"] = float(sum(1 for d in decls.values() if d.get("status") == "PARTIAL"))
        feats["number_of_cannot_determine"] = float(sum(1 for d in decls.values() if d.get("status") == "CANNOT_DETERMINE"))
        feats["number_of_conflicting_fields"] = 0.0

        if fset == "SET_C":
            return feats

        # 4. Physical Packaging / YOLO Evidence (Set D)
        pkg = rec.packaging_features or {}
        c_counts = pkg.get("yolo_class_counts", {})
        feats["yolo_count_mrp_panel"] = float(c_counts.get("mrp details", 0) + c_counts.get("tax", 0))
        feats["yolo_count_mfg_date"] = float(c_counts.get("details", 0))
        feats["yolo_count_net_quantity"] = float(c_counts.get("weight", 0))
        feats["yolo_count_consumer_care"] = float(c_counts.get("consumer", 0))
        feats["yolo_count_manufacturer_info"] = float(c_counts.get("address", 0) + c_counts.get("brand", 0))
        feats["yolo_count_country_of_origin"] = float(c_counts.get("origin", 0))
        feats["yolo_total_detections"] = float(pkg.get("yolo_region_count", 0))
        feats["yolo_unique_classes"] = float(len(pkg.get("yolo_detected_classes", [])))
        feats["yolo_total_bbox_area"] = float(pkg.get("yolo_total_bbox_area", 0.0))
        feats["is_physical_package_source"] = 1.0 if rec.source_dataset == "yolo_physical_packaging" else 0.0

        if fset == "SET_D":
            return feats

        # 5. Multi-Image & Fusion (Set E)
        feats["image_count"] = float(rec.image_count)
        feats["images_with_evidence"] = float(rec.image_count)
        feats["cross_image_consistency"] = 1.0
        feats["duplicate_evidence_count"] = 0.0
        feats["has_conflicts"] = 0.0

        return feats

    def _extract_from_evidence_record(self, rec: ProductEvidenceRecord, fset: str) -> Dict[str, float]:
        # Helper to convert ProductEvidenceRecord into training record format
        d_dict = {did: d.to_dict() for did, d in rec.declarations.items()}
        t_rec = ProductComplianceTrainingRecord(
            product_id=rec.product_id,
            source_dataset="inference",
            target_provenance="UNKNOWN",
            overall_target="UNKNOWN",
            declarations=d_dict,
            d01_values=rec.declarations["D01"].normalized_value if "D01" in rec.declarations else {},
            d02_values=rec.declarations["D02"].normalized_value if "D02" in rec.declarations else {},
            d03_values=rec.declarations["D03"].normalized_value if "D03" in rec.declarations else {},
            d04_values=rec.declarations["D04"].normalized_value if "D04" in rec.declarations else {},
            d05_values=rec.declarations["D05"].normalized_value if "D05" in rec.declarations else {},
            d06_values=rec.declarations["D06"].normalized_value if "D06" in rec.declarations else {},
            d07_values=rec.declarations["D07"].normalized_value if "D07" in rec.declarations else {},
            packaging_features={"yolo_region_count": len(rec.visual_detections)},
            ocr_quality=rec.overall_ocr_confidence,
            text_length=len(" ".join(rec.raw_ocr_texts.values())),
            image_count=rec.image_count
        )
        return self._extract_from_training_record(t_rec, fset)

    def _extract_from_row(self, row: Union[pd.Series, Dict[str, Any]], fset: str) -> Dict[str, float]:
        if "declarations" in row and isinstance(row["declarations"], dict):
            # Parse dict into training record
            rec = ProductComplianceTrainingRecord(
                product_id=str(row.get("product_id", "row")),
                source_dataset=str(row.get("source_dataset", "flipkart")),
                target_provenance=str(row.get("target_provenance", "UNKNOWN")),
                overall_target=str(row.get("overall_target", "UNKNOWN")),
                category=str(row.get("category", "Other")),
                brand=str(row.get("brand", "")),
                product_title=str(row.get("product_title", "")),
                raw_text=str(row.get("raw_text", "")),
                declarations=row.get("declarations", {}),
                d01_values=row.get("d01_values", {}),
                d02_values=row.get("d02_values", {}),
                d03_values=row.get("d03_values", {}),
                d04_values=row.get("d04_values", {}),
                d05_values=row.get("d05_values", {}),
                d06_values=row.get("d06_values", {}),
                d07_values=row.get("d07_values", {}),
                packaging_features=row.get("packaging_features", {}),
                ocr_quality=float(row.get("ocr_quality", 0.90)),
                text_length=int(row.get("text_length", 0)),
                image_count=int(row.get("image_count", 1))
            )
            return self._extract_from_training_record(rec, fset)
        else:
            # Fallback for raw text row
            from src.extraction.declaration_extractor import DeclarationExtractor
            ext = DeclarationExtractor()
            text = str(row.get("description", "") or row.get("raw_text", "") or row.get("product_name", ""))
            decls = ext.extract_from_text_and_vision(raw_ocr_text=text, ocr_confidence=0.85)
            d_dict = {did: d.to_dict() for did, d in decls.items()}
            rec = ProductComplianceTrainingRecord(
                product_id=str(row.get("product_id", "row")),
                source_dataset=str(row.get("source_dataset", "flipkart")),
                target_provenance="WEAKLY_DERIVED_SILVER",
                overall_target=str(row.get("overall_target", "UNKNOWN")),
                declarations=d_dict,
                d01_values=decls["D01"].normalized_value,
                d02_values=decls["D02"].normalized_value,
                d03_values=decls["D03"].normalized_value,
                d04_values=decls["D04"].normalized_value,
                d05_values=decls["D05"].normalized_value,
                d06_values=decls["D06"].normalized_value,
                d07_values=decls["D07"].normalized_value,
                ocr_quality=0.85,
                text_length=len(text)
            )
            return self._extract_from_training_record(rec, fset)
