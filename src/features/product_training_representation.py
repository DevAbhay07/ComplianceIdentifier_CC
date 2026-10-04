"""
product_training_representation.py
==================================
Standardized Common Product Compliance Training Representation for NitiMark-ML.

Unifies disparate data sources into a single schema:
1. Flipkart catalog metadata (source_dataset="flipkart")
2. YOLO physical packaging dataset (source_dataset="yolo_physical_packaging")

Preserves:
- Structured D01-D07 declarations with actual semantic values (MRP, quantity, dates, units, contacts)
- Physical packaging visual evidence (detected region counts, geometries, bounding box areas)
- Optical character recognition quality and diagnostics
- Strict dataset provenance and target derivation metadata
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


class TargetProvenance(str, Enum):
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    WEAKLY_DERIVED_SILVER = "WEAKLY_DERIVED_SILVER"
    YOLO_REGION_ANNOTATION = "YOLO_REGION_ANNOTATION"
    UNKNOWN = "UNKNOWN"


class DatasetSource(str, Enum):
    FLIPKART = "flipkart"
    YOLO_PHYSICAL_PACKAGING = "yolo_physical_packaging"
    HUMAN_REFERENCE_BENCHMARK = "human_reference_benchmark"
    REAL_PACKAGING_BENCHMARK = "real_packaging_benchmark"


@dataclass
class DeclarationValueRecord:
    """Stores normalized and raw values for a statutory declaration."""
    status: str = "CANNOT_DETERMINE"  # PRESENT, ABSENT, PARTIAL, NOT_APPLICABLE, CANNOT_DETERMINE
    raw_text: str = ""
    confidence: float = 1.0
    source: str = "OCR"  # OCR, PACKAGING_REGION, COMBINED
    normalized_values: Dict[str, Any] = field(default_factory=dict)
    bounding_box: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductComplianceTrainingRecord:
    """
    Standardized Common Product Compliance Training Record.
    Unified representation consumed by downstream feature extractors and ML models.
    """
    product_id: str
    source_dataset: str          # "flipkart" or "yolo_physical_packaging"
    target_provenance: str       # HUMAN_VERIFIED, WEAKLY_DERIVED_SILVER, YOLO_REGION_ANNOTATION, UNKNOWN
    overall_target: str          # COMPLIANT, NON_COMPLIANT, UNCERTAIN, UNKNOWN
    category: str = "Other"
    brand: str = ""
    product_title: str = ""
    raw_text: str = ""
    
    # Declarations D01-D07
    declarations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Semantic Normalized Values
    d01_values: Dict[str, Any] = field(default_factory=dict)  # entity_name, address_markers, pincode, is_packer
    d02_values: Dict[str, Any] = field(default_factory=dict)  # common_name, identity_confidence
    d03_values: Dict[str, Any] = field(default_factory=dict)  # quantity, unit, unit_category
    d04_values: Dict[str, Any] = field(default_factory=dict)  # month, year, expiry_month, expiry_year
    d05_values: Dict[str, Any] = field(default_factory=dict)  # mrp, currency, tax_inclusive
    d06_values: Dict[str, Any] = field(default_factory=dict)  # email, phone, helpline
    d07_values: Dict[str, Any] = field(default_factory=dict)  # country, is_domestic
    
    # Physical Packaging & YOLO Visual Evidence
    packaging_features: Dict[str, Any] = field(default_factory=dict)  # region counts, bboxes, areas, density
    
    # Optical Quality Diagnostics
    ocr_quality: float = 1.0
    text_length: int = 0
    image_count: int = 1
    image_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductComplianceTrainingRecord":
        return cls(**data)
