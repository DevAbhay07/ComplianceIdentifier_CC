"""
declaration_schema.py
=====================
Standardized structured evidence schema for Legal Metrology statutory declarations (D01-D07).
Preserves raw values, normalized entities, bounding boxes, confidence scores, and provenance.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


class DeclarationStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    PARTIAL = "PARTIAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class EvidenceSource(str, Enum):
    OCR = "OCR"
    VISUAL_YOLO = "VISUAL_YOLO"
    COMBINED = "COMBINED"
    METADATA = "METADATA"
    MULTI_IMAGE_FUSION = "MULTI_IMAGE_FUSION"


DECLARATION_DEFINITIONS = {
    "D01": {
        "name": "Manufacturer / Packer / Importer Name & Address",
        "statutory_rule": "Rule 6(1)(a)",
        "mandatory": True,
        "description": "Name and complete physical address of manufacturer, packer, or importer."
    },
    "D02": {
        "name": "Generic or Common Name of Commodity",
        "statutory_rule": "Rule 6(1)(b)",
        "mandatory": True,
        "description": "Generic name or common nomenclature of the pre-packaged good."
    },
    "D03": {
        "name": "Net Quantity in Standard Metric Units",
        "statutory_rule": "Rule 6(1)(c)",
        "mandatory": True,
        "description": "Net quantity in standard metric units of weight, measure, or numerical count."
    },
    "D04": {
        "name": "Month & Year of Manufacture / Packing / Expiry",
        "statutory_rule": "Rule 6(1)(d)",
        "mandatory": True,
        "description": "Month and year in which commodity was manufactured, packed, or pre-packed."
    },
    "D05": {
        "name": "Maximum Retail Price (MRP incl. of all taxes)",
        "statutory_rule": "Rule 6(1)(da)",
        "mandatory": True,
        "description": "Retail price inclusive of all taxes clearly indicating Maximum Retail Price / MRP."
    },
    "D06": {
        "name": "Consumer Care / Grievance Redressal Contacts",
        "statutory_rule": "Rule 6(1)(n)",
        "mandatory": True,
        "description": "Name, telephone helpline, email, and address of the consumer grievance redressal cell."
    },
    "D07": {
        "name": "Country of Origin",
        "statutory_rule": "Rule 6(1)(ea)",
        "mandatory": False,  # Mandatory for imported goods / e-commerce declarations
        "description": "Country of origin or manufacture of imported commodities."
    }
}


@dataclass
class DeclarationEvidence:
    """
    Standardized structured evidence container for a single declaration (D01-D07).
    """
    declaration_id: str                          # D01 - D07
    declaration_name: str                        # Human readable title
    status: str                                  # DeclarationStatus value
    raw_value: str = ""                          # Raw matched substring
    normalized_value: Dict[str, Any] = field(default_factory=dict)  # Parsed entity dictionary
    source: str = EvidenceSource.OCR.value       # OCR / VISUAL_YOLO / COMBINED
    source_image: str = "default_image"          # Identifier of image providing evidence
    source_region: Optional[str] = None          # ROI identifier e.g. "roi_mrp_01"
    bounding_box: Optional[List[float]] = None   # [x1, y1, x2, y2]
    ocr_confidence: float = 1.0                  # Quality/confidence of OCR (0.0 - 1.0)
    extraction_confidence: float = 1.0           # Parser regex/heuristic confidence (0.0 - 1.0)
    evidence_text: str = ""                      # Human-readable evidence snippet / explanation
    conflicting_values: List[Dict[str, Any]] = field(default_factory=list)  # Preserved conflicting evidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeclarationEvidence":
        return cls(**data)


@dataclass
class ProductEvidenceRecord:
    """
    Aggregated product-level structured evidence record across all declarations D01-D07.
    """
    product_id: str
    image_count: int
    source_images: List[str]
    declarations: Dict[str, DeclarationEvidence]
    overall_ocr_confidence: float = 1.0
    image_quality: float = 1.0
    raw_ocr_texts: Dict[str, str] = field(default_factory=dict)
    visual_detections: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def declaration_completeness_ratio(self) -> float:
        if not self.declarations:
            return 0.0
        present = sum(1 for d in self.declarations.values() if d.status in [DeclarationStatus.PRESENT.value, "PRESENT"])
        return float(present / len(self.declarations))

    @property
    def conflicting_fields(self) -> List[str]:
        conflicts = []
        for did, d in self.declarations.items():
            if d.status in [DeclarationStatus.CONFLICTING_EVIDENCE.value, "CONFLICTING_EVIDENCE"] or len(d.conflicting_values) > 0:
                conflicts.append(did)
        return conflicts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "image_count": self.image_count,
            "source_images": self.source_images,
            "declarations": {k: v.to_dict() for k, v in self.declarations.items()},
            "overall_ocr_confidence": self.overall_ocr_confidence,
            "image_quality": self.image_quality,
            "declaration_completeness_ratio": self.declaration_completeness_ratio,
            "conflicting_fields": self.conflicting_fields,
            "raw_ocr_texts": self.raw_ocr_texts,
            "visual_detections": self.visual_detections,
            "metadata": self.metadata
        }

