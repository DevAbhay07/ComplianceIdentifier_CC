"""
schemas.py
==========
Pydantic v2 request and response schemas for the NitiMark Product Compliance ML API.

The API accepts human-readable product text (title, description, specifications)
and uses the existing NitiMark evidence parser + feature extractor + ML model
to predict compliance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Input payload for POST /predict.

    At least one of `title` or `description` must be supplied.
    `specifications` is optional structured metadata (key-value pairs from
    the e-commerce listing page).
    """

    title: str = Field(
        default="",
        description="Product title or name as displayed on the listing page.",
        examples=["Colgate Strong Teeth Toothpaste 200g"],
    )
    description: str = Field(
        default="",
        description="Full product description text.",
        examples=[
            "Colgate Strong Teeth Toothpaste 200g. "
            "Manufactured by Colgate-Palmolive (India) Ltd, Mumbai. "
            "MRP Rs. 109. Net Qty: 200 g. Made in India."
        ],
    )
    specifications: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional key-value specification pairs from the product page.",
        examples=[{"Net Quantity": "200 g", "Brand": "Colgate", "Category": "Oral Care"}],
    )
    brand: Optional[str] = Field(
        default=None,
        description="Product brand name (optional; inferred from title if absent).",
        examples=["Colgate"],
    )
    category: Optional[str] = Field(
        default=None,
        description="Product category (optional; defaults to 'General Commodity').",
        examples=["Oral Care"],
    )
    retail_price: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum Retail Price (MRP) in INR, if known.",
        examples=[109.0],
    )
    discounted_price: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Discounted / sale price in INR, if applicable.",
        examples=[95.0],
    )

    @model_validator(mode="after")
    def _require_text_input(self) -> "PredictRequest":
        if not (self.title.strip() or self.description.strip()):
            raise ValueError(
                "At least one of 'title' or 'description' must be a non-empty string."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Colgate Strong Teeth Toothpaste 200g",
                    "description": (
                        "Manufactured by Colgate-Palmolive (India) Ltd, "
                        "Sudarshan Chambers, 5th Floor, Bhulabhai Desai Road, Mumbai 400026. "
                        "MRP Rs. 109 (incl. of all taxes). Net Qty: 200 g. "
                        "Best Before: 24 months from mfg. Made in India. "
                        "Consumer Care: 1800-22-4080."
                    ),
                    "specifications": {
                        "Brand": "Colgate",
                        "Net Quantity": "200 g",
                        "Category": "Oral Care",
                    },
                    "brand": "Colgate",
                    "category": "Oral Care",
                    "retail_price": 109.0,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class DeclarationEvidence(BaseModel):
    """Evidence extracted for a single Legal Metrology declaration (D01-D07)."""

    declaration_name: str
    evidence_detected: bool
    evidence_text: str
    source: str


class UncertaintyDetail(BaseModel):
    """Uncertainty metrics returned alongside the prediction."""

    status: str = Field(description="LOW | MEDIUM | HIGH")
    score: float = Field(description="Uncertainty score in [0.0, 1.0]; higher = more uncertain.")
    reason: Optional[str] = None
    abstained: bool = Field(
        default=False,
        description="True when the model abstained to UNCERTAIN rather than committing.",
    )


class PredictResponse(BaseModel):
    """
    Response payload for POST /predict.

    `prediction` is the final compliance verdict after calibrated abstention:
        - COMPLIANT
        - NON_COMPLIANT
        - UNCERTAIN  (model confidence below threshold)

    `confidence` is the evidence-quality-calibrated probability (0.0–1.0).
    It is the product of raw model probability and an evidence quality factor
    (OCR quality × declaration completeness × conflict penalty).
    """

    prediction: str = Field(
        description="Final compliance verdict: COMPLIANT | NON_COMPLIANT | UNCERTAIN"
    )
    confidence: float = Field(
        description="Evidence-quality-calibrated confidence score in [0.0, 1.0]."
    )
    raw_prediction: str = Field(
        description="Raw ML model argmax label before abstention logic is applied."
    )
    decision_basis: str = Field(
        description="Human-readable explanation of the primary compliance decision factors."
    )
    uncertainty: UncertaintyDetail
    declarations_evidence: Dict[str, DeclarationEvidence] = Field(
        description="Per-declaration (D01–D07) evidence extracted from input text."
    )
    model_version: str = Field(description="Version identifier of the deployed model.")
    feature_set: str = Field(description="Feature set used by the model (e.g. SET_A).")
    disclaimer: str = Field(
        default=(
            "Automated Legal Metrology compliance screening. "
            "Not a statutory judicial determination."
        )
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": "COMPLIANT",
                    "confidence": 0.93,
                    "raw_prediction": "COMPLIANT",
                    "decision_basis": "All mandatory statutory declarations verified on packaging",
                    "uncertainty": {"status": "LOW", "score": 0.07, "reason": None, "abstained": False},
                    "declarations_evidence": {
                        "D01": {
                            "declaration_name": "Name & Address of Manufacturer / Packer / Importer",
                            "evidence_detected": True,
                            "evidence_text": "Entity: Colgate-Palmolive (India) Ltd | Address Markers: Mumbai",
                            "source": "OCR_AND_METADATA",
                        }
                    },
                    "model_version": "v2.0-product-compliance",
                    "feature_set": "SET_A",
                    "disclaimer": (
                        "Automated Legal Metrology compliance screening. "
                        "Not a statutory judicial determination."
                    ),
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    feature_set: Optional[str] = None


class RootResponse(BaseModel):
    """Response schema for GET /."""

    service: str
    version: str
    status: str
    docs_url: str


class ErrorResponse(BaseModel):
    """Generic error response schema."""

    detail: str
    error_type: Optional[str] = None
