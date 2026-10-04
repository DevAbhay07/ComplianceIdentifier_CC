"""
model_service.py
================
Service abstraction that loads and runs the existing NitiMark-ML product
compliance model.  This module is the single source of truth for all
ML inference in the API; no inference logic lives in main.py.

Architecture
------------
    PredictRequest (text input)
         │
         ▼
    evidence_parser  (parse_product_evidence)
         │
         ▼
    feature_extractor  (ProductComplianceFeatureExtractor → SET_A features)
         │
         ▼
    ProductComplianceModel.predict_single
         │
         ▼
    PredictResponse (structured JSON)

The model is loaded once at startup via ModelService.__init__ and reused
on every request.  Loading errors raise at startup, not at request time.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.models.product_compliance_model import ProductComplianceModel
from src.features.product_compliance_feature_extractor import ProductComplianceFeatureExtractor
from src.features.product_training_representation import ProductComplianceTrainingRecord
from src.inference.evidence_parser import parse_product_evidence

from api.schemas import (
    DeclarationEvidence,
    PredictRequest,
    PredictResponse,
    UncertaintyDetail,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = (
    Path(os.environ.get("MODEL_PATH", ""))
    if os.environ.get("MODEL_PATH")
    else _REPO_ROOT / "models" / "product_compliance" / "best_product_compliance_model.joblib"
)


# ---------------------------------------------------------------------------
# Model Service
# ---------------------------------------------------------------------------

class ModelService:
    """
    Loads the frozen ProductComplianceModel artifact and exposes a
    clean `.predict()` interface.

    Designed to be instantiated once (at FastAPI startup) and shared
    across all request handlers.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._model: Optional[ProductComplianceModel] = None
        self._feature_extractor: Optional[ProductComplianceFeatureExtractor] = None
        self._loaded: bool = False
        self._model_name: str = "unknown"
        self._feature_set: str = "unknown"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model artifact from disk.  Call once at API startup."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"[ModelService] Model artifact not found at: {self._model_path}\n"
                "Ensure models/product_compliance/best_product_compliance_model.joblib exists."
            )

        logger.info("[ModelService] Loading ProductComplianceModel from %s …", self._model_path)
        self._model = ProductComplianceModel.load(str(self._model_path))
        self._feature_extractor = ProductComplianceFeatureExtractor(
            default_feature_set=self._model.feature_set
        )
        self._loaded = True
        self._model_name = getattr(self._model, "model_name", "ProductComplianceModel")
        self._feature_set = getattr(self._model, "feature_set", "SET_A")
        logger.info(
            "[ModelService] Model loaded: %s | Feature Set: %s",
            self._model_name,
            self._feature_set,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def feature_set(self) -> str:
        return self._feature_set

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Run the full inference pipeline for a single product.

        Steps
        -----
        1. Parse text evidence for D01-D07 declarations.
        2. Build a ProductComplianceTrainingRecord from the parsed evidence.
        3. Extract feature vector (SET_A by default).
        4. Run model.predict_single with evidence quality modifiers.
        5. Return structured PredictResponse.
        """
        if not self._loaded or self._model is None or self._feature_extractor is None:
            raise RuntimeError("[ModelService] Model not loaded. Call .load() first.")

        # ── Step 1: Parse evidence ────────────────────────────────────────
        specs = request.specifications or {}
        evidence_res = parse_product_evidence(
            title=request.title,
            description=request.description,
            specifications=specs,
        )
        declarations_evidence: Dict[str, Any] = evidence_res.get("declarations", {})
        diagnostics: Dict[str, Any] = evidence_res.get("diagnostics", {})

        # ── Step 2: Build training record from evidence ───────────────────
        # Map the evidence parser output into a declaration dict that
        # ProductComplianceFeatureExtractor understands.
        decl_dict: Dict[str, Dict[str, Any]] = {}
        for did, ev in declarations_evidence.items():
            if ev.get("evidence_detected"):
                decl_dict[did] = {"status": "PRESENT", "confidence": 0.85}
            else:
                decl_dict[did] = {"status": "ABSENT", "confidence": 0.0}

        # Presence count and completeness ratio
        present_count = sum(1 for d in decl_dict.values() if d["status"] == "PRESENT")
        completeness_ratio = present_count / max(len(decl_dict), 1)

        record = ProductComplianceTrainingRecord(
            product_id="API_INFERENCE",
            source_dataset="flipkart",
            target_provenance="UNKNOWN",
            overall_target="UNKNOWN",
            category=request.category or "General Commodity",
            brand=request.brand or "",
            product_title=request.title,
            raw_text=f"{request.title} {request.description}".strip(),
            declarations=decl_dict,
            # D01–D07 value dicts — populated from available evidence hints
            d01_values=_extract_d01_values(request),
            d02_values={"common_name": request.title, "identity_confidence": 0.8},
            d03_values=_extract_d03_values(request),
            d04_values={},
            d05_values=_extract_d05_values(request),
            d06_values={},
            d07_values={},
            ocr_quality=0.85,
            text_length=len(f"{request.title} {request.description}".strip()),
            image_count=0,
        )

        # ── Step 3: Extract feature vector ───────────────────────────────
        X = self._feature_extractor.transform([record], feature_set=self._feature_set)

        # ── Step 4: Run model inference ───────────────────────────────────
        pred_obj = self._model.predict_single(
            features=X[0],
            completeness_ratio=completeness_ratio,
            has_conflicts=False,
            ocr_confidence=0.85,
        )

        # Calibrated confidence (mirrors predict_product_compliance.py logic)
        raw_conf = float(pred_obj.confidence)
        ocr_factor = 0.85
        comp_factor = min(1.0, max(0.2, completeness_ratio))
        calibrated_conf = round(max(0.1, raw_conf * ocr_factor * comp_factor), 4)

        unc_score = round(1.0 - calibrated_conf, 4)
        if pred_obj.abstention_applied or pred_obj.final_prediction == "UNCERTAIN" or unc_score >= 0.45:
            unc_status = "HIGH"
        elif unc_score >= 0.25:
            unc_status = "MEDIUM"
        else:
            unc_status = "LOW"

        # ── Step 5: Build response ────────────────────────────────────────
        decl_evidence_out: Dict[str, DeclarationEvidence] = {
            did: DeclarationEvidence(
                declaration_name=ev.get("declaration_name", did),
                evidence_detected=ev.get("evidence_detected", False),
                evidence_text=ev.get("evidence_text", ""),
                source=ev.get("source", "UNKNOWN"),
            )
            for did, ev in declarations_evidence.items()
        }

        return PredictResponse(
            prediction=pred_obj.final_prediction,
            confidence=calibrated_conf,
            raw_prediction=pred_obj.raw_prediction,
            decision_basis=pred_obj.abstention_reason or _build_decision_basis(decl_dict, pred_obj.final_prediction),
            uncertainty=UncertaintyDetail(
                status=unc_status,
                score=unc_score,
                reason=pred_obj.abstention_reason,
                abstained=pred_obj.abstention_applied,
            ),
            declarations_evidence=decl_evidence_out,
            model_version=getattr(self._model, "model_version", "v2.0-product-compliance"),
            feature_set=self._feature_set,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_d01_values(req: PredictRequest) -> Dict[str, Any]:
    """Extract manufacturer/packer hints from request."""
    text = f"{req.title} {req.description}".lower()
    return {
        "entity_name": req.brand or "",
        "address_markers": bool(any(k in text for k in ["mumbai", "delhi", "bengaluru", "india", "pin"])),
        "pincode": None,
        "is_packer": "packed by" in text,
        "is_importer": "imported by" in text,
        "is_marketer": "marketed by" in text,
    }


def _extract_d03_values(req: PredictRequest) -> Dict[str, Any]:
    """Extract net quantity hints from request."""
    import re
    text = f"{req.title} {req.description}"
    specs = req.specifications or {}
    full = f"{text} {' '.join(str(v) for v in specs.values())}"
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(ml|l|litre|liter|g|gm|kg|oz|pcs|pc|units|capsules|tablets|count)",
        full,
        re.IGNORECASE,
    )
    if m:
        return {"quantity": float(m.group(1)), "unit": m.group(2).lower()}
    return {}


def _extract_d05_values(req: PredictRequest) -> Dict[str, Any]:
    """Extract price hints from request."""
    import re
    if req.retail_price is not None:
        return {"mrp": req.retail_price, "tax_inclusive": True}
    text = f"{req.title} {req.description}"
    m = re.search(r"(?:mrp|rs\.?|₹)\s*([\d,]+(?:\.\d{2})?)", text, re.IGNORECASE)
    if m:
        try:
            return {"mrp": float(m.group(1).replace(",", "")), "tax_inclusive": True}
        except ValueError:
            pass
    return {}


def _build_decision_basis(decl_dict: Dict[str, Dict[str, Any]], final_prediction: str) -> str:
    """Build a human-readable decision basis string."""
    absent = [
        did for did, d in decl_dict.items() if d.get("status") == "ABSENT"
    ]
    NAMES = {
        "D01": "Manufacturer/Packer name & address",
        "D02": "Generic commodity name",
        "D03": "Net quantity",
        "D04": "Date of manufacture/expiry",
        "D05": "Maximum Retail Price (MRP)",
        "D06": "Consumer care contact",
        "D07": "Country of origin",
    }
    if absent:
        missing = ", ".join(NAMES.get(d, d) for d in absent)
        return f"Missing mandatory declarations: {missing}"
    if final_prediction == "COMPLIANT":
        return "All mandatory statutory declarations verified on packaging"
    return "Based on multi-declaration feature analysis"
