"""
main.py
=======
FastAPI application entry point for the NitiMark Product Compliance ML API.

Endpoints
---------
  GET  /             → Service info
  GET  /health       → Health check (model loaded?)
  POST /predict      → Compliance prediction
  GET  /docs         → Swagger UI (auto-generated)
  GET  /openapi.json → OpenAPI schema (auto-generated)

Run locally
-----------
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.model_service import ModelService
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RootResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Service (singleton — instantiated before app startup)
# ---------------------------------------------------------------------------
_model_service = ModelService()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once at startup; release on shutdown."""
    logger.info("=== NitiMark Product Compliance ML API — starting up ===")
    try:
        _model_service.load()
        logger.info("Model loaded successfully. API is ready.")
    except FileNotFoundError as exc:
        logger.error("Model file not found: %s", exc)
        # Allow API to start but /health will report model_loaded=False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error loading model: %s", exc)
    yield
    logger.info("=== NitiMark Product Compliance ML API — shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NitiMark — Product Compliance ML API",
    description=(
        "REST API that wraps the NitiMark ML model for Legal Metrology compliance "
        "classification of packaged commodity products.\n\n"
        "Submit product information (title, description, specifications) and receive "
        "a **COMPLIANT / NON_COMPLIANT / UNCERTAIN** verdict with confidence score."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error.", "error_type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    response_model=RootResponse,
    summary="Service Information",
    tags=["Meta"],
)
async def root() -> RootResponse:
    """Returns basic service metadata."""
    return RootResponse(
        service="NitiMark — Product Compliance ML API",
        version="1.0.0",
        status="running",
        docs_url="/docs",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Meta"],
)
async def health() -> HealthResponse:
    """
    Returns the health status of the service and whether the ML model
    was loaded successfully at startup.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=_model_service.is_loaded,
        model_name=_model_service.model_name if _model_service.is_loaded else None,
        feature_set=_model_service.feature_set if _model_service.is_loaded else None,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Product Compliance",
    tags=["Inference"],
    status_code=status.HTTP_200_OK,
)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Submit product information to receive a compliance prediction.

    **Input**: product title, description, optional specifications and metadata.

    **Output**: COMPLIANT | NON_COMPLIANT | UNCERTAIN verdict with:
    - Calibrated confidence score
    - Per-declaration (D01–D07) evidence summary
    - Uncertainty assessment

    The model evaluates whether the product listing contains the mandatory
    statutory declarations required by the Legal Metrology (Packaged Commodities)
    Rules in India (D01–D07).
    """
    if not _model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded. Please check server logs.",
        )

    try:
        result = _model_service.predict(request)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction error for request: %s", request.model_dump())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
