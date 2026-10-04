"""
conftest.py
===========
Shared pytest fixtures for the NitiMark API test suite.

The FastAPI TestClient is used so tests run without a real uvicorn server
and without network calls. The ML model is loaded from disk (the real
artifact at models/product_compliance/best_product_compliance_model.joblib).

If the model file is absent (e.g. in a fresh CI checkout without LFS),
the `client` fixture still works but /predict will return 503.
Tests that require a loaded model are clearly marked with
`@pytest.mark.requires_model` so they can be skipped separately.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ---------------------------------------------------------------------------
# Shared client fixture (session-scoped — creates the app once per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Returns a FastAPI TestClient that shares the full app lifecycle
    (lifespan startup + shutdown) across all tests in the session.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Shared valid payload
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_payload() -> dict:
    """
    A realistic product payload that matches the real API input schema.

    This product (Colgate Strong Teeth Toothpaste) contains all 7 mandatory
    Legal Metrology declarations (D01-D07) in its description, so the model
    is expected to return COMPLIANT or very high confidence.
    """
    return {
        "title": "Colgate Strong Teeth Toothpaste 200g",
        "description": (
            "Manufactured by Colgate-Palmolive (India) Ltd, "
            "Sudarshan Chambers, 5th Floor, Bhulabhai Desai Road, Mumbai 400026. "
            "MRP Rs. 109 (incl. of all taxes). Net Qty: 200 g. "
            "Best Before: 24 months from mfg date. Country of Origin: India. "
            "Consumer Care: 1800-22-4080. Email: customercare@colgate.com"
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


@pytest.fixture
def non_compliant_payload() -> dict:
    """
    A product payload that is deliberately missing most mandatory declarations.
    The model is expected to return NON_COMPLIANT or UNCERTAIN.
    """
    return {
        "title": "Generic White Powder 500g",
        "description": "White powder. No further details.",
    }
