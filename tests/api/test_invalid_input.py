"""
test_invalid_input.py  — TEST 3
=================================
Verifies that POST /predict returns HTTP 422 for malformed or incomplete inputs.

FastAPI / Pydantic v2 automatically validates request bodies and returns 422
when required fields are missing or values fail validation constraints.

The NitiMark API requires at least one of `title` or `description` to be
a non-empty string (enforced by @model_validator in PredictRequest).
"""

import pytest


class TestInvalidInput:
    """Test suite for POST /predict — invalid / malformed inputs."""

    def test_empty_body_returns_422(self, client):
        """Sending an empty JSON body {} must return HTTP 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422, (
            f"Expected 422 for empty body, got {response.status_code}. Body: {response.text}"
        )

    def test_empty_title_and_description_returns_422(self, client):
        """Both title='' and description='' violates the model_validator and must return 422."""
        response = client.post("/predict", json={"title": "", "description": ""})
        assert response.status_code == 422, (
            f"Expected 422 for empty strings, got {response.status_code}. Body: {response.text}"
        )

    def test_negative_retail_price_returns_422(self, client):
        """retail_price < 0 violates the ge=0.0 constraint and must return 422."""
        response = client.post(
            "/predict",
            json={"title": "Colgate 200g", "retail_price": -50.0},
        )
        assert response.status_code == 422, (
            f"Expected 422 for negative retail_price, got {response.status_code}."
        )

    def test_negative_discounted_price_returns_422(self, client):
        """discounted_price < 0 violates the ge=0.0 constraint and must return 422."""
        response = client.post(
            "/predict",
            json={"title": "Dettol 250ml", "discounted_price": -1.0},
        )
        assert response.status_code == 422, (
            f"Expected 422 for negative discounted_price, got {response.status_code}."
        )

    def test_non_dict_specifications_returns_422(self, client):
        """specifications must be a dict; sending a string must return 422."""
        response = client.post(
            "/predict",
            json={
                "title": "Colgate 200g",
                "specifications": "this is not a dict",
            },
        )
        assert response.status_code == 422, (
            f"Expected 422 for non-dict specifications, got {response.status_code}."
        )

    def test_missing_title_only_description_allowed(self, client):
        """Sending description only (title omitted) should succeed — title defaults to empty."""
        response = client.post(
            "/predict",
            json={"description": "Manufactured by Colgate-Palmolive. MRP Rs 99. Net Qty 200g."},
        )
        # Should be 200 (model loaded) or 503 (model not loaded), NOT 422
        assert response.status_code in {200, 503}, (
            f"Expected 200/503 for description-only input, got {response.status_code}."
        )

    def test_422_response_contains_detail(self, client):
        """422 responses must contain a 'detail' field with error information."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data, f"422 response missing 'detail'. Got: {list(data.keys())}"
