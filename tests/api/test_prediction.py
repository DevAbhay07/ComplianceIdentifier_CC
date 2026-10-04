"""
test_prediction.py  — TEST 2
=============================
Verifies that POST /predict returns a valid response for a real,
well-formed product input.

The payload uses actual feature fields consumed by the NitiMark ML model.
The test does NOT assert a specific compliance label (COMPLIANT vs NON_COMPLIANT)
because that depends on runtime model state — it only validates that the
endpoint responds correctly with the expected structure and value ranges.
"""

import pytest


VALID_LABEL_VALUES = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"}


class TestPredictionEndpoint:
    """Test suite for POST /predict — valid inputs."""

    def test_predict_returns_200_for_valid_payload(self, client, valid_payload):
        """POST /predict with a valid payload must return HTTP 200."""
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Body: {response.text}"
        )

    def test_predict_response_has_prediction_field(self, client, valid_payload):
        """Response must contain a 'prediction' field."""
        response = client.post("/predict", json=valid_payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded — skipping prediction content test")
        data = response.json()
        assert "prediction" in data, f"Response missing 'prediction'. Got: {list(data.keys())}"

    def test_predict_prediction_is_valid_label(self, client, valid_payload):
        """'prediction' must be one of: COMPLIANT, NON_COMPLIANT, UNCERTAIN."""
        response = client.post("/predict", json=valid_payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded — skipping label validation test")
        data = response.json()
        assert data["prediction"] in VALID_LABEL_VALUES, (
            f"Invalid prediction label: '{data['prediction']}'. "
            f"Expected one of {VALID_LABEL_VALUES}"
        )

    def test_predict_confidence_is_float_in_range(self, client, valid_payload):
        """'confidence' must be a float in [0.0, 1.0]."""
        response = client.post("/predict", json=valid_payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded — skipping confidence range test")
        data = response.json()
        assert "confidence" in data, "Response missing 'confidence'"
        assert isinstance(data["confidence"], float), (
            f"'confidence' must be a float, got {type(data['confidence'])}"
        )
        assert 0.0 <= data["confidence"] <= 1.0, (
            f"'confidence' out of range: {data['confidence']}"
        )

    def test_predict_with_minimal_valid_payload(self, client):
        """POST /predict with only a title (minimum valid input) must return 200."""
        minimal = {"title": "Colgate Toothpaste 100g MRP Rs 59"}
        response = client.post("/predict", json=minimal)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Body: {response.text}"
        )

    def test_predict_with_specifications(self, client):
        """POST /predict with specifications dict must work correctly."""
        payload = {
            "title": "Dettol Hand Wash 250ml",
            "description": "Manufactured by Reckitt Benckiser India Ltd. MRP Rs 99. Made in India.",
            "specifications": {
                "Net Quantity": "250 ml",
                "Brand": "Dettol",
                "Country of Origin": "India",
            },
            "retail_price": 99.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code in {200, 503}, (
            f"Unexpected status: {response.status_code}"
        )
