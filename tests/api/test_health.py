"""
test_health.py  — TEST 1
========================
Verifies the GET /health endpoint.

Checks
------
- Returns HTTP 200
- Response JSON has `status` == "healthy"
- Response JSON has `model_loaded` (bool)
- When model is loaded, `model_name` and `feature_set` are non-empty strings
"""

import pytest


class TestHealthEndpoint:
    """Test suite for GET /health."""

    def test_health_returns_200(self, client):
        """GET /health must return HTTP 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Body: {response.text}"
        )

    def test_health_status_is_healthy(self, client):
        """Response body must contain status='healthy'."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data, "Response missing 'status' key"
        assert data["status"] == "healthy", (
            f"Expected status='healthy', got '{data['status']}'"
        )

    def test_health_model_loaded_field_present(self, client):
        """Response must contain a boolean 'model_loaded' field."""
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data, "Response missing 'model_loaded' key"
        assert isinstance(data["model_loaded"], bool), (
            f"'model_loaded' must be a bool, got {type(data['model_loaded'])}"
        )

    def test_health_model_metadata_when_loaded(self, client):
        """When model is loaded, model_name and feature_set must be non-empty strings."""
        response = client.get("/health")
        data = response.json()
        if data.get("model_loaded"):
            assert data.get("model_name"), "'model_name' is empty despite model being loaded"
            assert data.get("feature_set"), "'feature_set' is empty despite model being loaded"

    def test_health_response_schema(self, client):
        """Health response must contain all required schema keys."""
        response = client.get("/health")
        data = response.json()
        required_keys = {"status", "model_loaded"}
        missing = required_keys - data.keys()
        assert not missing, f"Health response missing keys: {missing}"
