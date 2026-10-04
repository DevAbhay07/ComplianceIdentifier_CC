"""
test_prediction_schema.py  — TEST 4
=====================================
Verifies that the POST /predict response conforms to the expected schema.

This test validates:
- All required top-level keys are present
- Data types are correct
- 'prediction' is one of the known label values
- 'confidence' is a float in [0.0, 1.0]
- 'uncertainty' sub-object has the expected structure
- 'declarations_evidence' maps D01-D07 to the correct structure
- 'model_version' and 'feature_set' are non-empty strings
- 'disclaimer' is a non-empty string
"""

import pytest


VALID_LABELS = {"COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"}
VALID_UNCERTAINTY_STATUSES = {"LOW", "MEDIUM", "HIGH"}
DECLARATION_IDS = {"D01", "D02", "D03", "D04", "D05", "D06", "D07"}


class TestPredictionSchema:
    """Test suite for POST /predict — response schema validation."""

    @pytest.fixture(autouse=True)
    def response_data(self, client, valid_payload):
        """
        Runs a valid prediction request and stores the response.
        Skips all tests in this class if the model is not loaded.
        """
        response = client.post("/predict", json=valid_payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded — skipping schema tests")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Body: {response.text}"
        )
        self._data = response.json()

    def test_top_level_keys_present(self):
        """Response must contain all required top-level keys."""
        required = {
            "prediction",
            "confidence",
            "raw_prediction",
            "decision_basis",
            "uncertainty",
            "declarations_evidence",
            "model_version",
            "feature_set",
            "disclaimer",
        }
        missing = required - self._data.keys()
        assert not missing, f"Response missing required keys: {missing}"

    def test_prediction_is_valid_label(self):
        """'prediction' must be COMPLIANT, NON_COMPLIANT, or UNCERTAIN."""
        assert self._data["prediction"] in VALID_LABELS, (
            f"Invalid label: '{self._data['prediction']}'"
        )

    def test_raw_prediction_is_valid_label(self):
        """'raw_prediction' must be COMPLIANT or NON_COMPLIANT (never UNCERTAIN)."""
        assert self._data["raw_prediction"] in {"COMPLIANT", "NON_COMPLIANT"}, (
            f"Invalid raw_prediction: '{self._data['raw_prediction']}'"
        )

    def test_confidence_is_float_in_range(self):
        """'confidence' must be a float in [0.0, 1.0]."""
        conf = self._data["confidence"]
        assert isinstance(conf, float), f"confidence must be float, got {type(conf)}"
        assert 0.0 <= conf <= 1.0, f"confidence out of range: {conf}"

    def test_uncertainty_structure(self):
        """'uncertainty' must have status, score, and abstained fields."""
        unc = self._data["uncertainty"]
        assert isinstance(unc, dict), f"'uncertainty' must be a dict, got {type(unc)}"
        assert "status" in unc, "'uncertainty' missing 'status'"
        assert "score" in unc, "'uncertainty' missing 'score'"
        assert "abstained" in unc, "'uncertainty' missing 'abstained'"
        assert unc["status"] in VALID_UNCERTAINTY_STATUSES, (
            f"Invalid uncertainty status: '{unc['status']}'"
        )
        assert isinstance(unc["score"], float), f"uncertainty.score must be float"
        assert 0.0 <= unc["score"] <= 1.0, f"uncertainty.score out of range: {unc['score']}"
        assert isinstance(unc["abstained"], bool), "uncertainty.abstained must be bool"

    def test_declarations_evidence_has_d01_to_d07(self):
        """'declarations_evidence' must contain entries for D01 through D07."""
        decls = self._data["declarations_evidence"]
        assert isinstance(decls, dict), f"'declarations_evidence' must be a dict"
        missing = DECLARATION_IDS - decls.keys()
        assert not missing, f"'declarations_evidence' missing: {missing}"

    def test_each_declaration_has_required_fields(self):
        """Each declaration entry must have declaration_name, evidence_detected, evidence_text, source."""
        required_decl_keys = {"declaration_name", "evidence_detected", "evidence_text", "source"}
        for did, decl in self._data["declarations_evidence"].items():
            missing = required_decl_keys - decl.keys()
            assert not missing, f"{did} declaration missing keys: {missing}"
            assert isinstance(decl["evidence_detected"], bool), (
                f"{did}.evidence_detected must be bool"
            )

    def test_model_version_is_non_empty_string(self):
        """'model_version' must be a non-empty string."""
        mv = self._data["model_version"]
        assert isinstance(mv, str) and mv, f"'model_version' must be non-empty string, got: {mv!r}"

    def test_feature_set_is_non_empty_string(self):
        """'feature_set' must be a non-empty string."""
        fs = self._data["feature_set"]
        assert isinstance(fs, str) and fs, f"'feature_set' must be non-empty string, got: {fs!r}"

    def test_disclaimer_is_non_empty_string(self):
        """'disclaimer' must be a non-empty string."""
        disc = self._data["disclaimer"]
        assert isinstance(disc, str) and disc, f"'disclaimer' must be non-empty string"

    def test_decision_basis_is_non_empty_string(self):
        """'decision_basis' must be a non-empty string."""
        db = self._data["decision_basis"]
        assert isinstance(db, str) and db, f"'decision_basis' must be non-empty string"
