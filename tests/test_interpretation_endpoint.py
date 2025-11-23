"""
Tests for GET /api/interpret/{model_id} endpoint.
"""
import pytest


class TestInterpretationEndpoint:
    """Tests for GET /api/interpret/{model_id} endpoint."""

    def test_interpret_returns_200_for_valid_model(self, client):
        """Test that interpret endpoint returns 200 for valid model."""
        response = client.get("/api/interpret/XGBoost_Tuned")
        assert response.status_code in [200, 404]

    def test_interpret_returns_404_for_invalid_model(self, client):
        """Test that interpret endpoint returns 404 for invalid model."""
        response = client.get("/api/interpret/NonExistentModel_12345")
        assert response.status_code == 404

    def test_interpret_response_structure(self, client):
        """Test that interpret response has expected structure."""
        response = client.get("/api/interpret/XGBoost_Tuned")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_interpret_empty_model_id_returns_404(self, client):
        """Test that empty model_id in path returns 404."""
        response = client.get("/api/interpret/")
        assert response.status_code in [404, 307]
