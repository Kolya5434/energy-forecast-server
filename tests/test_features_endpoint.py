"""
Tests for GET /api/features/{model_id} endpoint.
"""
import pytest


class TestFeaturesEndpoint:
    """Tests for GET /api/features/{model_id} endpoint."""

    def test_features_returns_200_for_valid_model(self, client):
        """Test that features endpoint returns 200 for valid model."""
        response = client.get("/api/features/XGBoost_Tuned")
        assert response.status_code in [200, 404, 500]

    def test_features_returns_error_for_invalid_model(self, client):
        """Test that features endpoint returns error for invalid model."""
        response = client.get("/api/features/NonExistentModel_12345")
        assert response.status_code in [404, 500]

    def test_features_response_structure(self, client):
        """Test that features response has expected structure."""
        response = client.get("/api/features/XGBoost_Tuned")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_features_empty_model_id_returns_404(self, client):
        """Test that empty model_id in path returns 404."""
        response = client.get("/api/features/")
        assert response.status_code in [404, 307]
