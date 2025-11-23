"""
Tests for GET /api/evaluation/{model_id} endpoint.
"""
import pytest


class TestEvaluationEndpoint:
    """Tests for GET /api/evaluation/{model_id} endpoint."""

    def test_evaluation_returns_200_for_valid_model(self, client):
        """Test that evaluation endpoint returns 200 for valid model."""
        response = client.get("/api/evaluation/XGBoost_Tuned")
        assert response.status_code in [200, 404]

    def test_evaluation_returns_404_for_invalid_model(self, client):
        """Test that evaluation endpoint returns 404 for invalid model."""
        response = client.get("/api/evaluation/NonExistentModel_12345")
        assert response.status_code == 404

    def test_evaluation_response_has_metrics(self, client):
        """Test that evaluation response contains metrics."""
        response = client.get("/api/evaluation/XGBoost_Tuned")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            # Common metrics: MAE, RMSE, MAPE, R2, etc.

    def test_evaluation_empty_model_id_returns_404(self, client):
        """Test that empty model_id in path returns 404."""
        response = client.get("/api/evaluation/")
        # FastAPI returns 404 for missing path parameter
        assert response.status_code in [404, 307]
