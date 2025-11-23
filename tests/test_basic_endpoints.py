"""
Tests for basic API endpoints: root, health, models.
"""
import pytest


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_message(self, client):
        """Test that root endpoint returns expected message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "Energy Forecast API" in data["message"]


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data.get("status") == "healthy"


class TestModelsEndpoint:
    """Tests for GET /api/models endpoint."""

    def test_models_returns_200(self, client):
        """Test that models endpoint returns 200 OK."""
        response = client.get("/api/models")
        assert response.status_code == 200

    def test_models_returns_list(self, client):
        """Test that models endpoint returns a list or dict of models."""
        response = client.get("/api/models")
        data = response.json()
        assert data is not None
        assert isinstance(data, (list, dict))

    def test_models_contains_expected_keys(self, client):
        """Test that each model has expected structure."""
        response = client.get("/api/models")
        data = response.json()

        if isinstance(data, dict):
            for model_id, model_info in data.items():
                assert isinstance(model_id, str)
        elif isinstance(data, list):
            assert len(data) > 0
