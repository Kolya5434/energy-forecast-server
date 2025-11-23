"""
Tests for GET /api/historical endpoint.
"""
import pytest


class TestHistoricalEndpoint:
    """Tests for GET /api/historical endpoint."""

    def test_historical_returns_200_with_defaults(self, client):
        """Test that historical endpoint returns 200 with default params."""
        response = client.get("/api/historical")
        assert response.status_code == 200

    def test_historical_with_days_param(self, client):
        """Test historical endpoint with days parameter."""
        response = client.get("/api/historical?days=7")
        assert response.status_code == 200

    def test_historical_with_granularity_daily(self, client):
        """Test historical endpoint with daily granularity."""
        response = client.get("/api/historical?granularity=daily")
        assert response.status_code == 200

    def test_historical_with_granularity_hourly(self, client):
        """Test historical endpoint with hourly granularity."""
        response = client.get("/api/historical?granularity=hourly")
        assert response.status_code == 200

    def test_historical_with_include_stats(self, client):
        """Test historical endpoint with include_stats=true."""
        response = client.get("/api/historical?include_stats=true")
        assert response.status_code == 200

    def test_historical_invalid_granularity_returns_400(self, client):
        """Test that invalid granularity returns 400."""
        response = client.get("/api/historical?granularity=invalid")
        assert response.status_code == 400

    def test_historical_days_below_min_returns_400(self, client):
        """Test that days < 1 returns 400."""
        response = client.get("/api/historical?days=0")
        assert response.status_code == 400

    def test_historical_days_above_max_returns_400(self, client):
        """Test that days > 365 returns 400."""
        response = client.get("/api/historical?days=400")
        assert response.status_code == 400

    def test_historical_response_structure(self, client):
        """Test that historical response has expected structure."""
        response = client.get("/api/historical?days=7")
        data = response.json()
        assert isinstance(data, dict)

    def test_historical_with_all_params(self, client):
        """Test historical endpoint with all parameters."""
        response = client.get("/api/historical?days=14&granularity=daily&include_stats=true")
        assert response.status_code == 200
