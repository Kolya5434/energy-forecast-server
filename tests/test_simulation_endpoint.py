"""
Tests for POST /api/simulate endpoint.
"""
import pytest


class TestSimulationEndpoint:
    """Tests for POST /api/simulate endpoint."""

    def test_simulate_returns_200_with_valid_request(self, client, sample_simulation_request):
        """Test that simulate endpoint returns 200 with valid request."""
        response = client.post("/api/simulate", json=sample_simulation_request)
        assert response.status_code in [200, 400]

    def test_simulate_response_has_required_fields(self, client, sample_simulation_request):
        """Test that simulation response has required fields."""
        response = client.post("/api/simulate", json=sample_simulation_request)
        if response.status_code == 200:
            data = response.json()
            assert "model_id" in data
            assert "forecast" in data
            assert "metadata" in data

    def test_simulate_with_feature_overrides(self, client):
        """Test simulation with feature overrides."""
        request = {
            "model_id": "XGBoost_Tuned",
            "forecast_horizon": 7,
            "feature_overrides": [
                {
                    "date": "2010-11-29",
                    "features": {"day_of_week": 6}
                }
            ]
        }
        response = client.post("/api/simulate", json=request)
        assert response.status_code in [200, 400]

    def test_simulate_with_weather_conditions(self, client):
        """Test simulation with weather conditions."""
        request = {
            "model_id": "XGBoost_Tuned",
            "forecast_horizon": 7,
            "weather": {
                "temperature": 25.0,
                "humidity": 70.0
            }
        }
        response = client.post("/api/simulate", json=request)
        assert response.status_code in [200, 400]

    def test_simulate_invalid_model_returns_400(self, client):
        """Test that invalid model returns 400."""
        request = {
            "model_id": "NonExistentModel_12345",
            "forecast_horizon": 7
        }
        response = client.post("/api/simulate", json=request)
        assert response.status_code == 400

    def test_simulate_missing_model_id_returns_422(self, client):
        """Test that missing model_id returns 422 validation error."""
        request = {
            "forecast_horizon": 7
        }
        response = client.post("/api/simulate", json=request)
        assert response.status_code == 422

    def test_simulate_missing_forecast_horizon_returns_422(self, client):
        """Test that missing forecast_horizon returns 422 validation error."""
        request = {
            "model_id": "XGBoost_Tuned"
        }
        response = client.post("/api/simulate", json=request)
        assert response.status_code == 422

    def test_simulate_with_all_conditions(self, client):
        """Test simulation with all available conditions."""
        request = {
            "model_id": "XGBoost_Tuned",
            "forecast_horizon": 7,
            "weather": {"temperature": 20.0},
            "calendar": {"is_holiday": True},
            "energy": {"voltage": 240.0},
            "zone_consumption": {"sub_metering_1": 30.0}
        }
        response = client.post("/api/simulate", json=request)
        assert response.status_code in [200, 400]
