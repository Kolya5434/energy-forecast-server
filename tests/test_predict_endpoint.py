"""
Tests for POST /api/predict endpoint.
"""
import pytest


class TestPredictEndpoint:
    """Tests for POST /api/predict endpoint."""

    def test_predict_returns_200_with_valid_request(self, client, sample_prediction_request):
        """Test that predict endpoint returns 200 with valid request."""
        response = client.post("/api/predict", json=sample_prediction_request)
        assert response.status_code == 200

    def test_predict_returns_list_of_predictions(self, client, sample_prediction_request):
        """Test that predict endpoint returns a list of predictions."""
        response = client.post("/api/predict", json=sample_prediction_request)
        data = response.json()
        assert isinstance(data, list)

    def test_predict_response_has_required_fields(self, client, sample_prediction_request):
        """Test that each prediction has required fields."""
        response = client.post("/api/predict", json=sample_prediction_request)
        data = response.json()

        for prediction in data:
            assert "model_id" in prediction
            assert "forecast" in prediction
            assert "metadata" in prediction

    def test_predict_with_weather_conditions(self, client, sample_prediction_request_with_conditions):
        """Test prediction with weather conditions."""
        response = client.post("/api/predict", json=sample_prediction_request_with_conditions)
        assert response.status_code == 200

    def test_predict_with_multiple_models(self, client):
        """Test prediction with multiple models."""
        request = {
            "model_ids": ["XGBoost_Tuned", "SARIMA"],
            "forecast_horizon": 7
        }
        response = client.post("/api/predict", json=request)
        # Should return 200 even if some models don't exist
        assert response.status_code in [200, 404]

    def test_predict_invalid_model_skipped(self, client):
        """Test that invalid model is skipped (returns 200 with empty or filtered results)."""
        request = {
            "model_ids": ["NonExistentModel_12345"],
            "forecast_horizon": 7
        }
        response = client.post("/api/predict", json=request)
        # API skips unknown models and returns 200 with empty list or 404 if all models invalid
        assert response.status_code in [200, 404]

    def test_predict_missing_model_ids_returns_422(self, client):
        """Test that missing model_ids returns 422 validation error."""
        request = {
            "forecast_horizon": 7
        }
        response = client.post("/api/predict", json=request)
        assert response.status_code == 422

    def test_predict_missing_forecast_horizon_returns_422(self, client):
        """Test that missing forecast_horizon returns 422 validation error."""
        request = {
            "model_ids": ["XGBoost_Tuned"]
        }
        response = client.post("/api/predict", json=request)
        assert response.status_code == 422

    def test_predict_empty_model_ids_returns_error(self, client):
        """Test that empty model_ids list returns error."""
        request = {
            "model_ids": [],
            "forecast_horizon": 7
        }
        response = client.post("/api/predict", json=request)
        assert response.status_code in [400, 404, 422]

    def test_predict_with_all_conditions(self, client):
        """Test prediction with all available conditions."""
        request = {
            "model_ids": ["XGBoost_Tuned"],
            "forecast_horizon": 7,
            "weather": {
                "temperature": 15.0,
                "humidity": 65.0,
                "wind_speed": 5.5
            },
            "calendar": {
                "is_holiday": False,
                "is_weekend": False
            },
            "include_confidence": True,
            "include_patterns": True
        }
        response = client.post("/api/predict", json=request)
        assert response.status_code == 200

    def test_predict_forecast_contains_dates(self, client, sample_prediction_request):
        """Test that forecast contains date keys."""
        response = client.post("/api/predict", json=sample_prediction_request)
        data = response.json()

        if data:
            forecast = data[0].get("forecast", {})
            assert len(forecast) > 0
            # Keys should be date strings
            for key in forecast.keys():
                assert isinstance(key, str)
