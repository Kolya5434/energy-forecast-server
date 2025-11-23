"""
Pytest configuration and fixtures for API testing.
"""
import pytest
from fastapi.testclient import TestClient

# Import from main to ensure routes are registered
from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request payload."""
    return {
        "model_ids": ["XGBoost_Tuned"],
        "forecast_horizon": 7
    }


@pytest.fixture
def sample_prediction_request_with_conditions():
    """Sample prediction request with weather and calendar conditions."""
    return {
        "model_ids": ["XGBoost_Tuned"],
        "forecast_horizon": 7,
        "weather": {
            "temperature": 15.0,
            "humidity": 65.0
        },
        "calendar": {
            "is_holiday": False,
            "is_weekend": False
        }
    }


@pytest.fixture
def sample_simulation_request():
    """Sample simulation request payload."""
    return {
        "model_id": "XGBoost_Tuned",
        "forecast_horizon": 7,
        "feature_overrides": [],
        "weather": {
            "temperature": 20.0
        }
    }


@pytest.fixture
def sample_compare_request():
    """Sample compare scenarios request payload."""
    return {
        "model_id": "XGBoost_Tuned",
        "forecast_horizon": 7,
        "scenarios": [
            {"name": "cold_weather", "weather": {"temperature": -5}},
            {"name": "hot_weather", "weather": {"temperature": 35}}
        ]
    }