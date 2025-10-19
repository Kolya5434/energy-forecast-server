from fastapi import FastAPI
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Define the project's base directory for reliable path construction
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Intelligent Hybrid Energy Consumption Forecasting System",
    description="Інтелектуальна гібридна система прогнозування споживання енергії для «розумних» міст.",
    version="1.0.0"
)

# Configure CORS to allow requests from your frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://eneryge-forecast.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods
    allow_headers=["*"], # Allow all headers
)

# Expanded model configuration with metadata
AVAILABLE_MODELS = {
    # --- Classical models (daily data, no external features) ---
    "ARIMA": {
        "path": BASE_DIR / "models/arima_model.pkl",
        "type": "classical",
        "granularity": "daily",
        "feature_set": "none"
    },
    "SARIMA": {
        "path": BASE_DIR / "models/sarima_baseline_model.pkl",
        "type": "classical",
        "granularity": "daily",
        "feature_set": "none"
    },
    "Prophet": {
        "path": BASE_DIR / "models/prophet_baseline_model.json",
        "type": "classical",
        "granularity": "daily",
        "feature_set": "none"
    },

    # --- ML models (hourly data, full feature set) ---
    "RandomForest": {
        "path": BASE_DIR / "models/random_forest_model.pkl",
        "type": "ml",
        "granularity": "hourly",
        "feature_set": "full"
    },
    "XGBoost": {
        "path": BASE_DIR / "models/xgboost_model.pkl",
        "type": "ml",
        "granularity": "hourly",
        "feature_set": "full"
    },
    "LightGBM": {
        "path": BASE_DIR / "models/light_gbm_model.pkl",
        "type": "ml",
        "granularity": "hourly",
        "feature_set": "full"
    },

    # --- ML models & Ensembles (daily data, simple feature set) ---
    "XGBoost_Tuned": {
        "path": BASE_DIR / "models/xgboost_tuned_model.pkl",
        "type": "ml",
        "granularity": "daily",
        "feature_set": "simple"
    },
    "Voting": {
        "path": BASE_DIR / "models/voting_model.pkl",
        "type": "ensemble",
        "granularity": "daily",
        "feature_set": "simple"
    },
    "Stacking": {
        "path": BASE_DIR / "models/stacking_model.pkl",
        "type": "ensemble",
        "granularity": "daily",
        "feature_set": "simple"
    },

    # --- DL models (hourly data, base scaled features, sequential) ---
    "LSTM": {
        "path": BASE_DIR / "models/lstm_model.keras",
        "type": "dl",
        "granularity": "hourly",
        "feature_set": "base_scaled",
        "is_sequential": True,
        "sequence_length": 24
    },
    "GRU": {
        "path": BASE_DIR / "models/gru_model.keras",
        "type": "dl",
        "granularity": "hourly",
        "feature_set": "base_scaled",
        "is_sequential": True,
        "sequence_length": 24
    },
    "Transformer": {
        "path": BASE_DIR / "models/transformer_model.keras",
        "type": "dl",
        "granularity": "hourly",
        "feature_set": "base_scaled",
        "is_sequential": True,
        "sequence_length": 24
    },
}