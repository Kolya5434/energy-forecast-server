from fastapi import FastAPI
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Intelligent Hybrid Energy Consumption Forecasting System",
    description="Інтелектуальна гібридна система прогнозування споживання енергії для «розумних» міст.",
    version="1.0.0"
)

origins = [
    "http://localhost:5173", # Адреса вашого Vite фронтенду
    "http://127.0.0.1:5173", # Іноді Vite використовує цю адресу
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Дозволити всі методи (GET, POST, etc.)
    allow_headers=["*"], # Дозволити всі заголовки
)

AVAILABLE_MODELS = {
    "ARIMA": {"path": BASE_DIR / "models/arima_model.pkl", "type": "classical"},
    "SARIMA": {"path": BASE_DIR / "models/sarima_baseline_model.pkl", "type": "classical"},
    "Prophet": {"path": BASE_DIR / "models/prophet_baseline_model.json", "type": "classical"},
    "RandomForest": {"path": BASE_DIR / "models/random_forest_model.pkl", "type": "ml"},
    "XGBoost": {"path": BASE_DIR / "models/xgboost_model.pkl", "type": "ml"},
    "LightGBM": {"path": BASE_DIR / "models/light_gbm_model.pkl", "type": "ml"},
    "XGBoost_Tuned": {"path": BASE_DIR / "models/xgboost_tuned_model.pkl", "type": "ml"},
    "LSTM": {"path": BASE_DIR / "models/lstm_model.keras", "type": "dl"},
    "GRU": {"path": BASE_DIR / "models/gru_model.keras", "type": "dl"},
    "Transformer": {"path": BASE_DIR / "models/transformer_model.keras", "type": "dl"},
    "Voting": {"path": BASE_DIR / "models/voting_model.pkl", "type": "ensemble"},
    "Stacking": {"path": BASE_DIR / "models/stacking_model.pkl", "type": "ensemble"},
}