from fastapi import FastAPI
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Define the project's base directory
BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: runs on startup and shutdown"""
    print("🚀 Запуск сервера Energy Forecast API...")

    try:
        # 1. Завантажуємо моделі з Hugging Face
        from .utils import download_models_from_hf
        download_models_from_hf()

        # 2. Завантажуємо дані та моделі в пам'ять
        from . import services
        services.initialize_services()

        print("✅ Всі компоненти готові до роботи!")

    except Exception as e:
        print(f"❌ ПОМИЛКА при ініціалізації: {e}")
        print("   Сервер запуститься, але API може не працювати")

    yield

    print("🛑 Зупинка сервера...")


app = FastAPI(
    title="Intelligent Hybrid Energy Consumption Forecasting System",
    description="Інтелектуальна гібридна система прогнозування споживання енергії для «розумних» міст.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS to allow requests from your frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://eneryge-forecast.vercel.app",
    "https://mykola121-energy-forecast-api.hf.space",
    "https://*.hf.space",
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
        "feature_set": "none",
        "description": "Авторегресійна інтегрована модель ковзного середнього",
        "supports_conditions": False,
        "supports_simulation": False,
    },
    "SARIMA": {
        "path": BASE_DIR / "models/sarima_baseline_model.pkl",
        "type": "classical",
        "granularity": "daily",
        "feature_set": "none",
        "description": "Сезонна ARIMA модель для врахування періодичності",
        "supports_conditions": False,
        "supports_simulation": False,
    },
    "Prophet": {
        "path": BASE_DIR / "models/prophet_baseline_model.json",
        "type": "classical",
        "granularity": "daily",
        "feature_set": "none",
        "description": "Facebook Prophet для прогнозування часових рядів зі святами",
        "supports_conditions": False,
        "supports_simulation": False,
    },

    # --- ML models (hourly data, full feature set) ---
    "RandomForest": {
        "path": BASE_DIR / "models/random_forest_model.pkl",
        "type": "ml",
        "granularity": "hourly",
        "feature_set": "full",
        "description": "Ансамбль дерев рішень з повним набором ознак",
        "supports_conditions": True,
        "supports_simulation": True,
    },
    "XGBoost": {
        "path": BASE_DIR / "models/xgboost_model.pkl",
        "type": "ml",
        "granularity": "hourly",
        "feature_set": "full",
        "description": "Градієнтний бустинг XGBoost (погодинна модель)",
        "supports_conditions": True,
        "supports_simulation": True,
    },
    "LightGBM": {
        "path": BASE_DIR / "models/light_gbm_model.pkl",
        "type": "ml",
        "granularity": "hourly",
        "feature_set": "full",
        "description": "Швидкий градієнтний бустинг Microsoft LightGBM",
        "supports_conditions": True,
        "supports_simulation": True,
    },

    # --- ML models & Ensembles (daily data, simple feature set) ---
    "XGBoost_Tuned": {
        "path": BASE_DIR / "models/xgboost_tuned_model.pkl",
        "type": "ml",
        "granularity": "daily",
        "feature_set": "simple",
        "description": "Оптимізована XGBoost модель для денних прогнозів",
        "supports_conditions": True,
        "supports_simulation": True,
    },
    "Voting": {
        "path": BASE_DIR / "models/voting_model.pkl",
        "type": "ensemble",
        "granularity": "daily",
        "feature_set": "simple",
        "description": "Ансамбль з голосуванням кількох ML моделей",
        "supports_conditions": True,
        "supports_simulation": True,
    },
    "Stacking": {
        "path": BASE_DIR / "models/stacking_model.pkl",
        "type": "ensemble",
        "granularity": "daily",
        "feature_set": "simple",
        "description": "Стекінг-ансамбль з мета-навчанням",
        "supports_conditions": True,
        "supports_simulation": True,
    },

    # --- DL models (hourly data, base scaled features, sequential) ---
    "LSTM": {
        "path": BASE_DIR / "models/lstm_model.keras",
        "type": "dl",
        "granularity": "hourly",
        "feature_set": "base_scaled",
        "is_sequential": True,
        "sequence_length": 24,
        "description": "Рекурентна нейромережа з довгою короткочасною пам'яттю",
        "supports_conditions": False,
        "supports_simulation": False,
    },
    "GRU": {
        "path": BASE_DIR / "models/gru_model.keras",
        "type": "dl",
        "granularity": "hourly",
        "feature_set": "base_scaled",
        "is_sequential": True,
        "sequence_length": 24,
        "description": "Рекурентна нейромережа з керованими рекурентними блоками",
        "supports_conditions": False,
        "supports_simulation": False,
    },
    "Transformer": {
        "path": BASE_DIR / "models/transformer_model.keras",
        "type": "dl",
        "granularity": "hourly",
        "feature_set": "base_scaled",
        "is_sequential": True,
        "sequence_length": 24,
        "description": "Трансформер з механізмом уваги для часових рядів",
        "supports_conditions": False,
        "supports_simulation": False,
    },
}