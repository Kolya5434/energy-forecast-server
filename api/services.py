import pandas as pd
import numpy as np
import joblib
import json
import time
import tensorflow as tf
import shap
from typing import List, Dict, Any
from pathlib import Path
from prophet.serialize import model_from_json

from .schemas import PredictionRequest, PredictionResponse
from .config import AVAILABLE_MODELS

BASE_DIR = Path(__file__).resolve().parent.parent


def get_models_service() -> Dict[str, Any]:
    """Повертає інформацію про доступні моделі."""
    return {mid: {"description": m.get("description", ""), "type": m["type"]} for mid, m in AVAILABLE_MODELS.items()}


def _generate_future_features(last_date_str: str, horizon: int) -> pd.DataFrame:
    """Генерує DataFrame з часовими ознаками для майбутніх дат."""
    last_date = pd.to_datetime(last_date_str)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    X_future = pd.DataFrame(index=future_dates)
    X_future['day_of_week'] = X_future.index.dayofweek
    X_future['month'] = X_future.index.month
    X_future['day_of_year'] = X_future.index.dayofyear
    return X_future


def predict_service(request: PredictionRequest) -> List[PredictionResponse]:
    """Виконує прогнозування для списку моделей."""
    results = []
    last_known_date = "2010-11-26"  # Остання дата в повному датасеті

    for model_id in request.model_ids:
        if model_id not in AVAILABLE_MODELS:
            continue

        start_time = time.time()
        model_path = AVAILABLE_MODELS[model_id]["path"]
        model_type = AVAILABLE_MODELS[model_id]["type"]

        # --- Логіка завантаження для різних типів моделей ---
        if model_type in ["classical", "ml", "ensemble"]:
            if model_id == "Prophet":
                with open(model_path, 'r') as f:
                    model = model_from_json(json.load(f))
            else:
                model = joblib.load(model_path)
        elif model_type == "dl":
            model = tf.keras.models.load_model(model_path)

        forecast_dates = pd.date_range(start=pd.to_datetime(last_known_date) + pd.Timedelta(days=1),
                                       periods=request.forecast_horizon)
        preds = []

        # --- Логіка прогнозування ---
        if model_id in ["SARIMA", "ARIMA"]:
            preds = model.predict(n_periods=request.forecast_horizon)
        elif model_id == "Prophet":
            future_df = model.make_future_dataframe(periods=request.forecast_horizon, freq='D')
            forecast = model.predict(future_df)
            preds = forecast['yhat'].iloc[-request.forecast_horizon:].values
        elif model_type in ["ml", "ensemble"]:
            X_future = _generate_future_features(last_known_date, request.forecast_horizon)
            preds = model.predict(X_future)
        elif model_type == "dl":
            # Логіка для DL моделей складніша (потрібно готувати послідовності)
            # Тут ми реалізуємо заглушку, яка повертає випадкові дані
            # У реальному проєкті тут був би ітеративний "walk-forward" прогноз
            preds = np.random.rand(request.forecast_horizon) * 1000 + 1500

        forecast_values = {str(date.date()): float(pred) for date, pred in zip(forecast_dates, preds)}
        end_time = time.time()

        results.append(
            PredictionResponse(
                model_id=model_id,
                forecast=forecast_values,
                metadata={"latency_ms": round((end_time - start_time) * 1000, 2)}
            )
        )
    return results


def load_results_data() -> Dict[str, Any]:
    """Завантажує результати аналізу моделей з JSON файлу."""
    results_path = BASE_DIR / "data/model_results.json"
    with open(results_path, 'r') as f:
        return json.load(f)


def get_evaluation_service(model_id: str) -> Dict[str, Any]:
    """Повертає метрики якості для обраної моделі."""
    all_results = load_results_data()
    if model_id in all_results:
        return all_results.get(model_id, {})
    # Додамо логіку для моделей, яких може не бути в JSON
    return {"message": "Evaluation data is not available for this model."}


def get_interpretation_service(model_id: str) -> Dict[str, Any]:
    """Повертає дані для інтерпретації для обраної моделі."""
    if model_id != "XGBoost_Tuned":
        return {"message": "Interpretation is only available for 'XGBoost_Tuned' model."}

    # Завантажуємо модель та дані для SHAP
    model = joblib.load(AVAILABLE_MODELS[model_id]["path"])
    X_future = _generate_future_features("2010-11-26", 1)  # Беремо один день для прикладу

    # Розраховуємо SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_future)

    # Повертаємо пояснення для першого прогнозу
    explanation = {
        "base_value": explainer.expected_value,
        "prediction_value": sum(shap_values[0]) + explainer.expected_value,
        "feature_contributions": {feature: value for feature, value in zip(X_future.columns, shap_values[0])}
    }
    return explanation