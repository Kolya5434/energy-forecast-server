import pandas as pd
import joblib
import json
import time
import tensorflow as tf
from typing import List, Dict, Any
from prophet.serialize import model_from_json
from fastapi import HTTPException

from .schemas import PredictionRequest
from .config import AVAILABLE_MODELS, BASE_DIR
from .features import generate_simple_features, generate_full_features

try:
    print("Loading historical data...")
    HISTORICAL_DATA_HOURLY = pd.read_csv(
        BASE_DIR / "data/dataset_for_modeling.csv",
        index_col='DateTime',
        parse_dates=True
    )
    HISTORICAL_DATA_DAILY = HISTORICAL_DATA_HOURLY['Global_active_power'].resample('D').sum().to_frame()
    print("Historical data loaded.")

    print("Loading models...")
    MODELS_CACHE = {}
    for model_id, config in AVAILABLE_MODELS.items():
        try:
            if config["type"] == "dl":
                MODELS_CACHE[model_id] = tf.keras.models.load_model(config["path"])
            elif model_id == "Prophet":
                with open(config["path"], 'r') as f:
                    MODELS_CACHE[model_id] = model_from_json(json.load(f))
            else:
                MODELS_CACHE[model_id] = joblib.load(config["path"])
            print(f" - Loaded model: {model_id}")
        except Exception as load_err:
            print(f"   - FAILED to load model {model_id}: {load_err}")
    print("Model loading complete.")

except Exception as startup_err:
    print(f"CRITICAL ERROR during server startup: {startup_err}")
    HISTORICAL_DATA_HOURLY = None
    HISTORICAL_DATA_DAILY = None
    MODELS_CACHE = {}


# --- Кінець завантаження ---

def get_models_service() -> Dict[str, Any]:
    # ... (код без змін)
    return {mid: {"type": m["type"], "granularity": m["granularity"], "feature_set": m["feature_set"]}
            for mid, m in AVAILABLE_MODELS.items()}


def predict_service(request: PredictionRequest) -> List[Dict[str, Any]]:
    if not MODELS_CACHE or HISTORICAL_DATA_HOURLY is None or HISTORICAL_DATA_DAILY is None:
        raise HTTPException(status_code=503, detail="Сервіс недоступний: Моделі або історичні дані не завантажено.")

    results = []
    last_known_date_hourly = HISTORICAL_DATA_HOURLY.index.max()
    last_known_date_daily = HISTORICAL_DATA_DAILY.index.max()

    for model_id in request.model_ids:
        if model_id not in MODELS_CACHE:
            print(f"Warning: Model {model_id} requested but not loaded. Skipping.")
            continue

        start_time = time.time()
        model_config = AVAILABLE_MODELS[model_id]
        model = MODELS_CACHE[model_id]

        forecast_values: Dict[str, Any] = {}
        preds = []
        final_dates = []

        try:
            if model_config["granularity"] == "daily":
                future_dates = pd.date_range(start=last_known_date_daily + pd.Timedelta(days=1),
                                             periods=request.forecast_horizon)

                if model_config["feature_set"] == "simple":
                    X_future = generate_simple_features(future_dates)
                    if hasattr(model, 'feature_names_in_'):
                        X_future = X_future[model.feature_names_in_]

                    if X_future.isnull().values.any():
                        print(f"Warning: NaNs detected in features for {model_id}. Attempting to fill.")
                        X_future = X_future.ffill().bfill()  # Пробуємо заповнити
                        if X_future.isnull().values.any():
                            raise ValueError(f"Could not resolve NaNs in features for {model_id}")

                    preds = model.predict(X_future)

                elif model_config["feature_set"] == "none":
                    if model_id == "Prophet":
                        future_df = pd.DataFrame({'ds': future_dates})
                        forecast = model.predict(future_df)
                        preds = forecast['yhat'].values
                    else:  # ARIMA/SARIMA
                        preds = model.predict(n_periods=request.forecast_horizon)

                final_preds, final_dates = preds, future_dates

            elif model_config["granularity"] == "hourly":
                num_hours = request.forecast_horizon * 24
                future_dates_hourly = pd.date_range(start=last_known_date_hourly + pd.Timedelta(hours=1),
                                                    periods=num_hours, freq='h')

                if model_config.get("is_sequential"):
                    # ... (Логіка для DL моделей, потребує тестування) ...
                    sequence_len = model_config["sequence_length"]
                    history_buffer_len = sequence_len + 168
                    current_history = HISTORICAL_DATA_HOURLY.iloc[-history_buffer_len:].copy()
                    # ... (решта логіки walk-forward) ...
                    preds = [0] * num_hours  # Заглушка

                elif model_config["feature_set"] == "full":
                    history_slice = HISTORICAL_DATA_HOURLY.tail(168)
                    X_future = generate_full_features(history_slice, future_dates_hourly)

                    if hasattr(model, 'feature_names_in_'):
                        expected_cols = model.feature_names_in_
                        missing_in_future = set(expected_cols) - set(X_future.columns)
                        if missing_in_future:
                            raise ValueError(f"Feature generation missed expected columns: {missing_in_future}")
                        extra_in_future = set(X_future.columns) - set(expected_cols)
                        if extra_in_future:
                            X_future = X_future.drop(columns=list(extra_in_future))
                        X_future = X_future[expected_cols]

                    if X_future.isnull().values.any():
                        print(f"Warning: NaNs detected in features for {model_id}. Attempting to fill.")
                        X_future = X_future.ffill().bfill()
                        if X_future.isnull().values.any():
                            raise ValueError(f"Could not resolve NaNs in features for {model_id}")

                    preds = model.predict(X_future)

                if preds:
                    hourly_preds = pd.Series(preds, index=future_dates_hourly)
                    daily_preds = hourly_preds.resample('D').sum()
                    final_preds = daily_preds.values
                    final_dates = daily_preds.index
                else:  # Якщо прогноз не вдалося зробити
                    final_preds = []
                    final_dates = pd.date_range(start=last_known_date_daily + pd.Timedelta(days=1),
                                                periods=request.forecast_horizon)

            forecast_values = {str(date.date()): float(pred) for date, pred in zip(final_dates, final_preds)}

        except Exception as pred_err:
            error_message = f"Помилка прогнозування для моделі {model_id}: {pred_err}"
            print(error_message)
            forecast_values = {}  # Залишаємо forecast порожнім
            metadata = {"error": error_message}

        end_time = time.time()

        response_item = {
            "model_id": model_id,
            "forecast": forecast_values,
            "metadata": {"latency_ms": round((end_time - start_time) * 1000, 2)}
        }
        if "error" in locals().get('metadata', {}):
            response_item["metadata"]["error"] = metadata["error"]

        results.append(response_item)

    return results


def load_results_data() -> Dict[str, Any]:
    """Завантажує результати аналізу моделей з JSON файлу."""
    results_path = BASE_DIR / "data/model_results.json"
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Results file not found at {results_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from {results_path}")
        return {}


def get_evaluation_service(model_id: str) -> Dict[str, Any]:
    """Повертає метрики якості для обраної моделі."""
    all_results = load_results_data()
    result = all_results.get(model_id)
    if result:
        return result
    else:
        return {"error": f"Evaluation data not available for model '{model_id}'."}


def get_interpretation_service(model_id: str) -> Dict[str, Any]:
    """Повертає дані для інтерпретації для обраної моделі."""
    all_results = load_results_data()

    if model_id in all_results and "interpretation" in all_results[model_id] and all_results[model_id][
        "interpretation"]:
        return all_results[model_id]["interpretation"]

    if model_id == "XGBoost_Tuned" and model_id in MODELS_CACHE:
        try:
            model = MODELS_CACHE[model_id]
            last_known_date = HISTORICAL_DATA_DAILY.index.max()
            future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=1)
            X_future = generate_simple_features(future_dates)
            if hasattr(model, 'feature_names_in_'):
                X_future = X_future[model.feature_names_in_]

            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_future)

            explanation = {
                "shap_values": {
                    "base_value": float(explainer.expected_value),
                    "prediction_value": float(sum(shap_values[0]) + explainer.expected_value),
                    "feature_contributions": {
                        feature: float(value) for feature, value in zip(X_future.columns, shap_values[0])
                    }
                }
            }
            return explanation
        except Exception as shap_err:
            print(f"Error calculating SHAP for {model_id}: {shap_err}")
            return {"error": f"Could not calculate SHAP values: {shap_err}"}

    return {"error": f"Interpretation data not available for model '{model_id}'."}