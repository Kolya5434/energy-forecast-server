import numpy as np
import pandas as pd
import joblib
import json
import time
import tensorflow as tf
from typing import List, Dict, Any, Tuple
from prophet.serialize import model_from_json
from fastapi import HTTPException
import sklearn.ensemble
import functools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta

from .config import AVAILABLE_MODELS, BASE_DIR
from .schemas import (
    PredictionRequest, SimulationRequest, WeatherConditions,
    CalendarConditions, TimeScenario, EnergyConditions, ZoneConsumption
)
from .features import generate_simple_features, generate_full_features, create_sequences_for_dl
from .evaluation import evaluate_model


logger = logging.getLogger(__name__)


def _apply_weather_conditions(X_future: pd.DataFrame, weather: WeatherConditions) -> pd.DataFrame:
    """
    Застосовує погодні умови до DataFrame з ознаками.
    Повертає оновлений DataFrame.
    """
    weather_mapping = {
        'temperature': weather.temperature,
        'humidity': weather.humidity,
        'wind_speed': weather.wind_speed,
    }

    applied_features = []
    for feature_name, value in weather_mapping.items():
        if value is not None and feature_name in X_future.columns:
            X_future[feature_name] = value
            applied_features.append(f"{feature_name}={value}")

    if applied_features:
        logger.info(f"Applied weather conditions: {', '.join(applied_features)}")

    return X_future


def _apply_calendar_conditions(X_future: pd.DataFrame, calendar: CalendarConditions) -> pd.DataFrame:
    """
    Застосовує календарні умови до DataFrame з ознаками.
    Повертає оновлений DataFrame.
    """
    calendar_mapping = {
        'is_holiday': 1 if calendar.is_holiday else 0 if calendar.is_holiday is not None else None,
        'is_weekend': 1 if calendar.is_weekend else 0 if calendar.is_weekend is not None else None,
    }

    applied_features = []
    for feature_name, value in calendar_mapping.items():
        if value is not None and feature_name in X_future.columns:
            X_future[feature_name] = value
            applied_features.append(f"{feature_name}={value}")

    if applied_features:
        logger.info(f"Applied calendar conditions: {', '.join(applied_features)}")

    return X_future


def _apply_time_scenario(X_future: pd.DataFrame, time_scenario: TimeScenario) -> pd.DataFrame:
    """
    Застосовує часові сценарії до DataFrame з ознаками.
    Дозволяє моделювати пікові години та сезонність.
    """
    time_mapping = {
        'hour': time_scenario.hour,
        'day_of_week': time_scenario.day_of_week,
        'day_of_month': time_scenario.day_of_month,
        'day_of_year': time_scenario.day_of_year,
        'week_of_year': time_scenario.week_of_year,
        'month': time_scenario.month,
        'year': time_scenario.year,
        'quarter': time_scenario.quarter,
    }

    applied_features = []
    for feature_name, value in time_mapping.items():
        if value is not None and feature_name in X_future.columns:
            X_future[feature_name] = value
            applied_features.append(f"{feature_name}={value}")

    if applied_features:
        logger.info(f"Applied time scenario: {', '.join(applied_features)}")

    return X_future


def _apply_zone_consumption(X_future: pd.DataFrame, zone: ZoneConsumption) -> pd.DataFrame:
    """
    Застосовує зонове споживання до DataFrame з ознаками.
    Дозволяє моделювати сценарії по категоріях приладів.
    """
    zone_mapping = {
        'Sub_metering_1': zone.sub_metering_1,
        'Sub_metering_2': zone.sub_metering_2,
        'Sub_metering_3': zone.sub_metering_3,
    }

    applied_features = []
    for feature_name, value in zone_mapping.items():
        if value is not None and feature_name in X_future.columns:
            X_future[feature_name] = value
            applied_features.append(f"{feature_name}={value}")

    if applied_features:
        logger.info(f"Applied zone consumption: {', '.join(applied_features)}")

    return X_future


def _apply_anomaly_flag(X_future: pd.DataFrame, is_anomaly: bool) -> pd.DataFrame:
    """
    Застосовує прапорець аномалії до DataFrame з ознаками.
    """
    if 'is_anomaly' in X_future.columns:
        X_future['is_anomaly'] = 1 if is_anomaly else 0
        logger.info(f"Applied anomaly flag: is_anomaly={1 if is_anomaly else 0}")

    return X_future


def _apply_energy_conditions(X_future: pd.DataFrame, energy: EnergyConditions) -> pd.DataFrame:
    """
    Застосовує енергетичні параметри мережі до DataFrame з ознаками.
    Дозволяє моделювати what-if сценарії для напруги та струму.
    """
    energy_mapping = {
        'Voltage': energy.voltage,
        'Global_reactive_power': energy.global_reactive_power,
        'Global_intensity': energy.global_intensity,
    }

    applied_features = []
    for feature_name, value in energy_mapping.items():
        if value is not None and feature_name in X_future.columns:
            X_future[feature_name] = value
            applied_features.append(f"{feature_name}={value}")

    if applied_features:
        logger.info(f"Applied energy conditions: {', '.join(applied_features)}")

    return X_future


def _apply_all_conditions(X_future: pd.DataFrame, request) -> pd.DataFrame:
    """
    Застосовує всі умови з request до DataFrame з ознаками.
    Використовується як в predict_service, так і в simulate_service.
    """
    if request is None:
        return X_future

    if hasattr(request, 'weather') and request.weather:
        X_future = _apply_weather_conditions(X_future, request.weather)

    if hasattr(request, 'calendar') and request.calendar:
        X_future = _apply_calendar_conditions(X_future, request.calendar)

    if hasattr(request, 'time_scenario') and request.time_scenario:
        X_future = _apply_time_scenario(X_future, request.time_scenario)

    if hasattr(request, 'energy') and request.energy:
        X_future = _apply_energy_conditions(X_future, request.energy)

    if hasattr(request, 'zone_consumption') and request.zone_consumption:
        X_future = _apply_zone_consumption(X_future, request.zone_consumption)

    if hasattr(request, 'is_anomaly') and request.is_anomaly is not None:
        X_future = _apply_anomaly_flag(X_future, request.is_anomaly)

    return X_future


# Response cache with TTL (Time To Live)
_response_cache: Dict[Tuple, Tuple[List[Dict[str, Any]], datetime]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes cache

MODELS_CACHE: Dict[str, Any] = {}
HISTORICAL_DATA_HOURLY = None
HISTORICAL_DATA_DAILY = None
_scaler = None


def initialize_services():
    global MODELS_CACHE, HISTORICAL_DATA_HOURLY, HISTORICAL_DATA_DAILY, _scaler

    try:
        print("Loading historical data...")
        _hourly_data_path = BASE_DIR / "data/dataset_for_modeling.csv"
        if not _hourly_data_path.exists():
            raise FileNotFoundError(f"Historical data file not found at {_hourly_data_path}")

        HISTORICAL_DATA_HOURLY = pd.read_csv(
            _hourly_data_path,
            index_col='DateTime',
            parse_dates=True
        )
        HISTORICAL_DATA_DAILY = HISTORICAL_DATA_HOURLY['Global_active_power'].resample('D').sum().to_frame()
        print("✅ Historical data loaded successfully.")

        print("Loading scaler...")
        _scaler_path = BASE_DIR / "models/standard_scaler.pkl"
        if _scaler_path.exists():
            _scaler = joblib.load(_scaler_path)
            print("✅ Scaler loaded.")
        else:
            print("⚠️  WARNING: Scaler file not found. DL models might fail.")

        print("Loading models into cache...")
        for model_id, config in AVAILABLE_MODELS.items():
            try:
                model_path = config["path"]
                if not model_path.exists():
                    print(f"   ⚠️  Model file not found for {model_id} at {model_path}. Skipping.")
                    continue

                if config["type"] == "dl":
                    if 'tf' not in globals() or tf is None:
                        print(f"   ⚠️  TensorFlow not available. Skipping DL model: {model_id}")
                        continue
                    MODELS_CACHE[model_id] = tf.keras.models.load_model(model_path)
                elif model_id == "Prophet":
                    if 'model_from_json' not in globals() or model_from_json is None:
                        print(f"   ⚠️  Prophet library function not available. Skipping Prophet model.")
                        continue
                    with open(model_path, 'r') as f:
                        MODELS_CACHE[model_id] = model_from_json(json.load(f))
                else:
                    MODELS_CACHE[model_id] = joblib.load(model_path)
                print(f"   ✅ Loaded model: {model_id}")
            except ImportError as ie:
                print(f"   ❌ FAILED to load model {model_id} due to missing library: {ie}")
            except Exception as load_err:
                print(f"   ❌ FAILED to load model {model_id}: {load_err}")

        print(f"✅ Model loading complete. {len(MODELS_CACHE)} models loaded successfully.")

    except FileNotFoundError as fnf_err:
        print(f"❌ CRITICAL ERROR: Data file missing: {fnf_err}")
        raise
    except Exception as startup_err:
        print(f"❌ CRITICAL ERROR during services initialization: {startup_err}")
        raise




def get_models_service() -> Dict[str, Any]:
    """Повертає розширену інформацію про доступні (завантажені) моделі."""
    loaded_model_ids = list(MODELS_CACHE.keys())
    result = {}
    for mid in loaded_model_ids:
        if mid not in AVAILABLE_MODELS:
            continue
        config = AVAILABLE_MODELS[mid]
        static_metrics = STATIC_PERFORMANCE_METRICS.get(mid, {})
        result[mid] = {
            "type": config["type"],
            "granularity": config["granularity"],
            "feature_set": config["feature_set"],
            "description": config.get("description", ""),
            "supports_conditions": config.get("supports_conditions", False),
            "supports_simulation": config.get("supports_simulation", False),
            "avg_latency_ms": static_metrics.get("avg_latency_ms"),
            "memory_increment_mb": static_metrics.get("memory_increment_mb"),
        }
    return result


def _predict_single_model(
    model_id: str,
    forecast_horizon: int,
    last_known_date_hourly: pd.Timestamp,
    last_known_date_daily: pd.Timestamp,
    request: PredictionRequest = None
) -> Dict[str, Any]:
    """
    Executes prediction for a single model.
    This function is extracted for parallel execution via ThreadPoolExecutor.
    Supports optional condition parameters from request.
    """
    if model_id not in MODELS_CACHE:
        logger.warning(f"Model {model_id} requested but not loaded/available. Skipping.")
        return {
            "model_id": model_id,
            "forecast": {},
            "metadata": {"error": "Model not loaded or unavailable."}
        }

    start_time = time.time()
    model_config = AVAILABLE_MODELS[model_id]
    model = MODELS_CACHE[model_id]

    forecast_values: Dict[str, Any] = {}
    preds = []
    final_preds = []
    final_dates = pd.DatetimeIndex([])
    metadata = {}

    try:
        if model_config["granularity"] == "daily":
            future_dates = pd.date_range(start=last_known_date_daily + pd.Timedelta(days=1),
                                         periods=forecast_horizon, freq='D')

            if model_config["feature_set"] == "simple":
                X_future = generate_simple_features(future_dates)
                # Застосування умов з request
                X_future = _apply_all_conditions(X_future, request)
                if hasattr(model, 'feature_names_in_'):
                    X_future = X_future[model.feature_names_in_]
                if X_future.isnull().values.any():
                    X_future = X_future.ffill().bfill()
                if X_future.isnull().values.any():
                    raise ValueError("NaNs in features after fill")
                preds = model.predict(X_future)

            elif model_config["feature_set"] == "none":
                if model_id == "Prophet":
                    future_df = pd.DataFrame({'ds': future_dates})
                    forecast = model.predict(future_df)
                    preds = forecast['yhat'].values
                else:
                    preds = model.predict(n_periods=forecast_horizon)

            final_preds, final_dates = preds, future_dates

        elif model_config["granularity"] == "hourly":
            num_hours = forecast_horizon * 24
            future_dates_hourly = pd.date_range(start=last_known_date_hourly + pd.Timedelta(hours=1),
                                                periods=num_hours, freq='h')
            hourly_predictions = []

            if model_config.get("is_sequential"):  # DL Models
                # Walk-Forward прогнозування для DL моделей
                sequence_len = model_config["sequence_length"]
                if _scaler is None:
                    raise ValueError("Scaler not loaded. Cannot predict with DL model.")

                scaler_features = list(_scaler.get_feature_names_out())
                target_col_name = 'Global_active_power'

                try:
                    target_index_in_scaler = scaler_features.index(target_col_name)
                except ValueError:
                    raise ValueError(f"Target column '{target_col_name}' not found in scaler features.")

                # Перевірка наявності необхідних колонок
                missing_cols = set(scaler_features) - set(HISTORICAL_DATA_HOURLY.columns)
                if missing_cols:
                    raise ValueError(f"Historical data missing required columns: {missing_cols}")

                current_history = HISTORICAL_DATA_HOURLY[scaler_features].iloc[-sequence_len:].copy()

                if current_history.isnull().values.any():
                    current_history = current_history.ffill().bfill().fillna(0)

                logger.info(f"Starting walk-forward prediction for {model_id} ({num_hours} hours)...")

                for i in range(num_hours):
                    input_scaled = _scaler.transform(current_history)
                    X_input = np.delete(input_scaled, target_index_in_scaler, axis=1)
                    X_input_reshaped = np.expand_dims(X_input, axis=0)

                    pred_scaled = model.predict(X_input_reshaped, verbose=0)[0][0]

                    if np.isnan(pred_scaled):
                        pred_scaled = input_scaled[-1, target_index_in_scaler]

                    dummy_row_scaled = input_scaled[-1].copy()
                    dummy_row_scaled[target_index_in_scaler] = pred_scaled
                    inversed_row = _scaler.inverse_transform(dummy_row_scaled.reshape(1, -1))[0]
                    inversed_pred_value = inversed_row[target_index_in_scaler]

                    # Обробка некоректних значень
                    if np.isnan(inversed_pred_value) or inversed_pred_value < 0:
                        inversed_pred_value = 0.0

                    hourly_predictions.append(float(inversed_pred_value))

                    next_timestamp = current_history.index[-1] + pd.Timedelta(hours=1)
                    new_row_df = pd.DataFrame([inversed_row], columns=scaler_features, index=[next_timestamp])
                    current_history = pd.concat([current_history.iloc[1:], new_row_df])

                    if (i + 1) % 48 == 0 or i == 0:
                        logger.info(f"  Progress ({model_id}): {i + 1}/{num_hours} hours")

            elif model_config["feature_set"] == "full":  # ML Models
                history_slice = HISTORICAL_DATA_HOURLY.tail(168)
                X_future = generate_full_features(history_slice, future_dates_hourly)
                # Застосування умов з request
                X_future = _apply_all_conditions(X_future, request)

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
                    logger.warning(f"NaNs detected in features for {model_id}. Filling with 0.")
                    X_future = X_future.fillna(0)
                    if X_future.isnull().values.any():
                        raise ValueError(f"NaNs persist after fillna for {model_id}")

                hourly_predictions = model.predict(X_future)

            # --- Агрегація годинних до денних ---
            if len(hourly_predictions) > 0:
                hourly_preds_series = pd.Series(hourly_predictions, index=future_dates_hourly)
                daily_preds = hourly_preds_series.resample('D').sum()
                final_preds = daily_preds.values
                final_dates = daily_preds.index
            else:
                final_preds = []
                final_dates = pd.date_range(start=last_known_date_daily + pd.Timedelta(days=1),
                                            periods=forecast_horizon, freq='D')

        forecast_values = {str(date.date()): float(pred) for date, pred in zip(final_dates, final_preds)}

    except Exception as pred_err:
        error_message = f"Помилка прогнозування для моделі {model_id}: {pred_err}"
        logger.error(error_message)
        forecast_values = {}
        metadata["error"] = error_message

    end_time = time.time()
    metadata["latency_ms"] = round((end_time - start_time) * 1000, 2)

    # Додаємо інформацію про застосовані умови
    if request:
        conditions_applied = {}
        if request.weather:
            conditions_applied["weather"] = request.weather.model_dump(exclude_none=True)
        if request.calendar:
            conditions_applied["calendar"] = request.calendar.model_dump(exclude_none=True)
        if request.time_scenario:
            conditions_applied["time_scenario"] = request.time_scenario.model_dump(exclude_none=True)
        if request.energy:
            conditions_applied["energy"] = request.energy.model_dump(exclude_none=True)
        if request.zone_consumption:
            conditions_applied["zone_consumption"] = request.zone_consumption.model_dump(exclude_none=True)
        if request.is_anomaly is not None:
            conditions_applied["is_anomaly"] = request.is_anomaly
        if conditions_applied:
            metadata["conditions_applied"] = conditions_applied

    response_item = {
        "model_id": model_id,
        "forecast": forecast_values,
        "metadata": metadata
    }
    return response_item


def predict_service(request: PredictionRequest) -> List[Dict[str, Any]]:
    """
    Виконує прогнозування для списку моделей, враховуючи їхні вимоги.

    Uses parallel execution with ThreadPoolExecutor for faster predictions.
    Includes response caching with TTL for identical requests.
    """
    if not MODELS_CACHE or HISTORICAL_DATA_HOURLY is None or HISTORICAL_DATA_DAILY is None:
        raise HTTPException(status_code=503, detail="Сервіс недоступний: Моделі або історичні дані не завантажено.")

    last_known_date_hourly = HISTORICAL_DATA_HOURLY.index.max()
    last_known_date_daily = HISTORICAL_DATA_DAILY.index.max()

    # Create cache key from request parameters and current date
    # Include conditions in cache key to differentiate requests with different parameters
    conditions_key = (
        str(request.weather.model_dump() if request.weather else None),
        str(request.calendar.model_dump() if request.calendar else None),
        str(request.time_scenario.model_dump() if request.time_scenario else None),
        str(request.energy.model_dump() if request.energy else None),
        str(request.zone_consumption.model_dump() if request.zone_consumption else None),
        request.is_anomaly
    )
    cache_key = (
        tuple(sorted(request.model_ids)),
        request.forecast_horizon,
        str(last_known_date_daily.date()),
        conditions_key
    )

    # Check cache
    current_time = datetime.now()
    if cache_key in _response_cache:
        cached_result, cache_time = _response_cache[cache_key]
        age_seconds = (current_time - cache_time).total_seconds()

        if age_seconds < CACHE_TTL_SECONDS:
            logger.info(f"⚡ Cache HIT! Returning cached result (age: {age_seconds:.1f}s)")
            return cached_result
        else:
            # Cache expired, remove it
            logger.info(f"Cache expired (age: {age_seconds:.1f}s), recalculating...")
            del _response_cache[cache_key]

    # Determine optimal number of workers based on CPU cores
    # Use min(cpu_count, num_models) to avoid creating unnecessary threads
    max_workers = min(os.cpu_count() or 4, len(request.model_ids))

    logger.info(f"Starting parallel predictions for {len(request.model_ids)} models with {max_workers} workers")

    results = []

    # Execute predictions in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all prediction tasks
        future_to_model = {
            executor.submit(
                _predict_single_model,
                model_id,
                request.forecast_horizon,
                last_known_date_hourly,
                last_known_date_daily,
                request  # Pass request for condition parameters
            ): model_id
            for model_id in request.model_ids
        }

        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed prediction for {model_id} - Latency: {result['metadata'].get('latency_ms', 'N/A')}ms")
            except Exception as exc:
                logger.error(f"Model {model_id} generated an exception during parallel execution: {exc}")
                results.append({
                    "model_id": model_id,
                    "forecast": {},
                    "metadata": {"error": f"Parallel execution error: {exc}"}
                })

    # Store result in cache
    _response_cache[cache_key] = (results, current_time)
    logger.info(f"💾 Stored result in cache (key: {len(_response_cache)} entries)")

    # Cleanup old cache entries (keep max 100 entries)
    if len(_response_cache) > 100:
        oldest_keys = sorted(_response_cache.keys(), key=lambda k: _response_cache[k][1])[:50]
        for old_key in oldest_keys:
            del _response_cache[old_key]
        logger.info(f"🧹 Cleaned up cache, removed {len(oldest_keys)} old entries")

    return results


STATIC_PERFORMANCE_METRICS = {
    "ARIMA": {"avg_latency_ms": 0.975800, "memory_increment_mb": 37.78},
    "SARIMA": {"avg_latency_ms": 1.574268, "memory_increment_mb": 12.53},
    "XGBoost_Tuned": {"avg_latency_ms": 0.776591, "memory_increment_mb": 8.03},
    "LSTM": {"avg_latency_ms": 33.313401, "memory_increment_mb": 57.78},
    "XGBoost": {"avg_latency_ms": 1.981492, "memory_increment_mb": 8.05},
    "RandomForest": {"avg_latency_ms": 15.702372, "memory_increment_mb": 57.44},
    "LightGBM": {"avg_latency_ms": 1.118460, "memory_increment_mb": 8.05},
    "GRU": {"avg_latency_ms": 30.404129, "memory_increment_mb": None},
    "Transformer": {"avg_latency_ms": 35.484061, "memory_increment_mb": 59.84},
    "Voting": {"avg_latency_ms": 3.622100, "memory_increment_mb": 8.05},
    "Stacking": {"avg_latency_ms": 3.477340, "memory_increment_mb": 8.06},
    "Prophet": {"avg_latency_ms": 17.804852, "memory_increment_mb": 58.06},
}


# Caching for evaluation - stores up to 32 results
@functools.lru_cache(maxsize=32)
def _cached_evaluate_model(model_id: str) -> Dict[str, Any]:
    """
   Internal function with caching for model evaluation.
   Accepts only hashable arguments (model_id).
    """
    try:
        evaluation_results = evaluate_model(
            model_id=model_id,
            historical_data_daily=HISTORICAL_DATA_DAILY,
            historical_data_hourly=HISTORICAL_DATA_HOURLY,
            models_cache=MODELS_CACHE,
            available_models=AVAILABLE_MODELS
        )

        static_metrics = STATIC_PERFORMANCE_METRICS.get(model_id, {
            "avg_latency_ms": None,
            "memory_increment_mb": None
        })
        evaluation_results["performance_metrics"] = static_metrics

        return evaluation_results

    except (KeyError, NotImplementedError, ValueError, FileNotFoundError) as e:
        print(f"Помилка при оцінці моделі {model_id}: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"Неочікувана помилка під час оцінки {model_id}: {e}")
        return {"error": f"Неочікувана помилка під час оцінки моделі '{model_id}'."}


def get_evaluation_service(model_id: str) -> Dict[str, Any]:
    """
    Service function for evaluating a cached model.
    Uses a cached function for fast access.
    """
    if model_id not in MODELS_CACHE:
        return {"error": f"Model '{model_id}' not loaded or unavailable."}
    if HISTORICAL_DATA_DAILY is None or HISTORICAL_DATA_HOURLY is None:
        return {"error": "Historical data not available for evaluation."}

    return _cached_evaluate_model(model_id)


@functools.lru_cache(maxsize=32)
def _cached_get_interpretation(model_id: str) -> Dict[str, Any]:
    """
    Internal caching function for model interpretation.
    """
    if model_id not in MODELS_CACHE:
        return {"error": f"Модель '{model_id}' не завантажена."}

    model_config = AVAILABLE_MODELS.get(model_id)
    if not model_config or model_config["type"] not in ["ml", "ensemble"]:
        return {"error": f"Інтерпретація недоступна для моделі типу '{model_config.get('type')}'."}

    model = MODELS_CACHE[model_id]
    response: Dict[str, Any] = {"model_id": model_id}

    # Feature Importance
    if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
        try:
            importances = model.feature_importances_
            feature_names = model.feature_names_in_
            importances_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
            response["feature_importance"] = importances_dict
        except Exception as fi_err:
            print(f"Не вдалося отримати feature_importances_ для {model_id}: {fi_err}")
            response["feature_importance"] = {"error": f"Не вдалося отримати feature importance: {fi_err}"}
    else:
        response["feature_importance"] = None

    # SHAP Values
    shap_explanation = None
    shap_error = None
    try:
        import shap

        is_tree_model = False
        if hasattr(model, '_Booster'):
            is_tree_model = True
        elif 'sklearn' in globals() and isinstance(model, (sklearn.ensemble.RandomForestRegressor,
                                                           sklearn.ensemble.GradientBoostingRegressor)):
            is_tree_model = True

        if not is_tree_model:
            raise TypeError("SHAP TreeExplainer підтримує лише деревні моделі.")

        explainer = shap.TreeExplainer(model)

        X_future = None
        if model_config["granularity"] == "daily" and model_config["feature_set"] == "simple":
            last_known_date = HISTORICAL_DATA_DAILY.index.max()
            future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=1, freq='D')
            X_future = generate_simple_features(future_dates)
        elif model_config["granularity"] == "hourly" and model_config["feature_set"] == "full":
            last_known_date = HISTORICAL_DATA_HOURLY.index.max()
            history_slice = HISTORICAL_DATA_HOURLY.tail(168)
            future_dates = pd.date_range(start=last_known_date + pd.Timedelta(hours=1), periods=1, freq='h')
            X_future = generate_full_features(history_slice, future_dates)

        if X_future is None:
            raise NotImplementedError(f"Підготовка даних SHAP для конфігурації моделі '{model_id}' не реалізована.")

        if hasattr(model, 'feature_names_in_'):
            X_future = X_future[model.feature_names_in_]

        if X_future.isnull().values.any():
            print(f"Warning: NaNs detected in features for SHAP ({model_id}). Filling with 0.")
            X_future = X_future.fillna(0)
            if X_future.isnull().values.any():
                raise ValueError("NaNs persist after fillna before SHAP")

        shap_values_output = explainer.shap_values(X_future.iloc[[0]])

        if isinstance(shap_values_output, list):
            if len(shap_values_output) > 0 and isinstance(shap_values_output[0], np.ndarray):
                shap_contributions = shap_values_output[0].flatten()
            else:
                raise ValueError("Неочікуваний формат списку від shap_values")
        elif isinstance(shap_values_output, np.ndarray):
            shap_contributions = shap_values_output.flatten()
        else:
            raise ValueError(f"Неочікуваний тип результату від shap_values: {type(shap_values_output)}")

        if len(shap_contributions) != len(X_future.columns):
            raise ValueError(
                f"Розмірність SHAP values ({len(shap_contributions)}) не збігається з кількістю ознак ({len(X_future.columns)}).")

        shap_explanation = {
            "base_value": float(explainer.expected_value),
            "prediction_value": float(explainer.expected_value + sum(shap_contributions)),
            "feature_contributions": {
                feature: float(value) for feature, value in zip(X_future.columns, shap_contributions)
            }
        }

    except ImportError:
        shap_error = "Бібліотеку 'shap' не встановлено."
        print(shap_error)
    except TypeError as te:
        shap_error = f"SHAP TreeExplainer не підходить: {te}"
        print(f"SHAP Error for {model_id}: {shap_error}")
    except NotImplementedError as nie:
        shap_error = str(nie)
        print(f"SHAP Error for {model_id}: {shap_error}")
    except Exception as e:
        shap_error = f"Не вдалося розрахувати SHAP values: {e}"
        print(f"Помилка при розрахунку SHAP для {model_id}: {e}")

    if shap_explanation:
        response["shap_values"] = shap_explanation
    else:
        response["shap_values"] = {"error": shap_error if shap_error else "Невідома помилка SHAP."}

    has_fi = isinstance(response.get("feature_importance"), dict) and "error" not in response.get("feature_importance",
                                                                                                  {})
    has_shap = isinstance(response.get("shap_values"), dict) and "error" not in response.get("shap_values", {})

    if not has_fi and not has_shap:
        final_error = response.get("shap_values", {}).get("error", "Не вдалося отримати жодних даних інтерпретації.")
        return {"error": final_error}

    return response


def get_interpretation_service(model_id: str) -> Dict[str, Any]:
    """
    Service function for interpreting the model with caching.
    """
    return _cached_get_interpretation(model_id)


def simulate_service(request: SimulationRequest) -> Dict[str, Any]:
    """
    Performs a forecast for ONE model, applying attribute changes.
    """
    model_id = request.model_id
    horizon = request.forecast_horizon
    overrides = request.feature_overrides

    if model_id not in MODELS_CACHE:
        raise KeyError(f"Модель '{model_id}' не завантажена або недоступна.")

    model_config = AVAILABLE_MODELS[model_id]
    model = MODELS_CACHE[model_id]

    if model_config["feature_set"] == "none":
        raise ValueError(
            f"Симуляція недоступна для моделі '{model_id}', оскільки вона не використовує зовнішні ознаки.")
    if model_config["type"] == "dl":
        raise NotImplementedError("Симуляція для DL моделей наразі не реалізована.")

    start_time = time.time()

    X_future = None
    future_dates = None

    if model_config["granularity"] == "daily":
        last_known_date = HISTORICAL_DATA_DAILY.index.max()
        future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        if model_config["feature_set"] == "simple":
            X_future = generate_simple_features(future_dates)

    elif model_config["granularity"] == "hourly":
        last_known_date = HISTORICAL_DATA_HOURLY.index.max()
        history_slice = HISTORICAL_DATA_HOURLY.tail(168)
        future_dates = pd.date_range(start=last_known_date + pd.Timedelta(hours=1), periods=horizon * 24, freq='h')
        if model_config["feature_set"] == "full":
            X_future = generate_full_features(history_slice, future_dates)

    if X_future is None:
        raise NotImplementedError(f"Генерація ознак для симуляції моделі '{model_id}' не реалізована.")

    # Застосування погодних умов (якщо передані)
    if request.weather:
        X_future = _apply_weather_conditions(X_future, request.weather)

    # Застосування календарних умов (якщо передані)
    if request.calendar:
        X_future = _apply_calendar_conditions(X_future, request.calendar)

    # Застосування часових сценаріїв (якщо передані)
    if request.time_scenario:
        X_future = _apply_time_scenario(X_future, request.time_scenario)

    # Застосування енергетичних параметрів (якщо передані)
    if request.energy:
        X_future = _apply_energy_conditions(X_future, request.energy)

    # Застосування зонового споживання (якщо передане)
    if request.zone_consumption:
        X_future = _apply_zone_consumption(X_future, request.zone_consumption)

    # Застосування прапорця аномалії (якщо передано)
    if request.is_anomaly is not None:
        X_future = _apply_anomaly_flag(X_future, request.is_anomaly)

    # Застосування індивідуальних змін (feature_overrides)
    logger.info(f"Applying {len(overrides)} feature overrides...")
    for override in overrides:
        try:
            date_to_change = pd.to_datetime(override.date)
            target_rows = X_future.index.date == date_to_change.date()

            if not np.any(target_rows):
                logger.warning(f"Дата {override.date} не знайдена в горизонті прогнозу. Зміна проігнорована.")
                continue

            for feature, value in override.features.items():
                if feature in X_future.columns:
                    X_future.loc[target_rows, feature] = value
                    print(f"   - Overridden '{feature}' on {override.date} to {value}")
                else:
                    print(f"Warning: Ознака '{feature}' не знайдена в моделі. Зміна проігнорована.")

        except Exception as e:
            print(f"Помилка при застосуванні зміни для дати {override.date}: {e}")

    if hasattr(model, 'feature_names_in_'):
        X_future = X_future[model.feature_names_in_]

    if X_future.isnull().values.any():
        print(f"Warning: NaNs detected in features for simulation. Filling with 0.")
        X_future = X_future.fillna(0)

    preds = model.predict(X_future)

    if model_config["granularity"] == "hourly":
        hourly_preds_series = pd.Series(preds, index=future_dates)
        daily_preds = hourly_preds_series.resample('D').sum()
        final_preds = daily_preds.values
        final_dates = daily_preds.index
    else:
        final_preds = preds
        final_dates = future_dates

    forecast_values = {str(date.date()): float(pred) for date, pred in zip(final_dates, final_preds)}
    end_time = time.time()

    metadata = {"latency_ms": round((end_time - start_time) * 1000, 2), "simulated": True}

    # Додаємо інформацію про застосовані умови
    conditions_applied = {}
    if request.weather:
        conditions_applied["weather"] = request.weather.model_dump(exclude_none=True)
    if request.calendar:
        conditions_applied["calendar"] = request.calendar.model_dump(exclude_none=True)
    if request.time_scenario:
        conditions_applied["time_scenario"] = request.time_scenario.model_dump(exclude_none=True)
    if request.energy:
        conditions_applied["energy"] = request.energy.model_dump(exclude_none=True)
    if request.zone_consumption:
        conditions_applied["zone_consumption"] = request.zone_consumption.model_dump(exclude_none=True)
    if request.is_anomaly is not None:
        conditions_applied["is_anomaly"] = request.is_anomaly
    if overrides:
        conditions_applied["feature_overrides"] = [
            {"date": o.date, "features": o.features} for o in overrides
        ]
    if conditions_applied:
        metadata["conditions_applied"] = conditions_applied

    return {
        "model_id": model_id,
        "forecast": forecast_values,
        "metadata": metadata
    }


def get_historical_service(
    days: int = 30,
    granularity: str = "daily",
    include_stats: bool = False
) -> Dict[str, Any]:
    """
    Повертає історичні дані споживання енергії.

    Args:
        days: Кількість днів історії (за замовчуванням 30)
        granularity: 'daily' або 'hourly'
        include_stats: Чи включати статистику (min, max, mean, std)
    """
    if HISTORICAL_DATA_HOURLY is None or HISTORICAL_DATA_DAILY is None:
        raise HTTPException(status_code=503, detail="Історичні дані не завантажено.")

    if granularity == "hourly":
        data = HISTORICAL_DATA_HOURLY['Global_active_power'].tail(days * 24)
        values = {str(idx): float(val) for idx, val in data.items()}
    else:  # daily
        data = HISTORICAL_DATA_DAILY['Global_active_power'].tail(days)
        values = {str(idx.date()): float(val) for idx, val in data.items()}

    result = {
        "granularity": granularity,
        "period_days": days,
        "data_points": len(values),
        "date_range": {
            "start": str(data.index.min()),
            "end": str(data.index.max())
        },
        "values": values
    }

    if include_stats:
        result["statistics"] = {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "median": float(data.median())
        }

    return result


def get_features_service(model_id: str) -> Dict[str, Any]:
    """
    Повертає інформацію про ознаки, які використовує конкретна модель.

    Args:
        model_id: ID моделі
    """
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Модель '{model_id}' не знайдена.")

    model_config = AVAILABLE_MODELS[model_id]
    model = MODELS_CACHE.get(model_id)

    feature_info = {
        "model_id": model_id,
        "type": model_config["type"],
        "granularity": model_config["granularity"],
        "feature_set": model_config["feature_set"],
        "supports_conditions": model_config.get("supports_conditions", False),
    }

    # Отримуємо список ознак з моделі
    if model and hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
        feature_info["feature_names"] = feature_names
        feature_info["feature_count"] = len(feature_names)

        # Групуємо ознаки за категоріями
        categories = {
            "weather": ["temperature", "humidity", "wind_speed"],
            "calendar": ["is_holiday", "is_weekend"],
            "time": ["hour", "day_of_week", "day_of_month", "day_of_year",
                     "week_of_year", "month", "year", "quarter"],
            "energy": ["Voltage", "Global_reactive_power", "Global_intensity"],
            "zone_consumption": ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
            "anomaly": ["is_anomaly"],
        }

        available_conditions = {}
        for category, features in categories.items():
            available_features = [f for f in features if f in feature_names]
            if available_features:
                available_conditions[category] = available_features

        feature_info["available_conditions"] = available_conditions

    elif model_config["feature_set"] == "none":
        feature_info["feature_names"] = []
        feature_info["feature_count"] = 0
        feature_info["available_conditions"] = {}
        feature_info["note"] = "Ця модель не використовує зовнішні ознаки (автономна модель часових рядів)."

    else:
        feature_info["feature_names"] = None
        feature_info["note"] = "Модель не завантажена або не має інформації про ознаки."

    return feature_info