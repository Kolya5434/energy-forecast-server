import numpy as np
import pandas as pd
import joblib
import json
import time
import tensorflow as tf
from typing import List, Dict, Any
from prophet.serialize import model_from_json
from fastapi import HTTPException
import sklearn.ensemble

from .config import AVAILABLE_MODELS, BASE_DIR
from .schemas import PredictionRequest  # Імпортуємо схему запиту
from .features import generate_simple_features, generate_full_features, create_sequences_for_dl
from .evaluation import evaluate_model  # Імпортуємо функцію оцінки

MODELS_CACHE: Dict[str, Any] = {}
HISTORICAL_DATA_HOURLY = None
HISTORICAL_DATA_DAILY = None

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
    print("Historical data loaded successfully.")

    print("Loading models into cache...")
    _scaler = None
    _scaler_path = BASE_DIR / "models/standard_scaler.pkl"
    if _scaler_path.exists():
        _scaler = joblib.load(_scaler_path)
        print(" - Scaler loaded.")
    else:
        print(" - WARNING: Scaler file not found. DL models might fail.")

    for model_id, config in AVAILABLE_MODELS.items():
        try:
            model_path = config["path"]
            if not model_path.exists():
                print(f"   - WARNING: Model file not found for {model_id} at {model_path}. Skipping.")
                continue

            if config["type"] == "dl":
                # Перевіряємо наявність TensorFlow
                if 'tf' not in globals() or tf is None:
                    print(f"   - WARNING: TensorFlow not available. Skipping DL model: {model_id}")
                    continue
                MODELS_CACHE[model_id] = tf.keras.models.load_model(model_path)
            elif model_id == "Prophet":
                # Перевіряємо наявність Prophet
                if 'model_from_json' not in globals() or model_from_json is None:
                    print(f"   - WARNING: Prophet library function not available. Skipping Prophet model.")
                    continue
                with open(model_path, 'r') as f:
                    MODELS_CACHE[model_id] = model_from_json(json.load(f))
            else:  # classical, ml, ensemble
                MODELS_CACHE[model_id] = joblib.load(model_path)
            print(f" - Loaded model: {model_id}")
        except ImportError as ie:
            print(f"   - FAILED to load model {model_id} due to missing library: {ie}. Install required libraries.")
        except Exception as load_err:
            print(f"   - FAILED to load model {model_id}: {load_err}")
    print(f"Model loading complete. {len(MODELS_CACHE)} models loaded successfully.")

except FileNotFoundError as fnf_err:
    print(f"CRITICAL ERROR: Data file missing: {fnf_err}")
except Exception as startup_err:
    print(f"CRITICAL ERROR during server startup: {startup_err}")


def get_models_service() -> Dict[str, Any]:
    """Повертає інформацію про доступні (завантажені) моделі."""
    loaded_model_ids = list(MODELS_CACHE.keys())
    return {mid: {"type": AVAILABLE_MODELS[mid]["type"],
                  "granularity": AVAILABLE_MODELS[mid]["granularity"],
                  "feature_set": AVAILABLE_MODELS[mid]["feature_set"]}
            for mid in loaded_model_ids if mid in AVAILABLE_MODELS}


def predict_service(request: PredictionRequest) -> List[Dict[str, Any]]:
    """Виконує прогнозування для списку моделей, враховуючи їхні вимоги."""
    if not MODELS_CACHE or HISTORICAL_DATA_HOURLY is None or HISTORICAL_DATA_DAILY is None:
        raise HTTPException(status_code=503, detail="Сервіс недоступний: Моделі або історичні дані не завантажено.")

    results = []
    last_known_date_hourly = HISTORICAL_DATA_HOURLY.index.max()
    last_known_date_daily = HISTORICAL_DATA_DAILY.index.max()

    for model_id in request.model_ids:
        if model_id not in MODELS_CACHE:
            print(f"Warning: Model {model_id} requested but not loaded/available. Skipping.")
            results.append({
                "model_id": model_id,
                "forecast": {},
                "metadata": {"error": "Model not loaded or unavailable."}
            })
            continue

        start_time = time.time()
        model_config = AVAILABLE_MODELS[model_id]
        model = MODELS_CACHE[model_id]

        forecast_values: Dict[str, Any] = {}
        preds = []
        final_preds = []
        final_dates = pd.DatetimeIndex([])  # Ініціалізуємо як порожній DatetimeIndex
        metadata = {}  # Ініціалізуємо метадані тут

        try:
            # --- Логіка диспетчеризації та прогнозування ---
            if model_config["granularity"] == "daily":
                future_dates = pd.date_range(start=last_known_date_daily + pd.Timedelta(days=1),
                                             periods=request.forecast_horizon, freq='D')

                if model_config["feature_set"] == "simple":
                    X_future = generate_simple_features(future_dates)
                    if hasattr(model, 'feature_names_in_'):
                        X_future = X_future[model.feature_names_in_]
                    if X_future.isnull().values.any(): X_future = X_future.ffill().bfill()  # Basic NaN fill
                    if X_future.isnull().values.any(): raise ValueError("NaNs in features after fill")
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
                hourly_predictions = []

                if model_config.get("is_sequential"):  # DL Models
                    # --- ТИМЧАСОВА ЗАГЛУШКА (закоментовано реальну логіку) ---
                    print(f"Warning: Prediction for DL model '{model_id}' is temporarily disabled. Returning zeros.")
                    hourly_predictions = [0.0] * num_hours
                    # --- Кінець заглушки ---

                    """ # --- Реальна логіка Walk-Forward (потребує ретельного тестування) ---
                    sequence_len = model_config["sequence_length"]
                    if _scaler is None:
                        raise HTTPException(status_code=500, detail="Scaler not loaded. Cannot predict with DL model.")

                    scaler_features = _scaler.get_feature_names_out()
                    target_col_name = 'Global_active_power'
                    target_index_in_scaler = list(scaler_features).index(target_col_name)

                    current_history = HISTORICAL_DATA_HOURLY[scaler_features].iloc[-sequence_len:].copy()

                    print(f"Starting walk-forward prediction for {model_id}...")
                    for i in range(num_hours):
                        input_scaled = _scaler.transform(current_history)
                        X_input = np.delete(input_scaled, target_index_in_scaler, axis=1)
                        X_input_reshaped = np.expand_dims(X_input, axis=0)

                        pred_scaled = model.predict(X_input_reshaped, verbose=0)[0][0]

                        dummy_row_scaled = input_scaled[-1].copy()
                        dummy_row_scaled[target_index_in_scaler] = pred_scaled
                        inversed_row = _scaler.inverse_transform(dummy_row_scaled.reshape(1, -1))[0]
                        inversed_pred_value = inversed_row[target_index_in_scaler]

                        hourly_predictions.append(inversed_pred_value)

                        next_timestamp = current_history.index[-1] + pd.Timedelta(hours=1)
                        new_row_df = pd.DataFrame([inversed_row], columns=scaler_features, index=[next_timestamp])
                        current_history = pd.concat([current_history.iloc[1:], new_row_df])
                    """

                elif model_config["feature_set"] == "full":  # ML Models
                    history_slice = HISTORICAL_DATA_HOURLY.tail(168)  # Need 1 week for max lag/window
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
                        print(f"Warning: NaNs detected in features for {model_id}. Filling with 0.")
                        X_future = X_future.fillna(0)  # Fill NaNs
                        if X_future.isnull().values.any():
                            raise ValueError(f"NaNs persist after fillna for {model_id}")

                    hourly_predictions = model.predict(X_future)  # Assign to hourly_predictions

                # --- Агрегація годинних до денних ---
                if len(hourly_predictions) > 0:
                    hourly_preds_series = pd.Series(hourly_predictions, index=future_dates_hourly)
                    daily_preds = hourly_preds_series.resample('D').sum()
                    final_preds = daily_preds.values
                    final_dates = daily_preds.index
                else:  # Якщо прогноз не вдалося зробити (напр. DL заглушка була порожня)
                    final_preds = []
                    final_dates = pd.date_range(start=last_known_date_daily + pd.Timedelta(days=1),
                                                periods=request.forecast_horizon, freq='D')

            # --- Форматування результату ТІЛЬКИ якщо прогноз успішний ---
            # Перетворюємо numpy float32 на стандартний float Python
            forecast_values = {str(date.date()): float(pred) for date, pred in zip(final_dates, final_preds)}

        except Exception as pred_err:
            error_message = f"Помилка прогнозування для моделі {model_id}: {pred_err}"
            print(error_message)
            forecast_values = {}  # Залишаємо forecast порожнім у разі помилки
            metadata["error"] = error_message  # Додаємо помилку в метадані

        end_time = time.time()

        # --- Конструювання відповіді ---
        metadata["latency_ms"] = round((end_time - start_time) * 1000, 2)
        response_item = {
            "model_id": model_id,
            "forecast": forecast_values,
            "metadata": metadata  # Використовуємо словник metadata
        }
        results.append(response_item)

    return results


def get_evaluation_service(model_id: str) -> Dict[str, Any]:
    """
    Динамічно розраховує та повертає метрики якості для обраної моделі.
    """
    if model_id not in MODELS_CACHE:
        return {"error": f"Модель '{model_id}' не завантажена."}
    if HISTORICAL_DATA_DAILY is None or HISTORICAL_DATA_HOURLY is None:
        return {"error": "Історичні дані недоступні для оцінки."}

    try:
        # Передаємо необхідні дані в функцію оцінки
        evaluation_results = evaluate_model(
            model_id=model_id,
            historical_data_daily=HISTORICAL_DATA_DAILY,
            historical_data_hourly=HISTORICAL_DATA_HOURLY,
            models_cache=MODELS_CACHE,
            available_models=AVAILABLE_MODELS
        )

        # Можна додати вимірювання latency тут, якщо потрібно
        evaluation_results["performance_metrics"] = {
            "avg_latency_ms": None,  # Placeholder
            "memory_increment_mb": None  # Placeholder
        }

        return evaluation_results

    except (KeyError, NotImplementedError, ValueError, FileNotFoundError) as e:
        print(f"Помилка при оцінці моделі {model_id}: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"Неочікувана помилка під час оцінки {model_id}: {e}")
        return {"error": f"Неочікувана помилка під час оцінки моделі '{model_id}'."}

def get_interpretation_service(model_id: str) -> Dict[str, Any]:
    """
    Повертає дані для інтерпретації для обраної моделі,
    включаючи feature_importance та SHAP values (якщо можливо).
    """
    if model_id not in MODELS_CACHE:
        return {"error": f"Модель '{model_id}' не завантажена."}

    model_config = AVAILABLE_MODELS.get(model_id)
    if not model_config or model_config["type"] not in ["ml", "ensemble"]:
        return {"error": f"Інтерпретація недоступна для моделі типу '{model_config.get('type')}'."}

    model = MODELS_CACHE[model_id]
    response: Dict[str, Any] = {"model_id": model_id}

    feature_importance_dict = None
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


    shap_explanation = None
    shap_error = None
    try:
        import shap

        is_tree_model = False
        if hasattr(model, '_Booster'):
            is_tree_model = True
        elif 'sklearn' in globals() and isinstance(model, (sklearn.ensemble.RandomForestRegressor, sklearn.ensemble.GradientBoostingRegressor)):
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

        # Перевірка на NaN перед SHAP
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
            raise ValueError(f"Розмірність SHAP values ({len(shap_contributions)}) не збігається з кількістю ознак ({len(X_future.columns)}).")

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


    has_fi = isinstance(response.get("feature_importance"), dict) and "error" not in response.get("feature_importance", {})
    has_shap = isinstance(response.get("shap_values"), dict) and "error" not in response.get("shap_values", {})

    if not has_fi and not has_shap:
         final_error = response.get("shap_values", {}).get("error", "Не вдалося отримати жодних даних інтерпретації.")
         return {"error": final_error}

    return response