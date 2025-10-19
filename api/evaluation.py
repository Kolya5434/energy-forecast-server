import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from typing import Dict, Any

from .features import generate_simple_features, generate_full_features

def _calculate_all_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculates the full set of metrics."""
    # Ensure inputs are numpy arrays for calculations
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) != len(y_pred_arr):
        # This case should ideally not happen if data prep is correct
        # Try aligning based on index if possible, otherwise raise error
        common_index = y_true.index.intersection(pd.Index(range(len(y_pred)))) # Example alignment, might need adjustment
        if len(common_index) == 0:
             raise ValueError(f"Length mismatch and index alignment failed: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
        y_true_arr = y_true.loc[common_index].values
        # Assuming y_pred corresponds to the same period, adjust its length/indices if needed
        # This part depends heavily on how y_pred indices align with y_true
        # Simplified: Assume y_pred corresponds directly if lengths match after potential alignment
        if len(y_true_arr) != len(y_pred_arr): # Check length again after potential alignment
             raise ValueError(f"Length mismatch after alignment: y_true ({len(y_true_arr)}) vs y_pred ({len(y_pred_arr)})")


    mask = (y_true_arr != 0) & (~np.isnan(y_true_arr)) & (~np.isnan(y_pred_arr)) # More robust mask
    y_true_masked = y_true_arr[mask]
    y_pred_masked = y_pred_arr[mask]

    if len(y_true_masked) == 0: # Handle case where all true values are zero or NaN
        mape = np.inf
    else:
        mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100

    # Calculate other metrics using masked arrays where appropriate or original if NaNs handled earlier
    valid_indices = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
    if not np.any(valid_indices):
         return { # Return default/error values if no valid data points
            "MAE": np.nan, "RMSE": np.nan, "R²": np.nan,
            "Explained Variance": np.nan, "MAPE (%)": np.nan
         }

    y_true_valid = y_true_arr[valid_indices]
    y_pred_valid = y_pred_arr[valid_indices]


    return {
        "MAE": mean_absolute_error(y_true_valid, y_pred_valid),
        "RMSE": np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        "R²": r2_score(y_true_valid, y_pred_valid),
        "Explained Variance": explained_variance_score(y_true_valid, y_pred_valid),
        "MAPE (%)": mape if mape != np.inf else None,
    }

# Updated function signature to accept data and models
def evaluate_model(
    model_id: str,
    historical_data_daily: pd.DataFrame,
    historical_data_hourly: pd.DataFrame,
    models_cache: Dict[str, Any],
    available_models: Dict[str, Any] # Pass AVAILABLE_MODELS too
    ) -> dict:
    """
    Dynamically evaluates a model by predicting on the test set.
    Accepts historical data and loaded models as arguments.
    """
    if model_id not in models_cache:
        raise KeyError(f"Model '{model_id}' not found in models_cache.")
    if historical_data_daily is None or historical_data_hourly is None:
        raise ValueError("Historical data not provided for evaluation.")


    model_config = available_models[model_id]
    model = models_cache[model_id]

    y_pred = []
    y_true = pd.Series(dtype=float) # Initialize as float Series

    # --- Dispatch evaluation logic ---
    if model_config["granularity"] == "daily":
        test_data = historical_data_daily.loc['2010-01-01':]
        if test_data.empty:
             raise ValueError("No daily test data found for the evaluation period (2010 onwards).")
        y_true = test_data['Global_active_power']

        if model_config["feature_set"] == "simple":
            X_test = generate_simple_features(test_data.index)
            # Ensure columns match training order
            if hasattr(model, 'feature_names_in_'):
                X_test = X_test[model.feature_names_in_]
            y_pred = model.predict(X_test)
        elif model_config["feature_set"] == "none": # ARIMA, Prophet
            if model_id == "Prophet":
                 # Prophet needs 'ds' column
                test_df_prophet = pd.DataFrame({'ds': test_data.index})
                forecast = model.predict(test_df_prophet)
                # Prophet predicts for the dates given, no need to align usually
                y_pred = forecast['yhat'].values
            else: # ARIMA/SARIMA
                # Ensure enough history exists if model needs it (though predict shouldn't)
                y_pred = model.predict(n_periods=len(test_data))

    elif model_config["granularity"] == "hourly":
        test_data_hourly = historical_data_hourly.loc['2010-01-01':]
        if test_data_hourly.empty:
             raise ValueError("No hourly test data found for the evaluation period (2010 onwards).")
        y_true = test_data_hourly['Global_active_power']

        if model_config["feature_set"] == "full":
            history_for_features = historical_data_hourly.loc[:'2009-12-31']
            if history_for_features.empty:
                 raise ValueError("No hourly history data found before 2010 for feature generation.")

            X_test = generate_full_features(history_for_features, test_data_hourly.index)
            if hasattr(model, 'feature_names_in_'):
                 X_test = X_test[model.feature_names_in_]
            y_pred = model.predict(X_test)
        else: # Includes DL models 'base_scaled'
            raise NotImplementedError(f"Dynamic evaluation for hourly models with feature_set '{model_config['feature_set']}' ('{model_id}') is not implemented.")

    if len(y_pred) == 0:
        raise ValueError(f"Prediction array is empty for model '{model_id}'.")
    if len(y_true) != len(y_pred):
         # Add more specific check/logging
         print(f"Warning/Error: Length mismatch for model {model_id}. y_true: {len(y_true)}, y_pred: {len(y_pred)}")
         # Attempt to align based on index, assuming y_pred corresponds to y_true's index
         try:
             y_pred_series = pd.Series(y_pred, index=y_true.index[:len(y_pred)]) # Simple alignment attempt
             y_true, y_pred_series = y_true.align(y_pred_series, join='inner') # Align strictly
             y_pred = y_pred_series.values
             if len(y_true) != len(y_pred): # Check again after alignment
                  raise ValueError(f"Length mismatch persists after index alignment for {model_id}.")
         except Exception as align_err:
              raise ValueError(f"Length mismatch and alignment failed for model {model_id}: y_true ({len(y_true)}) vs y_pred ({len(y_pred)}). Error: {align_err}")


    metrics = _calculate_all_metrics(y_true, y_pred)

    return {
        "model_id": model_id,
        "accuracy_metrics": metrics
        # Add performance metrics here if calculated within this function
    }