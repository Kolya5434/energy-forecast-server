import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import functools
import hashlib

BASE_DIR = Path(__file__).resolve().parent.parent


def _hash_dates(dates: pd.DatetimeIndex) -> str:
    """Create a hash from DatetimeIndex for caching."""
    return hashlib.md5(str(dates.tolist()).encode()).hexdigest()


@functools.lru_cache(maxsize=64)
def _generate_simple_features_cached(dates_hash: str, start: str, end: str, freq: str) -> pd.DataFrame:
    """Cached version of simple feature generation."""
    dates = pd.date_range(start=start, end=end, freq=freq)
    X_future = pd.DataFrame(index=dates)
    X_future['day_of_week'] = X_future.index.dayofweek
    X_future['month'] = X_future.index.month
    X_future['day_of_year'] = X_future.index.dayofyear
    return X_future


def generate_simple_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generates a DataFrame with simple time-based features for daily data (with caching)."""
    dates_hash = _hash_dates(dates)
    start = str(dates[0])
    end = str(dates[-1])
    freq = dates.freq.freqstr if dates.freq else 'D'
    return _generate_simple_features_cached(dates_hash, start, end, freq)


def generate_full_features(history_df: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generates the full set of features for hourly data using historical context.
    Mimics 'walk-forward' generation for API predictions.
    """
    if history_df is None or history_df.empty:
        raise ValueError("Historical data is required for full feature generation.")

    required_history_len = 168  # Need at least 1 week for lags/rolling
    if len(history_df) < required_history_len:
        raise ValueError(
            f"Not enough historical data provided. Need {required_history_len} hours, got {len(history_df)}.")

    # Combine history and future dates for calculation continuity
    combined_index = history_df.index.union(future_dates)
    temp_df = pd.DataFrame(index=combined_index)

    # Copy known target values from history (needed for lags/rolling)
    temp_df['Global_active_power'] = history_df['Global_active_power']

    # --- Full Feature Generation (based on notebooks) ---
    # Temporal features
    temp_df['hour'] = temp_df.index.hour
    temp_df['day_of_week'] = temp_df.index.dayofweek
    temp_df['day_of_month'] = temp_df.index.day
    temp_df['day_of_year'] = temp_df.index.dayofyear
    temp_df['week_of_year'] = temp_df.index.isocalendar().week.astype(int)
    temp_df['month'] = temp_df.index.month
    temp_df['year'] = temp_df.index.year
    temp_df['quarter'] = temp_df.index.quarter
    temp_df['is_weekend'] = (temp_df.index.dayofweek >= 5).astype(int)

    # Lag features (ensure enough history)
    lags_to_generate = [1, 2, 3, 24, 48, 168]
    for lag in lags_to_generate:
        temp_df[f'Global_active_power_lag_{lag}'] = temp_df['Global_active_power'].shift(lag)

    # Rolling window features (ensure enough history)
    windows_to_generate = [3, 6, 12, 24, 168]
    for window in windows_to_generate:
        # Shift(1) uses past data only
        rolling_series = temp_df['Global_active_power'].shift(1).rolling(window=window)
        temp_df[f'Global_active_power_roll_mean_{window}'] = rolling_series.mean()
        temp_df[f'Global_active_power_roll_std_{window}'] = rolling_series.std()

    # Add other base features from the dataset (if available in history)
    base_features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                     'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'is_anomaly']
    for col in base_features:
        if col in history_df.columns:
            temp_df[col] = history_df[col]
        else:
            temp_df[col] = 0  # Or handle missing base features appropriately

    # --- Add Simulated External Features (Holidays, Weather) ---
    # Holidays (example for France) - install 'holidays' library: pip install holidays
    try:
        import holidays
        min_year = history_df.index.min().year
        max_year = future_dates.max().year
        fr_holidays = holidays.France(years=range(min_year, max_year + 1))
        temp_df['is_holiday'] = temp_df.index.map(lambda d: d.date() in fr_holidays).astype(int)
    except ImportError:
        temp_df['is_holiday'] = 0

    # Weather (Simulated - replace with actual data source if available)
    temp_df['temperature'] = 15 - np.cos((temp_df.index.dayofyear - 15) * 2 * np.pi / 365.25) * 10 \
                             + np.random.normal(0, 2, len(temp_df))
    temp_df['humidity'] = 70 + np.random.normal(0, 10, len(temp_df))
    temp_df['wind_speed'] = 5 + np.random.normal(0, 2, len(temp_df))
    # --- End Feature Generation ---

    # Extract features only for the future dates
    future_features = temp_df.loc[future_dates]

    # Fill any NaNs resulting from lag/rolling calculations at the boundary
    future_features = future_features.bfill().ffill()

    # Drop the target column if it exists
    if 'Global_active_power' in future_features.columns:
        future_features = future_features.drop(columns=['Global_active_power'])

    return future_features


def create_sequences_for_dl(data: pd.DataFrame, sequence_length: int) -> np.ndarray:
    """Prepares input sequences for DL models using the saved scaler."""
    try:
        scaler_path = BASE_DIR / "models/standard_scaler.pkl"
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Please run scaling notebook.")

    # Select only the columns the scaler was fitted on
    scaler_feature_names = scaler.get_feature_names_out()

    # Ensure all expected columns are present
    missing_cols = set(scaler_feature_names) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Input data is missing columns required by the scaler: {missing_cols}")

    data_subset = data[scaler_feature_names]

    # Scale the data
    scaled_data = scaler.transform(data_subset)

    # Assume target 'Global_active_power' is the first column for the scaler
    # Find the actual index based on feature names
    try:
        target_index = list(scaler_feature_names).index('Global_active_power')
    except ValueError:
        raise ValueError("'Global_active_power' not found in scaler feature names.")

    # Remove the target column to create the input features for the model
    X_data = np.delete(scaled_data, target_index, axis=1)

    # Return only the last sequence of the required length
    if len(X_data) < sequence_length:
        raise ValueError(f"Not enough data to create sequence of length {sequence_length}. Got {len(X_data)}.")

    return np.expand_dims(X_data[-sequence_length:], axis=0)  # Shape: (1, sequence_length, num_features)