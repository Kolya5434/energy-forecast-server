"""
Sensitivity Analysis Module for Energy Forecasting
Analyzes model sensitivity to hyperparameters and input features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import copy

logger = logging.getLogger(__name__)


def perform_feature_ablation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    metric_fn: Callable,
    feature_groups: Dict[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Perform feature ablation study to measure importance of feature groups.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        metric_fn: Metric function (y_true, y_pred) -> float
        feature_groups: Optional groups of features {"group_name": [feature_names]}

    Returns:
        Dictionary with ablation results
    """
    # Baseline performance with all features
    baseline_pred = model.predict(X_test)
    baseline_score = metric_fn(y_test, baseline_pred)

    results = {
        "baseline_score": float(baseline_score),
        "feature_ablations": {}
    }

    # If no groups provided, create individual feature groups
    if feature_groups is None:
        feature_groups = {col: [col] for col in X_test.columns}

    # Test each feature group
    for group_name, features in feature_groups.items():
        # Remove features from this group
        X_ablated = X_test.copy()

        # Replace with zeros or mean
        for feat in features:
            if feat in X_ablated.columns:
                X_ablated[feat] = 0  # or X_ablated[feat].mean()

        # Predict with ablated features
        ablated_pred = model.predict(X_ablated)
        ablated_score = metric_fn(y_test, ablated_pred)

        # Calculate impact
        score_diff = ablated_score - baseline_score

        results["feature_ablations"][group_name] = {
            "ablated_score": float(ablated_score),
            "score_difference": float(score_diff),
            "relative_importance": float(abs(score_diff) / baseline_score) if baseline_score != 0 else 0,
            "features": features
        }

    # Rank by importance
    ranked = sorted(results["feature_ablations"].items(),
                   key=lambda x: abs(x[1]["score_difference"]),
                   reverse=True)

    results["ranking"] = [
        {"rank": i+1, "group": group, **info}
        for i, (group, info) in enumerate(ranked)
    ]

    return results


def perform_input_perturbation_analysis(
    model: Any,
    X_baseline: pd.DataFrame,
    y_baseline: np.ndarray,
    features_to_perturb: List[str],
    perturbation_ranges: Dict[str, Tuple[float, float]],
    n_steps: int = 10
) -> Dict[str, Any]:
    """
    Analyze model sensitivity to input perturbations.

    Args:
        model: Trained model
        X_baseline: Baseline input features
        y_baseline: Baseline targets
        features_to_perturb: List of feature names to perturb
        perturbation_ranges: {feature: (min_multiplier, max_multiplier)}
        n_steps: Number of perturbation steps

    Returns:
        Dictionary with perturbation analysis results
    """
    from sklearn.metrics import mean_absolute_error

    results = {
        "baseline_mae": float(mean_absolute_error(y_baseline, model.predict(X_baseline))),
        "perturbations": {}
    }

    for feature in features_to_perturb:
        if feature not in X_baseline.columns:
            logger.warning(f"Feature {feature} not found in data. Skipping.")
            continue

        min_mult, max_mult = perturbation_ranges.get(feature, (0.5, 1.5))
        multipliers = np.linspace(min_mult, max_mult, n_steps)

        feature_results = {
            "multipliers": multipliers.tolist(),
            "mae_values": [],
            "predictions_mean": [],
            "predictions_std": []
        }

        for mult in multipliers:
            X_perturbed = X_baseline.copy()
            X_perturbed[feature] = X_perturbed[feature] * mult

            preds = model.predict(X_perturbed)
            mae = mean_absolute_error(y_baseline, preds)

            feature_results["mae_values"].append(float(mae))
            feature_results["predictions_mean"].append(float(np.mean(preds)))
            feature_results["predictions_std"].append(float(np.std(preds)))

        # Calculate sensitivity score
        mae_range = max(feature_results["mae_values"]) - min(feature_results["mae_values"])
        feature_results["sensitivity_score"] = float(mae_range / results["baseline_mae"]) if results["baseline_mae"] > 0 else 0

        results["perturbations"][feature] = feature_results

    # Rank features by sensitivity
    ranked = sorted(results["perturbations"].items(),
                   key=lambda x: x[1]["sensitivity_score"],
                   reverse=True)

    results["sensitivity_ranking"] = [
        {"rank": i+1, "feature": feat, "sensitivity_score": info["sensitivity_score"]}
        for i, (feat, info) in enumerate(ranked)
    ]

    return results


def analyze_hyperparameter_sensitivity(
    train_fn: Callable,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    param_grid: Dict[str, List[Any]],
    metric_fn: Callable,
    base_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze sensitivity to individual hyperparameters.

    Args:
        train_fn: Function to train model: train_fn(X, y, **params) -> model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        param_grid: {param_name: [values_to_test]}
        metric_fn: Metric function (y_true, y_pred) -> float
        base_params: Base parameters to use

    Returns:
        Dictionary with sensitivity results for each parameter
    """
    if base_params is None:
        base_params = {}

    results = {
        "parameters": {},
        "baseline_params": base_params
    }

    # Baseline score
    baseline_model = train_fn(X_train, y_train, **base_params)
    baseline_pred = baseline_model.predict(X_val)
    baseline_score = metric_fn(y_val, baseline_pred)
    results["baseline_score"] = float(baseline_score)

    # Test each parameter
    for param_name, param_values in param_grid.items():
        logger.info(f"Testing sensitivity to {param_name}...")

        param_results = {
            "values": [],
            "scores": [],
            "best_value": None,
            "best_score": None,
            "score_range": None,
            "sensitivity": None
        }

        for param_value in param_values:
            # Create params with this specific value
            test_params = base_params.copy()
            test_params[param_name] = param_value

            try:
                # Train model
                model = train_fn(X_train, y_train, **test_params)
                preds = model.predict(X_val)
                score = metric_fn(y_val, preds)

                param_results["values"].append(param_value)
                param_results["scores"].append(float(score))

            except Exception as e:
                logger.error(f"Error testing {param_name}={param_value}: {e}")
                continue

        if param_results["scores"]:
            # Determine best value (lower is better for error metrics)
            best_idx = np.argmin(param_results["scores"])
            param_results["best_value"] = param_results["values"][best_idx]
            param_results["best_score"] = param_results["scores"][best_idx]

            # Calculate sensitivity
            score_range = max(param_results["scores"]) - min(param_results["scores"])
            param_results["score_range"] = float(score_range)
            param_results["sensitivity"] = float(score_range / baseline_score) if baseline_score > 0 else 0

            results["parameters"][param_name] = param_results

    # Rank parameters by sensitivity
    ranked = sorted(results["parameters"].items(),
                   key=lambda x: x[1]["sensitivity"],
                   reverse=True)

    results["sensitivity_ranking"] = [
        {
            "rank": i+1,
            "parameter": param,
            "sensitivity": info["sensitivity"],
            "best_value": info["best_value"]
        }
        for i, (param, info) in enumerate(ranked)
    ]

    return results


def perform_noise_robustness_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    noise_levels: List[float] = None,
    n_iterations: int = 10
) -> Dict[str, Any]:
    """
    Test model robustness to input noise.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        noise_levels: Noise standard deviations as fraction of feature std
        n_iterations: Number of noise iterations per level

    Returns:
        Noise robustness analysis results
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # Baseline (no noise)
    baseline_preds = model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    results = {
        "baseline_mae": float(baseline_mae),
        "noise_levels": [],
        "mae_mean": [],
        "mae_std": [],
        "rmse_mean": [],
        "r2_mean": [],
        "degradation": []
    }

    for noise_level in noise_levels:
        maes = []
        rmses = []
        r2s = []

        for _ in range(n_iterations):
            # Add Gaussian noise
            X_noisy = X_test.copy()
            for col in X_test.columns:
                if pd.api.types.is_numeric_dtype(X_test[col]):
                    std = X_test[col].std()
                    noise = np.random.normal(0, noise_level * std, size=len(X_test))
                    X_noisy[col] = X_test[col] + noise

            # Predict
            preds_noisy = model.predict(X_noisy)

            # Calculate metrics
            maes.append(mean_absolute_error(y_test, preds_noisy))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds_noisy)))
            r2s.append(r2_score(y_test, preds_noisy))

        results["noise_levels"].append(float(noise_level))
        results["mae_mean"].append(float(np.mean(maes)))
        results["mae_std"].append(float(np.std(maes)))
        results["rmse_mean"].append(float(np.mean(rmses)))
        results["r2_mean"].append(float(np.mean(r2s)))
        results["degradation"].append(
            float((np.mean(maes) - baseline_mae) / baseline_mae * 100) if baseline_mae > 0 else 0
        )

    # Overall robustness score (lower degradation at high noise = more robust)
    avg_degradation = np.mean(results["degradation"][1:])  # Exclude 0 noise level
    results["robustness_score"] = float(max(0, 100 - avg_degradation))

    return results


def cross_validate_with_different_splits(
    train_fn: Callable,
    X: pd.DataFrame,
    y: np.ndarray,
    split_configs: List[Dict[str, Any]],
    metric_fn: Callable,
    model_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Perform cross-validation with different splitting strategies.

    Args:
        train_fn: Function to train model
        X, y: Full dataset
        split_configs: List of configurations like {"name": "5-fold", "n_splits": 5}
        metric_fn: Metric function
        model_params: Model parameters

    Returns:
        Cross-validation results for each strategy
    """
    from sklearn.model_selection import TimeSeriesSplit, KFold

    if model_params is None:
        model_params = {}

    results = {
        "strategies": {}
    }

    for config in split_configs:
        strategy_name = config["name"]
        n_splits = config.get("n_splits", 5)

        if "timeseries" in strategy_name.lower():
            splitter = TimeSeriesSplit(n_splits=n_splits)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        scores = []
        fold_info = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y[val_idx]

            # Train model
            model = train_fn(X_train_fold, y_train_fold, **model_params)
            preds = model.predict(X_val_fold)
            score = metric_fn(y_val_fold, preds)

            scores.append(float(score))
            fold_info.append({
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "score": float(score)
            })

        results["strategies"][strategy_name] = {
            "n_splits": n_splits,
            "scores": scores,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "cv_coefficient": float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else None,
            "fold_details": fold_info
        }

    return results