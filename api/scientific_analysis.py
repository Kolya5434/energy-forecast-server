"""
Scientific Analysis Module for Energy Forecasting
Provides statistical tests, residual analysis, and error analysis for models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def perform_statistical_comparison(
    predictions_dict: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Dict[str, Any]:
    """
    Performs statistical significance tests comparing model predictions.

    Args:
        predictions_dict: {model_id: predictions array}
        y_true: Ground truth values

    Returns:
        Dictionary with statistical test results
    """
    model_ids = list(predictions_dict.keys())
    errors_dict = {
        model_id: np.abs(y_true - preds)
        for model_id, preds in predictions_dict.items()
    }

    results = {
        "num_models": len(model_ids),
        "sample_size": len(y_true),
        "pairwise_tests": {}
    }

    # Pairwise comparisons
    for i, model_1 in enumerate(model_ids):
        for model_2 in model_ids[i+1:]:
            errors_1 = errors_dict[model_1]
            errors_2 = errors_dict[model_2]

            # Paired t-test
            t_stat, t_pvalue = stats.ttest_rel(errors_1, errors_2)

            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, w_pvalue = stats.wilcoxon(errors_1, errors_2)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(errors_1) + np.var(errors_2)) / 2)
            cohens_d = (np.mean(errors_1) - np.mean(errors_2)) / pooled_std if pooled_std > 0 else 0

            comparison_key = f"{model_1}_vs_{model_2}"
            results["pairwise_tests"][comparison_key] = {
                "t_test": {
                    "statistic": float(t_stat),
                    "p_value": float(t_pvalue),
                    "significant": bool(t_pvalue < 0.05)
                },
                "wilcoxon_test": {
                    "statistic": float(w_stat),
                    "p_value": float(w_pvalue),
                    "significant": bool(w_pvalue < 0.05)
                },
                "effect_size": {
                    "cohens_d": float(cohens_d),
                    "interpretation": _interpret_cohens_d(cohens_d)
                },
                "mean_error_diff": float(np.mean(errors_1) - np.mean(errors_2)),
                "better_model": model_1 if np.mean(errors_1) < np.mean(errors_2) else model_2
            }

    # Friedman test (if more than 2 models)
    if len(model_ids) > 2:
        errors_array = np.array([errors_dict[mid] for mid in model_ids])
        friedman_stat, friedman_pvalue = stats.friedmanchisquare(*errors_array)
        results["friedman_test"] = {
            "statistic": float(friedman_stat),
            "p_value": float(friedman_pvalue),
            "significant": bool(friedman_pvalue < 0.05),
            "interpretation": "At least one model significantly differs" if friedman_pvalue < 0.05 else "No significant difference"
        }

    return results


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex = None
) -> Dict[str, Any]:
    """
    Comprehensive residual analysis.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        timestamps: Optional timestamps for temporal analysis

    Returns:
        Dictionary with residual analysis results
    """
    residuals = y_true - y_pred

    # Basic statistics
    basic_stats = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "q25": float(np.percentile(residuals, 25)),
        "q75": float(np.percentile(residuals, 75)),
        "iqr": float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
    }

    # Normality tests
    shapiro_stat, shapiro_pvalue = stats.shapiro(residuals[:5000])  # Limit for performance
    ks_stat, ks_pvalue = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))

    normality = {
        "shapiro_wilk": {
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_pvalue),
            "is_normal": bool(shapiro_pvalue > 0.05)
        },
        "kolmogorov_smirnov": {
            "statistic": float(ks_stat),
            "p_value": float(ks_pvalue),
            "is_normal": bool(ks_pvalue > 0.05)
        },
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals))
    }

    # Autocorrelation (if timestamps provided)
    autocorr = {}
    if timestamps is not None and len(residuals) > 1:
        residuals_series = pd.Series(residuals, index=timestamps)
        autocorr = {
            "lag_1": float(residuals_series.autocorr(lag=1)) if len(residuals_series) > 1 else None,
            "lag_7": float(residuals_series.autocorr(lag=7)) if len(residuals_series) > 7 else None,
            "lag_30": float(residuals_series.autocorr(lag=30)) if len(residuals_series) > 30 else None
        }

    # Heteroscedasticity - divide into bins and check variance stability
    n_bins = 10
    bin_size = len(residuals) // n_bins
    bin_variances = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(residuals)
        bin_var = np.var(residuals[start_idx:end_idx])
        bin_variances.append(float(bin_var))

    heteroscedasticity = {
        "bin_variances": bin_variances,
        "variance_ratio": float(max(bin_variances) / min(bin_variances)) if min(bin_variances) > 0 else None,
        "is_homoscedastic": bool(max(bin_variances) / min(bin_variances) < 2) if min(bin_variances) > 0 else None
    }

    # Distribution percentiles for plotting
    percentiles = {
        f"p{int(p)}": float(np.percentile(residuals, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }

    return {
        "basic_statistics": basic_stats,
        "normality": normality,
        "autocorrelation": autocorr,
        "heteroscedasticity": heteroscedasticity,
        "percentiles": percentiles
    }


def perform_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex = None,
    features: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Detailed error analysis including temporal patterns and feature correlations.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        timestamps: Optional timestamps for temporal analysis
        features: Optional features DataFrame for correlation analysis

    Returns:
        Dictionary with error analysis results
    """
    errors = np.abs(y_true - y_pred)

    # Overall metrics
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": calculate_mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "max_error": float(np.max(errors)),
        "median_error": float(np.median(errors))
    }

    # Error distribution
    error_distribution = {
        "bins": 20,
        "histogram": {
            f"bin_{i}": int(count)
            for i, count in enumerate(np.histogram(errors, bins=20)[0])
        },
        "bin_edges": [float(x) for x in np.histogram(errors, bins=20)[1].tolist()]
    }

    # Large errors analysis
    threshold_95 = np.percentile(errors, 95)
    large_errors_idx = errors > threshold_95

    large_errors = {
        "threshold": float(threshold_95),
        "count": int(np.sum(large_errors_idx)),
        "percentage": float(np.sum(large_errors_idx) / len(errors) * 100),
        "mean_value": float(np.mean(errors[large_errors_idx])) if np.any(large_errors_idx) else None
    }

    # Temporal patterns (if timestamps provided)
    temporal_patterns = {}
    if timestamps is not None:
        errors_series = pd.Series(errors, index=timestamps)

        # By hour
        if hasattr(timestamps, 'hour'):
            hourly_errors = errors_series.groupby(timestamps.hour).agg(['mean', 'std', 'count'])
            temporal_patterns["hourly"] = {
                int(hour): {
                    "mean_error": float(row['mean']),
                    "std_error": float(row['std']),
                    "count": int(row['count'])
                }
                for hour, row in hourly_errors.iterrows()
            }

        # By day of week
        if hasattr(timestamps, 'dayofweek'):
            daily_errors = errors_series.groupby(timestamps.dayofweek).agg(['mean', 'std', 'count'])
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            temporal_patterns["daily"] = {
                day_names[day]: {
                    "mean_error": float(row['mean']),
                    "std_error": float(row['std']),
                    "count": int(row['count'])
                }
                for day, row in daily_errors.iterrows()
            }

        # By month
        if hasattr(timestamps, 'month'):
            monthly_errors = errors_series.groupby(timestamps.month).agg(['mean', 'std', 'count'])
            temporal_patterns["monthly"] = {
                int(month): {
                    "mean_error": float(row['mean']),
                    "std_error": float(row['std']),
                    "count": int(row['count'])
                }
                for month, row in monthly_errors.iterrows()
            }

    # Feature correlation with errors (if features provided)
    feature_correlations = {}
    if features is not None:
        for col in features.columns:
            if pd.api.types.is_numeric_dtype(features[col]):
                try:
                    corr = np.corrcoef(features[col].fillna(0), errors)[0, 1]
                    if not np.isnan(corr):
                        feature_correlations[col] = float(corr)
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for {col}: {e}")

    return {
        "metrics": metrics,
        "error_distribution": error_distribution,
        "large_errors": large_errors,
        "temporal_patterns": temporal_patterns,
        "feature_correlations": feature_correlations
    }


def calculate_prediction_intervals(
    residuals: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate prediction intervals based on residual distribution.

    Args:
        residuals: Model residuals
        confidence_level: Confidence level (default 0.95 for 95%)

    Returns:
        Dictionary with interval bounds
    """
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        "confidence_level": confidence_level,
        "lower_bound": float(np.percentile(residuals, lower_percentile)),
        "upper_bound": float(np.percentile(residuals, upper_percentile)),
        "margin": float(np.percentile(residuals, upper_percentile) - np.percentile(residuals, lower_percentile))
    }


def perform_cross_validation_analysis(
    cv_results: List[Dict[str, float]]
) -> Dict[str, Any]:
    """
    Analyze cross-validation results for stability assessment.

    Args:
        cv_results: List of metric dictionaries from each fold

    Returns:
        Aggregated statistics
    """
    metrics = {}

    # Extract all metric names
    metric_names = cv_results[0].keys() if cv_results else []

    for metric in metric_names:
        values = [fold[metric] for fold in cv_results if metric in fold]

        if values:
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else None,  # Coefficient of variation
                "values": [float(v) for v in values]
            }

    return {
        "num_folds": len(cv_results),
        "metrics": metrics,
        "stability_score": _calculate_stability_score(metrics)
    }


def _calculate_stability_score(metrics: Dict[str, Dict]) -> float:
    """
    Calculate overall stability score based on coefficient of variation.
    Lower CV = more stable model.
    """
    cvs = [m.get("cv", 0) for m in metrics.values() if m.get("cv") is not None]
    return float(1 / (1 + np.mean(cvs))) if cvs else 0.0


def calculate_model_complexity_metrics(
    model: Any,
    model_type: str
) -> Dict[str, Any]:
    """
    Calculate complexity metrics for a model.

    Args:
        model: The trained model
        model_type: Type of model ("ml", "dl", "classical", "ensemble")

    Returns:
        Complexity metrics
    """
    complexity = {
        "model_type": model_type
    }

    if model_type == "ml" or model_type == "ensemble":
        # For tree-based models
        if hasattr(model, 'n_estimators'):
            complexity["n_estimators"] = int(model.n_estimators)
        if hasattr(model, 'max_depth'):
            complexity["max_depth"] = int(model.max_depth) if model.max_depth is not None else None
        if hasattr(model, 'n_features_in_'):
            complexity["n_features"] = int(model.n_features_in_)

        # Estimate number of parameters
        if hasattr(model, 'estimators_'):
            try:
                total_nodes = sum(est.tree_.node_count for est in model.estimators_ if hasattr(est, 'tree_'))
                complexity["total_tree_nodes"] = int(total_nodes)
            except:
                pass

    elif model_type == "dl":
        # For neural networks
        if hasattr(model, 'count_params'):
            complexity["total_parameters"] = int(model.count_params())
        if hasattr(model, 'layers'):
            complexity["num_layers"] = len(model.layers)
            complexity["trainable_params"] = int(sum(np.prod(layer.get_weights()[0].shape)
                                                     for layer in model.layers
                                                     if layer.get_weights()))

    return complexity