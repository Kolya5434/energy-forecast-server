"""
Scientific Visualization Module for Energy Forecasting
Generates high-quality plots for scientific publications and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import io
import base64
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Set scientific plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def plot_residual_analysis(
    residuals: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    model_name: str = "Model"
) -> str:
    """
    Create a comprehensive residual analysis plot (2x2 grid).

    Returns:
        Base64 encoded PNG image
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Residual Analysis: {model_name}', fontsize=14, fontweight='bold')

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(y_pred, residuals, 2)
    p = np.poly1d(z)
    x_line = np.linspace(y_pred.min(), y_pred.max(), 100)
    axes[0, 0].plot(x_line, p(x_line), "r-", alpha=0.5, linewidth=2)

    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Histogram with KDE
    axes[1, 0].hist(residuals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    # Fit normal distribution
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 0].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal Fit')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Scale-Location (sqrt of standardized residuals)
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Standardized Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(y_pred, np.sqrt(np.abs(standardized_residuals)), 2)
    p = np.poly1d(z)
    axes[1, 1].plot(x_line, p(x_line), "r-", alpha=0.5, linewidth=2)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_error_distribution(
    errors_dict: Dict[str, np.ndarray],
    title: str = "Model Error Distribution Comparison"
) -> str:
    """
    Create box plot and violin plot comparing error distributions.

    Args:
        errors_dict: {model_name: errors_array}

    Returns:
        Base64 encoded PNG image
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    data = pd.DataFrame(errors_dict)

    # Box plot
    data.boxplot(ax=axes[0], grid=True)
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title('Box Plot')
    axes[0].tick_params(axis='x', rotation=45)

    # Violin plot
    data_melted = data.melt(var_name='Model', value_name='Error')
    sns.violinplot(data=data_melted, x='Model', y='Error', ax=axes[1])
    axes[1].set_title('Violin Plot')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str] = None
) -> str:
    """
    Create radar chart comparing multiple models across metrics.

    Args:
        metrics_dict: {model_name: {metric_name: value}}
        metric_names: List of metrics to include (default: all)

    Returns:
        Base64 encoded PNG image
    """
    if metric_names is None:
        # Get all unique metric names
        metric_names = list(set().union(*[m.keys() for m in metrics_dict.values()]))

    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Normalize metrics to 0-1 scale for better visualization
    normalized_metrics = {}
    for metric in metric_names:
        values = [metrics_dict[model].get(metric, 0) for model in metrics_dict.keys()]
        max_val = max(values) if values else 1
        min_val = min(values) if values else 0
        normalized_metrics[metric] = {
            'max': max_val,
            'min': min_val,
            'range': max_val - min_val if max_val != min_val else 1
        }

    # Plot each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_dict)))
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        values = []
        for metric in metric_names:
            val = metrics.get(metric, 0)
            # Normalize (for error metrics, invert so lower is better)
            norm_info = normalized_metrics[metric]
            if metric.lower() in ['mae', 'rmse', 'mape', 'error']:
                # Invert for error metrics
                normalized = 1 - ((val - norm_info['min']) / norm_info['range']) if norm_info['range'] > 0 else 0.5
            else:
                # Normal for performance metrics
                normalized = ((val - norm_info['min']) / norm_info['range']) if norm_info['range'] > 0 else 0.5

            values.append(normalized)

        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    ax.set_title('Model Performance Comparison (Radar Chart)', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_temporal_error_analysis(
    errors: np.ndarray,
    timestamps: pd.DatetimeIndex,
    model_name: str = "Model"
) -> str:
    """
    Analyze errors over time with multiple temporal aggregations.

    Returns:
        Base64 encoded PNG image
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Temporal Error Analysis: {model_name}', fontsize=14, fontweight='bold')

    errors_series = pd.Series(errors, index=timestamps)

    # 1. Errors over time
    axes[0].plot(timestamps, errors, alpha=0.5, linewidth=0.5)
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title('Errors Over Time')
    axes[0].grid(True, alpha=0.3)

    # Add rolling average
    rolling_mean = errors_series.rolling(window=24*7, center=True).mean()
    axes[0].plot(timestamps, rolling_mean, 'r-', linewidth=2, label='7-day Rolling Mean')
    axes[0].legend()

    # 2. Hourly pattern
    if hasattr(timestamps, 'hour'):
        hourly_errors = errors_series.groupby(timestamps.hour).agg(['mean', 'std'])
        axes[1].errorbar(hourly_errors.index, hourly_errors['mean'],
                        yerr=hourly_errors['std'], fmt='o-', capsize=5)
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Error by Hour of Day')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(range(0, 24, 2))

    # 3. Day of week pattern
    if hasattr(timestamps, 'dayofweek'):
        daily_errors = errors_series.groupby(timestamps.dayofweek).agg(['mean', 'std'])
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[2].bar(range(7), daily_errors['mean'], yerr=daily_errors['std'],
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2].set_xlabel('Day of Week')
        axes[2].set_ylabel('Mean Absolute Error')
        axes[2].set_title('Error by Day of Week')
        axes[2].set_xticks(range(7))
        axes[2].set_xticklabels(day_names)
        axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    model_name: str = "Model",
    n_points: int = 200
) -> str:
    """
    Plot predictions with confidence intervals.

    Returns:
        Base64 encoded PNG image
    """
    # Sample points if too many
    if len(y_true) > n_points:
        indices = np.linspace(0, len(y_true) - 1, n_points, dtype=int)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        lower_bound = lower_bound[indices]
        upper_bound = upper_bound[indices]
        if timestamps is not None:
            timestamps = timestamps[indices]

    x = timestamps if timestamps is not None else np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(x, y_true, 'o', label='Actual', alpha=0.6, markersize=4, color='black')
    ax.plot(x, y_pred, '-', label='Predicted', linewidth=2, color='blue')
    ax.fill_between(x, lower_bound, upper_bound, alpha=0.3, color='blue', label='95% Prediction Interval')

    ax.set_xlabel('Time' if timestamps is not None else 'Sample Index')
    ax.set_ylabel('Energy Consumption')
    ax.set_title(f'Predictions with Confidence Intervals: {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if timestamps is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_learning_curves(
    train_sizes: List[int],
    train_scores: List[float],
    val_scores: List[float],
    metric_name: str = "Score",
    model_name: str = "Model"
) -> str:
    """
    Plot learning curves showing training and validation scores.

    Returns:
        Base64 encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
    ax.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2, markersize=8)

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Learning Curves: {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    model_name: str = "Model"
) -> str:
    """
    Plot feature importance as horizontal bar chart.

    Returns:
        Base64 encoded PNG image
    """
    # Sort and select top N
    indices = np.argsort(importances)[-top_n:]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    colors = plt.cm.viridis(sorted_importances / sorted_importances.max())
    ax.barh(range(top_n), sorted_importances, color=colors, edgecolor='black')

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances: {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_correlation_matrix(
    data: pd.DataFrame,
    title: str = "Feature Correlation Matrix"
) -> str:
    """
    Plot correlation matrix heatmap.

    Returns:
        Base64 encoded PNG image
    """
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_forecast_comparison(
    actual: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    timestamps: Optional[pd.DatetimeIndex] = None,
    n_points: int = 200
) -> str:
    """
    Compare forecasts from multiple models.

    Args:
        actual: Actual values
        predictions_dict: {model_name: predictions}
        timestamps: Optional timestamps
        n_points: Number of points to plot (for performance)

    Returns:
        Base64 encoded PNG image
    """
    # Sample points if too many
    if len(actual) > n_points:
        indices = np.linspace(0, len(actual) - 1, n_points, dtype=int)
        actual = actual[indices]
        predictions_dict = {k: v[indices] for k, v in predictions_dict.items()}
        if timestamps is not None:
            timestamps = timestamps[indices]

    x = timestamps if timestamps is not None else np.arange(len(actual))

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(x, actual, 'o-', label='Actual', linewidth=2, markersize=3, color='black', alpha=0.7)

    colors = plt.cm.Set2(np.linspace(0, 1, len(predictions_dict)))
    for idx, (model_name, preds) in enumerate(predictions_dict.items()):
        ax.plot(x, preds, '-', label=model_name, linewidth=1.5, alpha=0.8, color=colors[idx])

    ax.set_xlabel('Time' if timestamps is not None else 'Sample Index')
    ax.set_ylabel('Energy Consumption')
    ax.set_title('Model Forecast Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if timestamps is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_sensitivity_analysis(
    parameter_values: List[float],
    metric_values: List[float],
    parameter_name: str,
    metric_name: str = "Performance",
    model_name: str = "Model"
) -> str:
    """
    Plot sensitivity of model performance to parameter changes.

    Returns:
        Base64 encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(parameter_values, metric_values, 'o-', linewidth=2, markersize=8)

    # Mark optimal point
    if metric_name.lower() in ['mae', 'rmse', 'mape', 'error']:
        best_idx = np.argmin(metric_values)
    else:
        best_idx = np.argmax(metric_values)

    ax.plot(parameter_values[best_idx], metric_values[best_idx],
           'r*', markersize=20, label=f'Optimal: {parameter_values[best_idx]:.4f}')

    ax.set_xlabel(parameter_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f'Sensitivity Analysis: {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)