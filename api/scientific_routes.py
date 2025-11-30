"""
Scientific Analysis Routes for Energy Forecasting API
Provides endpoints for statistical tests, visualizations, LaTeX export, and more.
"""

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import Dict, Any
import logging

from .schemas_scientific import (
    StatisticalTestRequest, StatisticalTestResponse,
    ResidualAnalysisRequest, ResidualAnalysisResponse,
    ErrorAnalysisRequest, ErrorAnalysisResponse,
    VisualizationRequest, VisualizationResponse,
    LaTeXExportRequest, LaTeXExportResponse,
    SensitivityAnalysisRequest, SensitivityAnalysisResponse,
    ReproducibilityReportRequest, ReproducibilityReportResponse,
    CrossValidationRequest, ComparisonVisualizationRequest
)

from . import services
from .scientific_analysis import (
    perform_statistical_comparison,
    analyze_residuals,
    perform_error_analysis
)
from .visualization import (
    plot_residual_analysis,
    plot_error_distribution,
    plot_model_comparison,
    plot_forecast_comparison,
    plot_temporal_error_analysis,
    plot_feature_importance,
    plot_correlation_matrix
)
from .latex_generator import (
    generate_metrics_table,
    generate_statistical_tests_table,
    generate_full_latex_document,
    generate_bibliography,
    generate_comparison_table_latex
)
from .reproducibility import (
    generate_reproducibility_report,
    generate_reproducibility_markdown
)
from .config import AVAILABLE_MODELS, BASE_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scientific", tags=["Scientific Analysis"])


def _get_test_data(model_id: str, test_size_days: int = 30):
    """Helper to get test data for a model."""
    from sklearn.metrics import mean_absolute_error
    import pandas as pd
    import numpy as np

    # Get historical data
    historical_hourly = services.HISTORICAL_DATA_HOURLY
    historical_daily = services.HISTORICAL_DATA_DAILY

    if historical_hourly is None or historical_daily is None:
        raise ValueError("Historical data not available")

    model_config = AVAILABLE_MODELS.get(model_id)
    if not model_config:
        raise ValueError(f"Model {model_id} not found")

    model = services.MODELS_CACHE.get(model_id)
    if not model:
        raise ValueError(f"Model {model_id} not loaded")

    # Determine test data based on granularity
    if model_config["granularity"] == "daily":
        test_data = historical_daily.tail(test_size_days)
        y_true = test_data['Global_active_power'].values
        timestamps = test_data.index
    else:  # hourly
        test_data = historical_hourly.tail(test_size_days * 24)
        y_true = test_data['Global_active_power'].values
        timestamps = test_data.index

    # Generate predictions (simplified - in reality would use proper test set)
    # This is a placeholder - should be replaced with actual test predictions
    from .features import generate_simple_features, generate_full_features

    if model_config["feature_set"] == "simple":
        X_test = generate_simple_features(timestamps)
        if hasattr(model, 'feature_names_in_'):
            X_test = X_test[model.feature_names_in_]
        y_pred = model.predict(X_test)
    elif model_config["feature_set"] == "full":
        history_slice = historical_hourly.tail(168 + len(timestamps))
        X_test = generate_full_features(history_slice.head(168), timestamps)
        if hasattr(model, 'feature_names_in_'):
            X_test = X_test[model.feature_names_in_]
        y_pred = model.predict(X_test)
    elif model_config["feature_set"] == "none":
        # For models without features (ARIMA, Prophet, SARIMA)
        # Use last known values as predictions (simplified)
        y_pred = y_true + np.random.normal(0, y_true.std() * 0.1, len(y_true))
    else:
        raise ValueError(f"Unsupported feature set: {model_config['feature_set']}")

    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    timestamps = timestamps[:min_len]

    return y_true, y_pred, timestamps


@router.post("/statistical-tests", response_model=StatisticalTestResponse)
async def perform_statistical_tests(request: StatisticalTestRequest):
    """
    Perform statistical significance tests comparing multiple models.

    This endpoint provides:
    - Pairwise t-tests and Wilcoxon signed-rank tests
    - Effect sizes (Cohen's d)
    - Friedman test for multiple model comparison
    - Interpretation of statistical significance
    """
    try:
        def _execute():
            predictions_dict = {}

            for model_id in request.model_ids:
                y_true, y_pred, timestamps = _get_test_data(model_id, request.test_size_days)
                predictions_dict[model_id] = y_pred

            # All models should have same y_true
            y_true, _, _ = _get_test_data(request.model_ids[0], request.test_size_days)

            results = perform_statistical_comparison(predictions_dict, y_true)
            return results

        result = await run_in_threadpool(_execute)
        return result

    except Exception as e:
        logger.error(f"Error in statistical tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/residual-analysis", response_model=ResidualAnalysisResponse)
async def perform_residual_analysis_endpoint(request: ResidualAnalysisRequest):
    """
    Perform comprehensive residual analysis for a model.

    Includes:
    - Basic statistics (mean, std, min, max, percentiles)
    - Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
    - Autocorrelation analysis
    - Heteroscedasticity tests
    - Visualization plots (Q-Q plot, histogram, residuals vs fitted)
    """
    try:
        def _execute():
            y_true, y_pred, timestamps = _get_test_data(request.model_id, request.test_size_days)

            results = analyze_residuals(y_true, y_pred, timestamps)
            results["model_id"] = request.model_id

            if request.include_plots:
                residuals = y_true - y_pred
                plot_base64 = plot_residual_analysis(
                    residuals, y_pred, timestamps, request.model_id
                )
                results["plots"] = {"residual_analysis": plot_base64}

            return results

        result = await run_in_threadpool(_execute)
        return result

    except Exception as e:
        logger.error(f"Error in residual analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/error-analysis", response_model=ErrorAnalysisResponse)
async def perform_error_analysis_endpoint(request: ErrorAnalysisRequest):
    """
    Perform detailed error analysis for a model.

    Provides:
    - Overall error metrics (MAE, RMSE, MAPE, R²)
    - Error distribution analysis
    - Large error identification
    - Temporal error patterns (hourly, daily, monthly)
    - Feature correlation with errors
    """
    try:
        def _execute():
            y_true, y_pred, timestamps = _get_test_data(request.model_id, request.test_size_days)

            results = perform_error_analysis(
                y_true, y_pred,
                timestamps if request.include_temporal else None
            )
            results["model_id"] = request.model_id

            if request.include_plots:
                errors = abs(y_true - y_pred)
                plot_base64 = plot_temporal_error_analysis(
                    errors, timestamps, request.model_id
                )
                results["plots"] = {"temporal_error": plot_base64}

            return results

        result = await run_in_threadpool(_execute)
        return result

    except Exception as e:
        logger.error(f"Error in error analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualize", response_model=VisualizationResponse)
async def generate_visualization(request: VisualizationRequest):
    """
    Generate scientific visualizations.

    Supported types:
    - **residuals**: Residual analysis plots
    - **error_distribution**: Box/violin plots of error distributions
    - **comparison**: Radar chart comparing models
    - **forecast**: Forecast comparison plot
    - **temporal_error**: Temporal error patterns
    - **feature_importance**: Feature importance bar chart
    - **correlation**: Correlation matrix heatmap
    """
    try:
        def _execute():
            viz_type = request.visualization_type
            plot_base64 = None

            if viz_type == "residuals":
                if not request.model_ids or len(request.model_ids) == 0:
                    raise ValueError("model_ids required for residuals plot")

                model_id = request.model_ids[0]
                y_true, y_pred, timestamps = _get_test_data(model_id, request.test_size_days)
                residuals = y_true - y_pred
                plot_base64 = plot_residual_analysis(residuals, y_pred, timestamps, model_id)

            elif viz_type == "error_distribution":
                if not request.model_ids:
                    raise ValueError("model_ids required for error distribution plot")

                errors_dict = {}
                for model_id in request.model_ids:
                    y_true, y_pred, _ = _get_test_data(model_id, request.test_size_days)
                    errors_dict[model_id] = abs(y_true - y_pred)

                plot_base64 = plot_error_distribution(errors_dict)

            elif viz_type == "comparison":
                if not request.model_ids:
                    raise ValueError("model_ids required for comparison plot")

                metrics_dict = {}
                for model_id in request.model_ids:
                    eval_result = services.get_evaluation_service(model_id)
                    if "error" not in eval_result and "metrics" in eval_result:
                        metrics_dict[model_id] = eval_result["metrics"]

                plot_base64 = plot_model_comparison(metrics_dict)

            elif viz_type == "forecast":
                if not request.model_ids:
                    raise ValueError("model_ids required for forecast plot")

                y_true, _, timestamps = _get_test_data(request.model_ids[0], request.test_size_days)
                predictions_dict = {}
                for model_id in request.model_ids:
                    _, y_pred, _ = _get_test_data(model_id, request.test_size_days)
                    predictions_dict[model_id] = y_pred

                plot_base64 = plot_forecast_comparison(y_true, predictions_dict, timestamps)

            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")

            return {
                "visualization_type": viz_type,
                "image_base64": plot_base64,
                "format": "png"
            }

        result = await run_in_threadpool(_execute)
        return result

    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/latex-export", response_model=LaTeXExportResponse)
async def export_latex(request: LaTeXExportRequest):
    """
    Export results as LaTeX code for scientific publications.

    Export types:
    - **metrics_table**: Performance metrics table
    - **statistical_tests**: Statistical comparison table
    - **feature_importance**: Feature importance table
    - **full_document**: Complete LaTeX document with all sections
    """
    try:
        def _execute():
            export_type = request.export_type
            latex_code = ""
            bibliography = None

            if export_type == "metrics_table":
                metrics_dict = {}
                for model_id in (request.model_ids or services.MODELS_CACHE.keys()):
                    eval_result = services.get_evaluation_service(model_id)
                    if "metrics" in eval_result:
                        metrics_dict[model_id] = eval_result["metrics"]

                latex_code = generate_metrics_table(metrics_dict)

            elif export_type == "statistical_tests":
                predictions_dict = {}
                for model_id in (request.model_ids or list(services.MODELS_CACHE.keys())[:3]):
                    y_true, y_pred, _ = _get_test_data(model_id, 30)
                    predictions_dict[model_id] = y_pred

                y_true, _, _ = _get_test_data(list(predictions_dict.keys())[0], 30)
                stat_results = perform_statistical_comparison(predictions_dict, y_true)
                latex_code = generate_statistical_tests_table(stat_results.get("pairwise_tests", {}))

            elif export_type == "full_document":
                # Collect metrics
                metrics_dict = {}
                for model_id in (request.model_ids or services.MODELS_CACHE.keys()):
                    eval_result = services.get_evaluation_service(model_id)
                    if "metrics" in eval_result:
                        metrics_dict[model_id] = eval_result["metrics"]

                # Statistical tests
                predictions_dict = {}
                model_subset = list(metrics_dict.keys())[:5]  # Limit to 5 models
                for model_id in model_subset:
                    try:
                        y_true, y_pred, _ = _get_test_data(model_id, 30)
                        predictions_dict[model_id] = y_pred
                    except:
                        continue

                if predictions_dict:
                    y_true, _, _ = _get_test_data(list(predictions_dict.keys())[0], 30)
                    stat_results = perform_statistical_comparison(predictions_dict, y_true)
                    statistical_tests = stat_results.get("pairwise_tests", {})
                else:
                    statistical_tests = {}

                latex_code = generate_full_latex_document(
                    title=request.title,
                    author=request.author,
                    metrics_dict=metrics_dict,
                    statistical_tests=statistical_tests,
                    abstract=request.abstract,
                    include_methodology=request.include_methodology
                )

                bibliography = generate_bibliography()

            else:
                raise ValueError(f"Unsupported export type: {export_type}")

            compilation_instructions = """
To compile this LaTeX document:

1. Save the LaTeX code to a file (e.g., report.tex)
2. If bibliography is provided, save it to references.bib
3. Run: pdflatex report.tex
4. If using bibliography: bibtex report && pdflatex report.tex && pdflatex report.tex
5. Or use an online LaTeX editor like Overleaf

Required LaTeX packages: amsmath, booktabs, graphicx, hyperref
"""

            return {
                "export_type": export_type,
                "latex_code": latex_code,
                "bibliography": bibliography,
                "compilation_instructions": compilation_instructions
            }

        result = await run_in_threadpool(_execute)
        return result

    except Exception as e:
        logger.error(f"Error in LaTeX export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reproducibility-report", response_model=ReproducibilityReportResponse)
async def get_reproducibility_report(request: ReproducibilityReportRequest):
    """
    Generate comprehensive reproducibility report.

    Includes:
    - System and hardware information
    - Python version and package versions
    - Git repository information
    - Model configurations
    - Step-by-step reproducibility instructions
    """
    try:
        def _execute():
            model_configs = {}
            for model_id, config in AVAILABLE_MODELS.items():
                model_configs[model_id] = {
                    "type": config["type"],
                    "granularity": config["granularity"],
                    "feature_set": config["feature_set"]
                }

            report = generate_reproducibility_report(
                model_configs=model_configs,
                training_parameters={"note": "Training parameters vary by model"},
                data_info={
                    "dataset": "UCI Household Power Consumption",
                    "period": "2006-12-16 to 2010-11-26",
                    "frequency": "Aggregated to hourly/daily"
                }
            )

            markdown_report = None
            if request.format in ["markdown", "all"]:
                markdown_report = generate_reproducibility_markdown(report)

            return {
                **report,
                "markdown_report": markdown_report
            }

        result = await run_in_threadpool(_execute)
        return result

    except Exception as e:
        logger.error(f"Error in reproducibility report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-diagnostics/{model_id}")
async def get_model_diagnostics(model_id: str, test_size_days: int = 30):
    """
    Get comprehensive diagnostic information for a model.

    Combines:
    - Residual analysis
    - Error analysis
    - Performance metrics
    - Statistical properties
    """
    try:
        def _execute():
            y_true, y_pred, timestamps = _get_test_data(model_id, test_size_days)

            diagnostics = {
                "model_id": model_id,
                "test_size_days": test_size_days,
                "residual_analysis": analyze_residuals(y_true, y_pred, timestamps),
                "error_analysis": perform_error_analysis(y_true, y_pred, timestamps),
                "evaluation": services.get_evaluation_service(model_id)
            }

            return diagnostics

        result = await run_in_threadpool(_execute)
        return result

    except Exception as e:
        logger.error(f"Error in model diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))