"""
Pydantic schemas for scientific analysis endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class StatisticalTestRequest(BaseModel):
    """Request for statistical significance testing."""
    model_ids: List[str] = Field(..., description="List of model IDs to compare")
    test_size_days: int = Field(default=30, description="Number of days for test set")

    class Config:
        json_schema_extra = {
            "example": {
                "model_ids": ["LSTM", "XGBoost", "ARIMA"],
                "test_size_days": 30
            }
        }


class ResidualAnalysisRequest(BaseModel):
    """Request for residual analysis."""
    model_id: str = Field(..., description="Model ID to analyze")
    test_size_days: int = Field(default=30, description="Number of days for test set")
    include_plots: bool = Field(default=True, description="Include visualization plots")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "LSTM",
                "test_size_days": 30,
                "include_plots": True
            }
        }


class ErrorAnalysisRequest(BaseModel):
    """Request for error analysis."""
    model_id: str = Field(..., description="Model ID to analyze")
    test_size_days: int = Field(default=30, description="Number of days for test set")
    include_temporal: bool = Field(default=True, description="Include temporal error patterns")
    include_plots: bool = Field(default=True, description="Include visualization plots")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "XGBoost",
                "test_size_days": 30,
                "include_temporal": True,
                "include_plots": True
            }
        }


class VisualizationRequest(BaseModel):
    """Request for scientific visualizations."""
    visualization_type: str = Field(
        ...,
        description="Type of visualization: 'residuals', 'error_distribution', 'comparison', 'forecast', 'temporal_error', 'feature_importance', 'correlation'"
    )
    model_ids: Optional[List[str]] = Field(default=None, description="Model IDs (required for some viz types)")
    test_size_days: int = Field(default=30, description="Number of days for test set")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "visualization_type": "comparison",
                "model_ids": ["LSTM", "XGBoost", "ARIMA"],
                "test_size_days": 30
            }
        }


class LaTeXExportRequest(BaseModel):
    """Request for LaTeX export."""
    export_type: str = Field(
        ...,
        description="Type of export: 'metrics_table', 'statistical_tests', 'feature_importance', 'full_document'"
    )
    model_ids: Optional[List[str]] = Field(default=None, description="Model IDs to include")
    include_methodology: bool = Field(default=True, description="Include methodology section (for full document)")
    title: Optional[str] = Field(default="Energy Forecasting Study", description="Document title")
    author: Optional[str] = Field(default="Author Name", description="Author name")
    abstract: Optional[str] = Field(default="", description="Abstract text")

    class Config:
        json_schema_extra = {
            "example": {
                "export_type": "full_document",
                "model_ids": ["LSTM", "XGBoost", "ARIMA"],
                "include_methodology": True,
                "title": "Hybrid ML/DL Energy Forecasting System",
                "author": "Mykola"
            }
        }


class SensitivityAnalysisRequest(BaseModel):
    """Request for sensitivity analysis."""
    analysis_type: str = Field(
        ...,
        description="Type: 'feature_ablation', 'input_perturbation', 'hyperparameter', 'noise_robustness'"
    )
    model_id: str = Field(..., description="Model ID to analyze")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Analysis-specific parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_type": "feature_ablation",
                "model_id": "XGBoost",
                "parameters": {
                    "feature_groups": {
                        "weather": ["temperature", "humidity"],
                        "calendar": ["is_holiday", "is_weekend"]
                    }
                }
            }
        }


class CrossValidationRequest(BaseModel):
    """Request for cross-validation analysis."""
    model_id: str = Field(..., description="Model ID to validate")
    n_splits: int = Field(default=5, ge=2, le=10, description="Number of CV splits")
    strategy: str = Field(default="timeseries", description="Split strategy: 'timeseries' or 'kfold'")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "XGBoost",
                "n_splits": 5,
                "strategy": "timeseries"
            }
        }


class ReproducibilityReportRequest(BaseModel):
    """Request for reproducibility report."""
    include_git: bool = Field(default=True, description="Include git information")
    include_system: bool = Field(default=True, description="Include system information")
    include_packages: bool = Field(default=True, description="Include package versions")
    format: str = Field(default="json", description="Output format: 'json', 'markdown', or 'all'")

    class Config:
        json_schema_extra = {
            "example": {
                "include_git": True,
                "include_system": True,
                "include_packages": True,
                "format": "markdown"
            }
        }


class ComparisonVisualizationRequest(BaseModel):
    """Request for model comparison visualizations."""
    model_ids: List[str] = Field(..., min_length=2, description="Models to compare (min 2)")
    metrics: Optional[List[str]] = Field(
        default=["mae", "rmse", "mape", "r2"],
        description="Metrics to include"
    )
    visualization_types: List[str] = Field(
        default=["radar", "bar", "box"],
        description="Types of plots to generate"
    )
    test_size_days: int = Field(default=30, description="Test set size")

    class Config:
        json_schema_extra = {
            "example": {
                "model_ids": ["LSTM", "XGBoost", "ARIMA", "Prophet"],
                "metrics": ["mae", "rmse", "r2"],
                "visualization_types": ["radar", "box"],
                "test_size_days": 30
            }
        }


# Response models

class StatisticalTestResponse(BaseModel):
    """Response with statistical test results."""
    num_models: int
    sample_size: int
    pairwise_tests: Dict[str, Dict[str, Any]]
    friedman_test: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None


class ResidualAnalysisResponse(BaseModel):
    """Response with residual analysis results."""
    model_id: str
    basic_statistics: Dict[str, float]
    normality: Dict[str, Any]
    autocorrelation: Dict[str, Optional[float]]
    heteroscedasticity: Dict[str, Any]
    percentiles: Dict[str, float]
    plots: Optional[Dict[str, str]] = None  # Base64 encoded plots


class ErrorAnalysisResponse(BaseModel):
    """Response with error analysis results."""
    model_id: str
    metrics: Dict[str, float]
    error_distribution: Dict[str, Any]
    large_errors: Dict[str, Any]
    temporal_patterns: Optional[Dict[str, Any]] = None
    feature_correlations: Optional[Dict[str, float]] = None
    plots: Optional[Dict[str, str]] = None


class VisualizationResponse(BaseModel):
    """Response with visualization."""
    visualization_type: str
    image_base64: str
    format: str = "png"
    metadata: Optional[Dict[str, Any]] = None


class LaTeXExportResponse(BaseModel):
    """Response with LaTeX code."""
    export_type: str
    latex_code: str
    bibliography: Optional[str] = None
    compilation_instructions: Optional[str] = None


class SensitivityAnalysisResponse(BaseModel):
    """Response with sensitivity analysis results."""
    analysis_type: str
    model_id: str
    results: Dict[str, Any]
    summary: Optional[Dict[str, Any]] = None
    plots: Optional[Dict[str, str]] = None


class ReproducibilityReportResponse(BaseModel):
    """Response with reproducibility report."""
    metadata: Dict[str, Any]
    system_information: Dict[str, Any]
    software_environment: Dict[str, Any]
    model_configurations: Optional[Dict[str, Any]] = None
    reproducibility_instructions: Dict[str, str]
    markdown_report: Optional[str] = None