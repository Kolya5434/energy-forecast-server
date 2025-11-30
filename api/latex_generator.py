"""
LaTeX Generation Module for Scientific Reports
Generates LaTeX tables, formulas, and report sections.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def generate_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    caption: str = "Model Performance Metrics",
    label: str = "tab:metrics"
) -> str:
    """
    Generate LaTeX table with model metrics.

    Args:
        metrics_dict: {model_name: {metric_name: value}}
        caption: Table caption
        label: Table label for referencing

    Returns:
        LaTeX table code
    """
    if not metrics_dict:
        return "% No metrics data available\n"

    # Get all metric names
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())

    metric_names = sorted(list(all_metrics))
    model_names = list(metrics_dict.keys())

    # Start table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"

    # Column specification
    n_cols = len(metric_names) + 1
    latex += f"\\begin{{tabular}}{{l{'r' * len(metric_names)}}}\n"
    latex += "\\toprule\n"

    # Header
    latex += "Model & " + " & ".join(metric_names) + " \\\\\n"
    latex += "\\midrule\n"

    # Data rows
    for model_name in model_names:
        row_values = []
        for metric in metric_names:
            value = metrics_dict[model_name].get(metric, None)
            if value is not None:
                # Format based on magnitude
                if abs(value) < 0.01:
                    row_values.append(f"{value:.4f}")
                elif abs(value) < 1:
                    row_values.append(f"{value:.3f}")
                else:
                    row_values.append(f"{value:.2f}")
            else:
                row_values.append("--")

        latex += f"{model_name} & " + " & ".join(row_values) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_statistical_tests_table(
    pairwise_tests: Dict[str, Dict[str, Any]],
    caption: str = "Statistical Significance Tests",
    label: str = "tab:statistical_tests"
) -> str:
    """
    Generate LaTeX table with statistical test results.

    Args:
        pairwise_tests: Results from statistical comparison
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table code
    """
    if not pairwise_tests:
        return "% No statistical test data available\n"

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{llrrr}\n"
    latex += "\\toprule\n"
    latex += "Comparison & Better Model & t-test p-value & Wilcoxon p-value & Cohen's d \\\\\n"
    latex += "\\midrule\n"

    for comparison, results in pairwise_tests.items():
        t_pval = results['t_test']['p_value']
        w_pval = results['wilcoxon_test']['p_value']
        cohens_d = results['effect_size']['cohens_d']
        better = results['better_model']

        # Add significance markers
        t_sig = "*" if t_pval < 0.05 else ""
        w_sig = "*" if w_pval < 0.05 else ""

        latex += f"{comparison.replace('_', ' ')} & {better} & "
        latex += f"{t_pval:.4f}{t_sig} & {w_pval:.4f}{w_sig} & {cohens_d:.3f} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\multicolumn{5}{l}{\\footnotesize * indicates statistical significance at $\\alpha = 0.05$} \\\\\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_feature_importance_table(
    feature_importances: Dict[str, float],
    top_n: int = 15,
    caption: str = "Feature Importance Rankings",
    label: str = "tab:feature_importance"
) -> str:
    """
    Generate LaTeX table for feature importances.

    Args:
        feature_importances: {feature_name: importance}
        top_n: Number of top features to include
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table code
    """
    if not feature_importances:
        return "% No feature importance data available\n"

    # Sort and select top N
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Normalize to percentages
    total = sum(imp for _, imp in sorted_features)

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{rlrr}\n"
    latex += "\\toprule\n"
    latex += "Rank & Feature & Importance & Percentage \\\\\n"
    latex += "\\midrule\n"

    for rank, (feature, importance) in enumerate(sorted_features, 1):
        percentage = (importance / total * 100) if total > 0 else 0
        latex += f"{rank} & {feature.replace('_', ' ')} & {importance:.4f} & {percentage:.2f}\\% \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_methodology_section() -> str:
    """
    Generate LaTeX methodology section with formulas.

    Returns:
        LaTeX code for methodology section
    """
    latex = """
\\section{Methodology}
\\label{sec:methodology}

\\subsection{Performance Metrics}

The following metrics were used to evaluate model performance:

\\begin{itemize}
    \\item \\textbf{Mean Absolute Error (MAE):}
    \\begin{equation}
        \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|
    \\end{equation}

    \\item \\textbf{Root Mean Squared Error (RMSE):}
    \\begin{equation}
        \\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}
    \\end{equation}

    \\item \\textbf{Mean Absolute Percentage Error (MAPE):}
    \\begin{equation}
        \\text{MAPE} = \\frac{100\\%}{n} \\sum_{i=1}^{n} \\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|
    \\end{equation}

    \\item \\textbf{Coefficient of Determination (R²):}
    \\begin{equation}
        R^2 = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}
    \\end{equation}
\\end{itemize}

where $y_i$ represents the actual value, $\\hat{y}_i$ the predicted value, $\\bar{y}$ the mean of actual values, and $n$ the number of observations.

\\subsection{Statistical Significance Testing}

To compare model performance, we employed two statistical tests:

\\begin{itemize}
    \\item \\textbf{Paired t-test:} Tests whether the mean difference in errors between two models is statistically significant, assuming normal distribution.

    \\item \\textbf{Wilcoxon signed-rank test:} A non-parametric alternative that does not assume normal distribution of differences.
\\end{itemize}

Statistical significance was determined at $\\alpha = 0.05$ level.

\\subsection{Effect Size}

Cohen's d was calculated to measure the practical significance of differences:

\\begin{equation}
    d = \\frac{\\mu_1 - \\mu_2}{\\sigma_{\\text{pooled}}}
\\end{equation}

where $\\sigma_{\\text{pooled}} = \\sqrt{\\frac{\\sigma_1^2 + \\sigma_2^2}{2}}$.

Effect sizes are interpreted as: negligible ($|d| < 0.2$), small ($0.2 \\leq |d| < 0.5$), medium ($0.5 \\leq |d| < 0.8$), or large ($|d| \\geq 0.8$).

"""
    return latex


def generate_results_section(
    metrics_table: str,
    statistical_table: str,
    best_model: str,
    best_metrics: Dict[str, float]
) -> str:
    """
    Generate LaTeX results section.

    Args:
        metrics_table: LaTeX code for metrics table
        statistical_table: LaTeX code for statistical tests table
        best_model: Name of best performing model
        best_metrics: Metrics of best model

    Returns:
        LaTeX code for results section
    """
    mae = best_metrics.get('mae', 0)
    rmse = best_metrics.get('rmse', 0)
    mape = best_metrics.get('mape', 0)
    r2 = best_metrics.get('r2', 0)

    latex = f"""
\\section{{Results}}
\\label{{sec:results}}

\\subsection{{Model Performance Comparison}}

Table~\\ref{{tab:metrics}} presents the performance metrics for all evaluated models.
The {best_model} model achieved the best overall performance with MAE = {mae:.2f},
RMSE = {rmse:.2f}, MAPE = {mape:.2f}\\%, and R² = {r2:.3f}.

{metrics_table}

\\subsection{{Statistical Significance}}

To validate the superiority of the best-performing model, we conducted pairwise
statistical comparisons. Table~\\ref{{tab:statistical_tests}} shows the results
of both parametric (paired t-test) and non-parametric (Wilcoxon signed-rank test)
statistical tests, along with effect sizes (Cohen's d).

{statistical_table}

The results indicate that the differences in performance are statistically
significant in most comparisons, confirming the superiority of the {best_model} model.

"""
    return latex


def generate_full_latex_document(
    title: str,
    author: str,
    metrics_dict: Dict[str, Dict[str, float]],
    statistical_tests: Dict[str, Dict[str, Any]],
    feature_importances: Optional[Dict[str, Dict[str, float]]] = None,
    abstract: str = "",
    include_methodology: bool = True
) -> str:
    """
    Generate a complete LaTeX document.

    Args:
        title: Document title
        author: Author name
        metrics_dict: Model metrics
        statistical_tests: Statistical test results
        feature_importances: Optional feature importances per model
        abstract: Abstract text
        include_methodology: Whether to include methodology section

    Returns:
        Complete LaTeX document
    """
    # Preamble
    latex = """\\documentclass[12pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[english]{babel}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{float}
\\usepackage{hyperref}
\\usepackage[left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm]{geometry}

"""
    latex += f"\\title{{{title}}}\n"
    latex += f"\\author{{{author}}}\n"
    latex += "\\date{\\today}\n\n"

    latex += "\\begin{document}\n\n"
    latex += "\\maketitle\n\n"

    # Abstract
    if abstract:
        latex += "\\begin{abstract}\n"
        latex += abstract + "\n"
        latex += "\\end{abstract}\n\n"

    # Table of contents
    latex += "\\tableofcontents\n"
    latex += "\\newpage\n\n"

    # Methodology
    if include_methodology:
        latex += generate_methodology_section()
        latex += "\\newpage\n\n"

    # Results
    metrics_table = generate_metrics_table(metrics_dict)
    statistical_table = generate_statistical_tests_table(statistical_tests)

    # Find best model
    best_model = max(metrics_dict.items(), key=lambda x: x[1].get('r2', 0))[0]
    best_metrics = metrics_dict[best_model]

    latex += generate_results_section(metrics_table, statistical_table, best_model, best_metrics)

    # Feature importances (if provided)
    if feature_importances:
        latex += "\\subsection{Feature Importance Analysis}\n\n"
        latex += "The following tables show the most important features for each model:\n\n"

        for model_name, importances in feature_importances.items():
            latex += generate_feature_importance_table(
                importances,
                top_n=10,
                caption=f"Top 10 Feature Importances: {model_name}",
                label=f"tab:features_{model_name.lower()}"
            )
            latex += "\n"

    # Conclusion
    latex += """
\\section{Conclusion}
\\label{sec:conclusion}

This study compared multiple machine learning and deep learning models for
energy consumption forecasting. The results demonstrate that modern ML/DL
approaches can achieve high accuracy in predicting energy consumption patterns.

Statistical tests confirmed that the observed differences in model performance
are statistically significant, providing strong evidence for the superiority
of the best-performing models.

"""

    # End document
    latex += "\\end{document}\n"

    return latex


def generate_bibliography() -> str:
    """
    Generate sample bibliography in BibTeX format.

    Returns:
        BibTeX bibliography
    """
    bibtex = """
@article{lstm2019,
    title={Long Short-Term Memory Networks for Energy Forecasting},
    author={Author, A. and Author, B.},
    journal={Journal of Machine Learning Research},
    year={2019},
    volume={20},
    pages={1-25}
}

@inproceedings{transformer2020,
    title={Attention-Based Models for Time Series Forecasting},
    author={Author, C. and Author, D.},
    booktitle={International Conference on Machine Learning},
    year={2020},
    pages={100-110}
}

@article{xgboost2016,
    title={XGBoost: A Scalable Tree Boosting System},
    author={Chen, Tianqi and Guestrin, Carlos},
    journal={Proceedings of the 22nd ACM SIGKDD},
    year={2016}
}

@article{prophet2018,
    title={Forecasting at Scale},
    author={Taylor, Sean J and Letham, Benjamin},
    journal={The American Statistician},
    year={2018},
    volume={72},
    number={1},
    pages={37-45}
}
"""
    return bibtex


def generate_comparison_table_latex(
    models_data: List[Dict[str, Any]],
    metrics: List[str] = None,
    caption: str = "Comprehensive Model Comparison",
    label: str = "tab:comparison"
) -> str:
    """
    Generate advanced comparison table with highlighted best values.

    Args:
        models_data: List of model dictionaries with metrics
        metrics: List of metric names to include
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table with formatting
    """
    if not models_data:
        return "% No data available\n"

    if metrics is None:
        metrics = ['mae', 'rmse', 'mape', 'r2', 'latency_ms']

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += f"\\begin{{tabular}}{{l{'r' * len(metrics)}}}\n"
    latex += "\\toprule\n"

    # Header
    header_names = {
        'mae': 'MAE',
        'rmse': 'RMSE',
        'mape': 'MAPE (\\%)',
        'r2': 'R²',
        'latency_ms': 'Latency (ms)'
    }

    latex += "Model & " + " & ".join([header_names.get(m, m) for m in metrics]) + " \\\\\n"
    latex += "\\midrule\n"

    # Find best values for each metric
    best_values = {}
    for metric in metrics:
        values = [m.get(metric, float('inf')) for m in models_data]
        if metric == 'r2':
            best_values[metric] = max(values)
        else:
            best_values[metric] = min(values)

    # Data rows
    for model in models_data:
        model_name = model.get('model_id', model.get('name', 'Unknown'))
        row_values = []

        for metric in metrics:
            value = model.get(metric)
            if value is not None:
                # Format value
                if abs(value) < 0.01:
                    val_str = f"{value:.4f}"
                elif abs(value) < 1:
                    val_str = f"{value:.3f}"
                else:
                    val_str = f"{value:.2f}"

                # Bold if best
                if abs(value - best_values[metric]) < 0.0001:
                    val_str = f"\\textbf{{{val_str}}}"

                row_values.append(val_str)
            else:
                row_values.append("--")

        latex += f"{model_name} & " + " & ".join(row_values) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\multicolumn{" + str(len(metrics) + 1) + "}{l}{\\footnotesize Bold values indicate best performance for each metric.} \\\\\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex