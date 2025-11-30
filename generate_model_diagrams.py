"""
Generate model architecture diagrams for documentation.
Creates visual representations of ML/DL model architectures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def save_diagram(fig, filename):
    """Save figure with high quality."""
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {filename}")


def draw_lstm_architecture():
    """Draw LSTM model architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'LSTM Model Architecture', fontsize=16, fontweight='bold',
            ha='center', va='top')

    # Input layer
    input_box = FancyBboxPatch((0.5, 7.5), 2, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 7.9, 'Input\nSequence (24, features)', ha='center', va='center', fontsize=10)

    # LSTM Layer 1
    lstm1_box = FancyBboxPatch((0.5, 6), 2, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(lstm1_box)
    ax.text(1.5, 6.4, 'LSTM Layer 1\n128 units', ha='center', va='center', fontsize=10)

    # LSTM Layer 2
    lstm2_box = FancyBboxPatch((0.5, 4.5), 2, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(lstm2_box)
    ax.text(1.5, 4.9, 'LSTM Layer 2\n64 units', ha='center', va='center', fontsize=10)

    # Dropout
    dropout_box = FancyBboxPatch((0.5, 3), 2, 0.8, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(dropout_box)
    ax.text(1.5, 3.4, 'Dropout\n(0.2)', ha='center', va='center', fontsize=10)

    # Dense Layer
    dense_box = FancyBboxPatch((0.5, 1.5), 2, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(dense_box)
    ax.text(1.5, 1.9, 'Dense Layer\n32 units (ReLU)', ha='center', va='center', fontsize=10)

    # Output
    output_box = FancyBboxPatch((0.5, 0), 2, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightgray', linewidth=2)
    ax.add_patch(output_box)
    ax.text(1.5, 0.4, 'Output\n1 unit (Linear)', ha='center', va='center', fontsize=10)

    # Arrows
    arrows = [
        ((1.5, 7.5), (1.5, 6.8)),
        ((1.5, 6), (1.5, 5.3)),
        ((1.5, 4.5), (1.5, 3.8)),
        ((1.5, 3), (1.5, 2.3)),
        ((1.5, 1.5), (1.5, 0.8))
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', lw=2,
                               color='black', mutation_scale=20)
        ax.add_patch(arrow)

    # Legend
    legend_y = 7
    ax.text(4.5, legend_y + 1, 'Model Details:', fontsize=12, fontweight='bold')
    details = [
        'Total Parameters: ~500K',
        'Optimizer: Adam (lr=0.001)',
        'Loss: Mean Squared Error',
        'Sequence Length: 24 hours',
        'Batch Size: 32',
        'Training: 100 epochs'
    ]

    for i, detail in enumerate(details):
        ax.text(4.5, legend_y - i*0.5, f'• {detail}', fontsize=10)

    save_diagram(fig, 'docs/diagrams/lstm_architecture.png')


def draw_transformer_architecture():
    """Draw Transformer model architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'Transformer Model Architecture', fontsize=16, fontweight='bold',
            ha='center', va='top')

    # Input
    input_box = FancyBboxPatch((0.5, 10), 2, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 10.4, 'Input Embedding\n+ Positional Encoding', ha='center', va='center', fontsize=9)

    # Multi-Head Attention
    mha_box = FancyBboxPatch((0.5, 8.5), 2, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#FFB6C1', linewidth=2)
    ax.add_patch(mha_box)
    ax.text(1.5, 8.9, 'Multi-Head\nAttention (4 heads)', ha='center', va='center', fontsize=9)

    # Add & Norm 1
    norm1_box = FancyBboxPatch((0.5, 7.5), 2, 0.5, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(norm1_box)
    ax.text(1.5, 7.75, 'Add & Normalize', ha='center', va='center', fontsize=9)

    # Feed Forward
    ff_box = FancyBboxPatch((0.5, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(ff_box)
    ax.text(1.5, 6.9, 'Feed Forward\n128 → 64', ha='center', va='center', fontsize=9)

    # Add & Norm 2
    norm2_box = FancyBboxPatch((0.5, 5.5), 2, 0.5, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(norm2_box)
    ax.text(1.5, 5.75, 'Add & Normalize', ha='center', va='center', fontsize=9)

    # Global Average Pooling
    pool_box = FancyBboxPatch((0.5, 4.2), 2, 0.8, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='lightcyan', linewidth=2)
    ax.add_patch(pool_box)
    ax.text(1.5, 4.6, 'Global Average\nPooling', ha='center', va='center', fontsize=9)

    # Dense
    dense_box = FancyBboxPatch((0.5, 2.9), 2, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(dense_box)
    ax.text(1.5, 3.3, 'Dense (32)\nReLU', ha='center', va='center', fontsize=9)

    # Output
    output_box = FancyBboxPatch((0.5, 1.6), 2, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightgray', linewidth=2)
    ax.add_patch(output_box)
    ax.text(1.5, 2, 'Output\n1 unit', ha='center', va='center', fontsize=9)

    # Arrows
    y_positions = [10, 8.5, 7.5, 6.5, 5.5, 4.2, 2.9, 1.6]
    for i in range(len(y_positions) - 1):
        arrow = FancyArrowPatch((1.5, y_positions[i]), (1.5, y_positions[i+1] + 0.8),
                               arrowstyle='->', lw=2, color='black', mutation_scale=20)
        ax.add_patch(arrow)

    # Skip connections
    skip1 = FancyArrowPatch((2.6, 10), (2.6, 7.75), arrowstyle='->', lw=1.5,
                           color='red', linestyle='--', mutation_scale=15)
    ax.add_patch(skip1)
    ax.text(3, 8.5, 'Skip', fontsize=8, color='red', rotation=90, va='center')

    skip2 = FancyArrowPatch((2.9, 7.5), (2.9, 5.75), arrowstyle='->', lw=1.5,
                           color='red', linestyle='--', mutation_scale=15)
    ax.add_patch(skip2)
    ax.text(3.3, 6.5, 'Skip', fontsize=8, color='red', rotation=90, va='center')

    # Details
    legend_y = 9
    ax.text(4.5, legend_y + 1, 'Model Details:', fontsize=12, fontweight='bold')
    details = [
        'Attention Heads: 4',
        'Model Dimension: 64',
        'FF Dimension: 128',
        'Dropout: 0.1',
        'Total Parameters: ~400K',
        'Optimizer: Adam',
        'Sequence Length: 24'
    ]

    for i, detail in enumerate(details):
        ax.text(4.5, legend_y - i*0.5, f'• {detail}', fontsize=10)

    save_diagram(fig, 'docs/diagrams/transformer_architecture.png')


def draw_xgboost_flow():
    """Draw XGBoost model flow diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'XGBoost Model Pipeline', fontsize=16, fontweight='bold',
            ha='center', va='top')

    # Data Input
    data_box = FancyBboxPatch((1, 7.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(data_box)
    ax.text(2.25, 7.9, 'Historical Data\n(Hourly)', ha='center', va='center', fontsize=10)

    # Feature Engineering
    feat_box = FancyBboxPatch((5, 7.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(feat_box)
    ax.text(6.25, 7.9, 'Feature Engineering\n50+ features', ha='center', va='center', fontsize=10)

    # XGBoost Training
    xgb_box = FancyBboxPatch((9, 7.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(xgb_box)
    ax.text(10.25, 7.9, 'XGBoost\nGradient Boosting', ha='center', va='center', fontsize=10)

    # Feature categories
    categories = [
        ('Temporal', 'hour, day, month, etc.', 5.5),
        ('Lagged', 'lag_1, lag_24, lag_168', 4.5),
        ('Rolling', 'roll_mean, roll_std', 3.5),
        ('Weather', 'temperature, humidity', 2.5),
        ('Calendar', 'holidays, weekends', 1.5)
    ]

    for i, (cat, desc, y) in enumerate(categories):
        box = FancyBboxPatch((0.5, y), 3, 0.6, boxstyle="round,pad=0.05",
                            edgecolor='gray', facecolor='white', linewidth=1.5)
        ax.add_patch(box)
        ax.text(2, y + 0.3, f'{cat}: {desc}', ha='center', va='center', fontsize=9)

    # Model details
    details = [
        ('n_estimators', '100 trees', 5.5),
        ('max_depth', '6', 4.5),
        ('learning_rate', '0.1', 3.5),
        ('subsample', '0.8', 2.5),
        ('colsample_bytree', '0.8', 1.5)
    ]

    for i, (param, value, y) in enumerate(details):
        ax.text(9.5, y, f'{param}: {value}', fontsize=9)

    # Arrows
    arrow1 = FancyArrowPatch((3.5, 7.9), (5, 7.9), arrowstyle='->', lw=2,
                            color='black', mutation_scale=20)
    ax.add_patch(arrow1)

    arrow2 = FancyArrowPatch((7.5, 7.9), (9, 7.9), arrowstyle='->', lw=2,
                            color='black', mutation_scale=20)
    ax.add_patch(arrow2)

    # Connect features to feature engineering
    for _, _, y in categories:
        arrow = FancyArrowPatch((3.5, y + 0.3), (5, 7.5), arrowstyle='->', lw=1,
                               color='gray', alpha=0.5, mutation_scale=15)
        ax.add_patch(arrow)

    save_diagram(fig, 'docs/diagrams/xgboost_pipeline.png')


def draw_ensemble_architecture():
    """Draw ensemble (Stacking/Voting) architecture."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'Stacking Ensemble Architecture', fontsize=16, fontweight='bold',
            ha='center', va='top')

    # Input
    input_box = FancyBboxPatch((5, 8), 2, 0.6, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 8.3, 'Input Features', ha='center', va='center', fontsize=10)

    # Base models
    base_models = [
        ('XGBoost', 1, 6),
        ('LightGBM', 4, 6),
        ('Random Forest', 7, 6),
        ('Linear Reg', 10, 6)
    ]

    colors = ['#90EE90', '#FFB6C1', '#ADD8E6', '#FFD700']

    for i, (name, x, y) in enumerate(base_models):
        box = FancyBboxPatch((x, y), 2, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=colors[i], linewidth=2)
        ax.add_patch(box)
        ax.text(x + 1, y + 0.4, name, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow from input to base model
        arrow = FancyArrowPatch((6, 8), (x + 1, y + 0.8), arrowstyle='->', lw=1.5,
                               color='gray', mutation_scale=15)
        ax.add_patch(arrow)

    # Meta predictions
    meta_y = 4.5
    for i, (name, x, y) in enumerate(base_models):
        box = FancyBboxPatch((x, meta_y), 2, 0.5, boxstyle="round,pad=0.05",
                            edgecolor='gray', facecolor='white', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 1, meta_y + 0.25, f'Pred {i+1}', ha='center', va='center', fontsize=8)

        # Arrow from base to meta pred
        arrow = FancyArrowPatch((x + 1, y), (x + 1, meta_y + 0.5), arrowstyle='->', lw=1.5,
                               color='black', mutation_scale=15)
        ax.add_patch(arrow)

    # Meta-learner
    meta_box = FancyBboxPatch((4.5, 2.5), 3, 0.8, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(meta_box)
    ax.text(6, 2.9, 'Meta-Learner\n(Ridge Regression)', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Arrows to meta-learner
    for i, (name, x, y) in enumerate(base_models):
        arrow = FancyArrowPatch((x + 1, meta_y), (6, 3.3), arrowstyle='->', lw=1.5,
                               color='black', alpha=0.7, mutation_scale=15)
        ax.add_patch(arrow)

    # Final output
    output_box = FancyBboxPatch((4.5, 1), 3, 0.6, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='lightgray', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 1.3, 'Final Prediction', ha='center', va='center', fontsize=10)

    arrow = FancyArrowPatch((6, 2.5), (6, 1.6), arrowstyle='->', lw=2,
                           color='black', mutation_scale=20)
    ax.add_patch(arrow)

    save_diagram(fig, 'docs/diagrams/ensemble_architecture.png')


def draw_complete_system():
    """Draw complete system architecture."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(7, 11.5, 'Energy Forecast API - System Architecture', fontsize=18,
            fontweight='bold', ha='center', va='top')

    # Frontend
    frontend = FancyBboxPatch((1, 9), 3, 1.5, boxstyle="round,pad=0.15",
                             edgecolor='#2196F3', facecolor='#E3F2FD', linewidth=3)
    ax.add_patch(frontend)
    ax.text(2.5, 9.75, 'Frontend\n(React + TypeScript)\nVercel', ha='center',
            va='center', fontsize=10, fontweight='bold')

    # API Gateway
    api_gw = FancyBboxPatch((6, 9), 2, 1.5, boxstyle="round,pad=0.15",
                           edgecolor='#4CAF50', facecolor='#E8F5E9', linewidth=3)
    ax.add_patch(api_gw)
    ax.text(7, 9.75, 'FastAPI\nREST API', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # HF Hub
    hf_hub = FancyBboxPatch((10.5, 9), 2.5, 1.5, boxstyle="round,pad=0.15",
                           edgecolor='#FF9800', facecolor='#FFF3E0', linewidth=3)
    ax.add_patch(hf_hub)
    ax.text(11.75, 9.75, 'Hugging Face\nModel Hub', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Model categories
    model_categories = [
        ('Classical Models\nARIMA, SARIMA, Prophet', 1, 6, '#E1BEE7'),
        ('ML Models\nXGB, LightGBM, RF', 4.5, 6, '#C8E6C9'),
        ('DL Models\nLSTM, GRU, Transformer', 8, 6, '#FFCCBC'),
        ('Ensemble\nVoting, Stacking', 11.5, 6, '#B3E5FC')
    ]

    for desc, x, y, color in model_categories:
        box = FancyBboxPatch((x, y), 2.2, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.6, desc, ha='center', va='center', fontsize=8.5)

    # Data & Features
    data_box = FancyBboxPatch((1, 3.5), 5, 1.5, boxstyle="round,pad=0.15",
                             edgecolor='#9C27B0', facecolor='#F3E5F5', linewidth=3)
    ax.add_patch(data_box)
    ax.text(3.5, 4.25, 'Historical Data & Feature Engineering\n' +
            'UCI Dataset • Temporal • Weather • Lags • Rolling Stats',
            ha='center', va='center', fontsize=9)

    # Scientific Analysis
    sci_box = FancyBboxPatch((7.5, 3.5), 5.5, 1.5, boxstyle="round,pad=0.15",
                            edgecolor='#F44336', facecolor='#FFEBEE', linewidth=3)
    ax.add_patch(sci_box)
    ax.text(10.25, 4.25, 'Scientific Analysis Module\n' +
            'Statistical Tests • Residuals • LaTeX • Viz • Sensitivity',
            ha='center', va='center', fontsize=9)

    # Outputs
    outputs = [
        ('Predictions', 1.5, 1, '#E8F5E9'),
        ('Metrics', 4, 1, '#FFF3E0'),
        ('Visualizations', 6.5, 1, '#E3F2FD'),
        ('LaTeX Reports', 9, 1, '#FFEBEE'),
        ('Reproducibility', 11.5, 1, '#F3E5F5')
    ]

    for name, x, y, color in outputs:
        box = FancyBboxPatch((x, y), 2, 0.7, boxstyle="round,pad=0.05",
                            edgecolor='gray', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 1, y + 0.35, name, ha='center', va='center', fontsize=8)

    # Arrows (simplified)
    # Frontend -> API
    arrow1 = FancyArrowPatch((4, 9.75), (6, 9.75), arrowstyle='<->', lw=2.5,
                            color='#2196F3', mutation_scale=25)
    ax.add_patch(arrow1)

    # API -> Models
    arrow2 = FancyArrowPatch((7, 9), (7, 7.2), arrowstyle='->', lw=2,
                            color='black', mutation_scale=20)
    ax.add_patch(arrow2)

    # HF Hub -> Models
    arrow3 = FancyArrowPatch((11.75, 9), (7, 7.2), arrowstyle='->', lw=2,
                            color='#FF9800', linestyle='--', mutation_scale=20)
    ax.add_patch(arrow3)
    ax.text(9.5, 8.5, 'Model\nDownload', fontsize=7, ha='center')

    # Data -> Models
    arrow4 = FancyArrowPatch((3.5, 5), (5, 6), arrowstyle='->', lw=2,
                            color='#9C27B0', mutation_scale=20)
    ax.add_patch(arrow4)

    # Models -> Scientific
    arrow5 = FancyArrowPatch((8, 6), (9, 5), arrowstyle='->', lw=2,
                            color='#F44336', mutation_scale=20)
    ax.add_patch(arrow5)

    # Scientific -> Outputs
    arrow6 = FancyArrowPatch((10.25, 3.5), (7, 1.7), arrowstyle='->', lw=2,
                            color='black', mutation_scale=20)
    ax.add_patch(arrow6)

    save_diagram(fig, 'docs/diagrams/system_architecture.png')


if __name__ == "__main__":
    import os

    # Create output directory
    os.makedirs('docs/diagrams', exist_ok=True)

    print("🎨 Generating model architecture diagrams...")

    draw_lstm_architecture()
    draw_transformer_architecture()
    draw_xgboost_flow()
    draw_ensemble_architecture()
    draw_complete_system()

    print("\n✅ All diagrams generated successfully!")
    print("📁 Location: docs/diagrams/")
