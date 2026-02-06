"""
Visualization utilities for experiment results.

Creates figures for the paper.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def plot_euclidean_vs_hyperbolic(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    title: str = "Euclidean vs Hyperbolic Probes",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot comparison of Euclidean vs Hyperbolic probes across layers.
    
    Args:
        results: List of experiment result dicts
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    euclidean = [r for r in results if r["probe_type"] == "euclidean"]
    hyperbolic = [r for r in results if r["probe_type"] == "hyperbolic"]
    
    layers_e = [r["layer"] for r in euclidean]
    layers_h = [r["layer"] for r in hyperbolic]
    rho_e = [r["spearman_rho"] for r in euclidean]
    rho_h = [r["spearman_rho"] for r in hyperbolic]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(layers_e))
    width = 0.35
    
    bars_e = ax.bar(x - width/2, rho_e, width, label='Euclidean', color='#3498db', alpha=0.8)
    bars_h = ax.bar(x + width/2, rho_h, width, label='Hyperbolic', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Spearman ρ')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers_e])
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars_e + bars_h:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_token_ablation(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot token selection method comparison.
    """
    methods = list(set(r["selection_method"] for r in results))
    avg_rho = {m: np.mean([r["spearman_rho"] for r in results if r["selection_method"] == m]) 
               for m in methods}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(list(avg_rho.keys()), list(avg_rho.values()), color=colors[:len(methods)])
    
    ax.set_xlabel('Token Selection Method')
    ax.set_ylabel('Average Spearman ρ')
    ax.set_title('Token Selection Strategy Comparison')
    ax.set_ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_layer_sweep(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot metrics across all layers.
    """
    layers = sorted(set(r["layer"] for r in results))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for probe_type, color in [("euclidean", "#3498db"), ("hyperbolic", "#e74c3c")]:
        subset = [r for r in results if r["probe_type"] == probe_type]
        subset = sorted(subset, key=lambda x: x["layer"])
        
        rho = [r["spearman_rho"] for r in subset]
        distortion = [r["avg_distortion"] for r in subset]
        layer_vals = [r["layer"] for r in subset]
        
        axes[0].plot(layer_vals, rho, 'o-', color=color, label=probe_type.title(), linewidth=2)
        axes[1].plot(layer_vals, distortion, 's-', color=color, label=probe_type.title(), linewidth=2)
    
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Spearman ρ')
    axes[0].set_title('Distance Correlation by Layer')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Average Distortion')
    axes[1].set_title('Distortion by Layer')
    axes[1].legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results_by_model: Dict[str, List[Dict[str, Any]]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Compare results across different models.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results_by_model.keys())
    probe_types = ["euclidean", "hyperbolic"]
    
    x = np.arange(len(models))
    width = 0.35
    
    colors = {"euclidean": "#3498db", "hyperbolic": "#e74c3c"}
    
    for i, probe_type in enumerate(probe_types):
        vals = []
        for model in models:
            subset = [r for r in results_by_model[model] if r["probe_type"] == probe_type]
            avg = np.mean([r["spearman_rho"] for r in subset]) if subset else 0
            vals.append(avg)
        
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=probe_type.title(), color=colors[probe_type])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Spearman ρ')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_poincare_disk(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    depths: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "Poincaré Disk Embedding",
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot 2D embeddings on a Poincaré disk.
    
    Args:
        embeddings: [n_samples, 2] Poincaré ball embeddings
        labels: Optional labels for coloring (TRUE/FALSE)
        depths: Optional depth values for coloring
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw unit disk boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.05, color='gray')
    
    # Plot embeddings
    if depths is not None:
        scatter = ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=depths, cmap='viridis', s=50, alpha=0.7, edgecolors='white', linewidth=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Reasoning Depth')
    elif labels is not None:
        colors = ['#2ecc71' if l == 'TRUE' else '#e74c3c' for l in labels]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='TRUE'),
                          Patch(facecolor='#e74c3c', label='FALSE')]
        ax.legend(handles=legend_elements, loc='upper right')
    else:
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c='#3498db', s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dimension_ablation(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot dimension ablation results as a heatmap.
    
    Args:
        results: List of dimension ablation results with dimension, probe_type, spearman_rho
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    dimensions = sorted(set(r["dimension"] for r in results))
    probe_types = sorted(set(r["probe_type"] for r in results))
    
    # Create matrix
    matrix = np.zeros((len(probe_types), len(dimensions)))
    for i, pt in enumerate(probe_types):
        for j, dim in enumerate(dimensions):
            subset = [r for r in results if r["probe_type"] == pt and r["dimension"] == dim]
            if subset:
                matrix[i, j] = np.mean([r["spearman_rho"] for r in subset])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels([str(d) for d in dimensions])
    ax.set_yticks(range(len(probe_types)))
    ax.set_yticklabels([pt.title() for pt in probe_types])
    
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Probe Type')
    ax.set_title('Dimension Ablation: Spearman ρ')
    
    # Add text annotations
    for i in range(len(probe_types)):
        for j in range(len(dimensions)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Spearman ρ')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot training loss curves for multiple probes.
    
    Args:
        histories: Dict mapping probe_name -> {"train_loss": [...], "val_loss": [...]}
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(histories)))
    
    for (name, history), color in zip(histories.items(), colors):
        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label=name, color=color, linewidth=2)
        if "val_loss" in history and history["val_loss"]:
            axes[1].plot(history["val_loss"], label=name, color=color, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_curvature_sweep(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot curvature sweep results.
    
    Args:
        results: List of results with curvature field
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    curvatures = sorted(set(r.get("curvature", 1.0) for r in results))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    rhos = []
    for c in curvatures:
        subset = [r for r in results if r.get("curvature", 1.0) == c]
        rhos.append(np.mean([r["spearman_rho"] for r in subset]) if subset else 0)
    
    bars = ax.bar([str(c) for c in curvatures], rhos, color='#9b59b6', alpha=0.8)
    
    ax.set_xlabel('Curvature (c)')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Hyperbolic Curvature Sweep')
    ax.set_ylim(0, 1)
    
    for bar, rho in zip(bars, rhos):
        ax.annotate(f'{rho:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, rho),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_results_table(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> str:
    """
    Create a LaTeX-formatted results table.
    """
    # Group by probe type
    euclidean = [r for r in results if r["probe_type"] == "euclidean"]
    hyperbolic = [r for r in results if r["probe_type"] == "hyperbolic"]
    
    def avg(lst, key):
        return np.mean([r[key] for r in lst]) if lst else 0
    
    def std(lst, key):
        return np.std([r[key] for r in lst]) if lst else 0
    
    table = r"""
\begin{table}[h]
\centering
\caption{Comparison of Euclidean vs Hyperbolic Probes}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Euclidean} & \textbf{Hyperbolic} \\
\midrule
Spearman $\rho$ & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f \\
Avg Distortion & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f \\
MAP@5 & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        avg(euclidean, "spearman_rho"), std(euclidean, "spearman_rho"),
        avg(hyperbolic, "spearman_rho"), std(hyperbolic, "spearman_rho"),
        avg(euclidean, "avg_distortion"), std(euclidean, "avg_distortion"),
        avg(hyperbolic, "avg_distortion"), std(hyperbolic, "avg_distortion"),
        avg(euclidean, "map_at_5"), std(euclidean, "map_at_5"),
        avg(hyperbolic, "map_at_5"), std(hyperbolic, "map_at_5"),
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)
    
    return table


# ==========================================
# Additional Useful Plots (for future use)
# ==========================================



def plot_geometry_comparison(
    euclidean_metric: float,
    hyperbolic_metric: float,
    lorentz_metric: Optional[float] = None,
    metric_name: str = "Spearman ρ",
    model_name: str = "model",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Bar chart comparing Euclidean vs Hyperbolic (vs Lorentz) probe performance.
    
    Args:
        euclidean_metric: Metric value for Euclidean probe
        hyperbolic_metric: Metric value for Hyperbolic probe
        lorentz_metric: Optional metric value for Lorentz probe
        metric_name: Name of the metric being compared
        model_name: Model name for title
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if lorentz_metric is not None:
        labels = ['Euclidean', 'Poincaré', 'Lorentz']
        values = [euclidean_metric, hyperbolic_metric, lorentz_metric]
        colors = ['#3498db', '#e74c3c', '#9b59b6']
    else:
        labels = ['Euclidean', 'Hyperbolic']
        values = [euclidean_metric, hyperbolic_metric]
        colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel(metric_name)
    ax.set_title(f'Geometry Comparison: {model_name}')
    ax.set_ylim(0, max(values) * 1.2)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_depth_scaling(
    depth_radii: Dict[int, float],
    model_name: str = "model",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Bar chart showing complexity scaling (hyperbolic radius by reasoning depth).
    
    Theory: Deeper reasoning chains should map to larger hyperbolic radii,
    showing the model encoding complexity in the hyperbolic manifold.
    
    Args:
        depth_radii: Dict mapping depth (1-5) to mean hyperbolic radius
        model_name: Model name for title
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    Usage:
        # Compute from H1 results with saved embeddings
        depth_radii = {}
        for depth in range(1, 6):
            mask = (depths == depth)
            depth_radii[depth] = np.linalg.norm(embeddings[mask], axis=1).mean()
        plot_depth_scaling(depth_radii, model_name="deepseek_7b")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    depths = sorted(depth_radii.keys())
    radii = [depth_radii[d] for d in depths]
    
    bars = ax.bar([f'Depth {d}' for d in depths], radii, color='#9b59b6')
    
    ax.set_ylabel('Mean Hyperbolic Radius')
    ax.set_xlabel('Reasoning Depth (hops)')
    ax.set_title(f'Complexity Scaling - {model_name}')
    
    for bar, r in zip(bars, radii):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_violin_distribution(
    data: Dict[str, np.ndarray],
    metric_name: str = "Hyperbolic Distance",
    model_name: str = "model",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    palette: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Violin plot showing distribution of a metric across conditions.
    
    Useful for comparing distributions of hyperbolic distances or radii
    between TRUE, FALSE (hallucination), and UNRELATED samples.
    
    Args:
        data: Dict mapping condition name to array of values
              e.g., {"True (1 Hop)": [...], "Hallucination": [...]}
        metric_name: Name of the metric being plotted
        model_name: Model name for title
        output_path: Path to save figure
        figsize: Figure size
        palette: Optional color palette dict
        
    Returns:
        Matplotlib figure
        
    Usage:
        data = {
            "True (1-2 hop)": distances[shallow_mask],
            "True (4-5 hop)": distances[deep_mask],
            "Hallucination": distances[false_mask],
        }
        plot_violin_distribution(data, metric_name="Hyperbolic Radius")
    """
    import pandas as pd
    
    if palette is None:
        palette = {
            'True (1-2 hop)': '#3498db',
            'True (4-5 hop)': '#2980b9',
            'Hallucination': '#e74c3c',
            'Unrelated': '#95a5a6',
        }
    
    # Build dataframe
    rows = []
    for condition, values in data.items():
        for v in values:
            rows.append({'Condition': condition, metric_name: v})
    df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use order from palette if available, else from data keys
    order = [k for k in palette.keys() if k in data]
    if not order:
        order = list(data.keys())
    
    sns.violinplot(
        data=df, x='Condition', y=metric_name, hue='Condition',
        order=order, palette=palette, ax=ax, legend=False
    )
    
    ax.set_title(f'{metric_name} Distribution - {model_name}')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_layer_ablation(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot layer ablation results comparing different layer selection strategies.
    
    Args:
        results: List of layer ablation results with 'strategy' and 'spearman_rho' fields
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    Example:
        results = [
            {"strategy": "fixed_sweep", "spearman_rho": 0.55, "n_layers_used": 8},
            {"strategy": "max_drift_top3", "spearman_rho": 0.62, "n_layers_used": 3},
            {"strategy": "threshold_p90", "spearman_rho": 0.60, "n_layers_used": 4},
        ]
        plot_layer_ablation(results, output_path="layer_ablation.png")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    strategies = [r["strategy"] for r in results]
    spearman = [r.get("spearman_rho", 0) for r in results]
    n_layers = [r.get("n_layers_used", 0) for r in results]
    
    # Color by number of layers used
    colors = plt.cm.viridis([n / max(n_layers) for n in n_layers])
    
    bars = ax.bar(strategies, spearman, color=colors, edgecolor='black', alpha=0.8)
    
    # Highlight best
    best_idx = np.argmax(spearman)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    # Add value labels
    for bar, val, nl in zip(bars, spearman, n_layers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}\n({nl}L)', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Spearman ρ')
    ax.set_xlabel('Layer Selection Strategy')
    ax.set_title('Layer Ablation: Selection Strategy Comparison')
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.set_ylim(0, max(spearman) * 1.15)
    
    # Add colorbar for n_layers
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(min(n_layers), max(n_layers)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='# Layers Used')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_head_ablation(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot head selection ablation results comparing different head aggregation modes.
    
    Args:
        results: List of head ablation results with 'head_mode' and 'spearman_rho' fields
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    Example:
        results = [
            {"head_mode": "mean", "spearman_rho": 0.52, "avg_tokens_selected": 5.2},
            {"head_mode": "max", "spearman_rho": 0.58, "avg_tokens_selected": 4.8},
            {"head_mode": "threshold", "spearman_rho": 0.56, "avg_tokens_selected": 5.0},
            {"head_mode": "all_pool", "spearman_rho": 0.45, "avg_tokens_selected": 128.0},
        ]
        plot_head_ablation(results, output_path="head_ablation.png")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    modes = [r["head_mode"] for r in results]
    spearman = [r.get("spearman_rho", 0) for r in results]
    avg_tokens = [r.get("avg_tokens_selected", 0) for r in results]
    
    # Color based on whether attention-based (green) or baseline (gray)
    colors = ['#2ecc71' if m != 'all_pool' else '#95a5a6' for m in modes]
    
    bars = ax.bar(modes, spearman, color=colors, edgecolor='black', alpha=0.8)
    
    # Highlight best
    best_idx = np.argmax(spearman)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    # Add value labels
    for bar, val, nt in zip(bars, spearman, avg_tokens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}\n({nt:.1f} tok)', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Spearman ρ')
    ax.set_xlabel('Head Selection Mode')
    ax.set_title('Head Ablation: Attention Head Aggregation Comparison')
    ax.set_ylim(0, max(spearman) * 1.15)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Attention-weighted'),
        Patch(facecolor='#95a5a6', edgecolor='black', label='Baseline (all tokens)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

