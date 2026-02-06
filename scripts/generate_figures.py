#!/usr/bin/env python3
"""
Generate figures for experiment results.

Usage:
    python scripts/generate_figures.py --results-dir outputs --output-dir figures
    python scripts/generate_figures.py --h1-results outputs/deepseek_7b/h1_results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.analysis.visualization import (
    plot_euclidean_vs_hyperbolic,
    plot_token_ablation,
    plot_layer_sweep,
    plot_model_comparison,
    plot_poincare_disk,
    plot_dimension_ablation,
    plot_training_curves,
    plot_curvature_sweep,
    create_results_table,
    # Additional plots
    plot_geometry_comparison,
    plot_depth_scaling,        # For depth → radius analysis
    plot_violin_distribution,  # For distribution comparison
    plot_layer_ablation,       # For layer selection ablation
    plot_head_ablation,        # For head aggregation ablation
)
from src.utils.logging import setup_logging, get_logger


def load_results(path: Path) -> list:
    """Load JSON results file."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    # Handle both list and dict formats
    if isinstance(data, dict):
        return data.get("results", [data])
    return data


def generate_h1_figures(results_dir: Path, output_dir: Path, logger):
    """Generate all H1 (Hierarchy) experiment figures."""
    logger.info("Generating H1 figures...")
    
    all_results = []
    model_results = {}
    
    # Load results from each model
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        h1_path = model_dir / "h1_results.json"
        if h1_path.exists():
            results = load_results(h1_path)
            all_results.extend(results)
            model_results[model_dir.name] = results
            logger.info(f"  Loaded {len(results)} results from {model_dir.name}")
    
    if not all_results:
        logger.warning("No H1 results found")
        return
    
    # 1. Euclidean vs Hyperbolic comparison
    fig1 = plot_euclidean_vs_hyperbolic(
        all_results,
        output_path=output_dir / "h1_euclidean_vs_hyperbolic.png",
        title="H1: Euclidean vs Hyperbolic Probes"
    )
    logger.info("  ✓ h1_euclidean_vs_hyperbolic.png")
    
    # 2. Layer sweep
    fig2 = plot_layer_sweep(
        all_results,
        output_path=output_dir / "h1_layer_sweep.png"
    )
    logger.info("  ✓ h1_layer_sweep.png")
    
    # 3. Model comparison (if multiple models)
    if len(model_results) > 1:
        fig3 = plot_model_comparison(
            model_results,
            output_path=output_dir / "h1_model_comparison.png"
        )
        logger.info("  ✓ h1_model_comparison.png")
    
    # 4. LaTeX table
    table = create_results_table(
        all_results,
        output_path=output_dir / "h1_results_table.tex"
    )
    logger.info("  ✓ h1_results_table.tex")


def generate_h2_figures(results_dir: Path, output_dir: Path, logger):
    """Generate all H2 (Token Ablation) experiment figures."""
    logger.info("Generating H2 figures...")
    
    all_results = []
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        h2_path = model_dir / "h2_results.json"
        if h2_path.exists():
            results = load_results(h2_path)
            all_results.extend(results)
            logger.info(f"  Loaded {len(results)} results from {model_dir.name}")
    
    if not all_results:
        logger.warning("No H2 results found")
        return
    
    # Token ablation comparison
    fig = plot_token_ablation(
        all_results,
        output_path=output_dir / "h2_token_ablation.png"
    )
    logger.info("  ✓ h2_token_ablation.png")


def generate_dimension_figures(results_path: Path, output_dir: Path, logger):
    """Generate dimension ablation figures."""
    logger.info("Generating dimension ablation figures...")
    
    if not results_path.exists():
        logger.warning(f"Dimension ablation results not found: {results_path}")
        return
    
    results = load_results(results_path)
    
    # Dimension ablation heatmap
    fig = plot_dimension_ablation(
        results,
        output_path=output_dir / "dimension_ablation.png"
    )
    logger.info("  ✓ dimension_ablation.png")
    
    # Curvature sweep (if present)
    curvature_results = [r for r in results if r.get("curvature") is not None]
    if curvature_results:
        fig2 = plot_curvature_sweep(
            curvature_results,
            output_path=output_dir / "curvature_sweep.png"
        )
        logger.info("  ✓ curvature_sweep.png")


def generate_poincare_figures(embeddings_dir: Path, output_dir: Path, logger):
    """Generate Poincaré disk visualizations from saved embeddings."""
    logger.info("Generating Poincaré disk figures...")
    
    if not embeddings_dir.exists():
        logger.warning(f"Embeddings directory not found: {embeddings_dir}")
        return
    
    # Look for 2D embeddings (d=2)
    for emb_file in embeddings_dir.glob("*hyperbolic*d2*.npy"):
        embeddings = np.load(emb_file)
        
        if embeddings.shape[1] != 2:
            continue
        
        # Try to load corresponding metadata
        meta_file = emb_file.with_suffix(".json")
        labels = None
        depths = None
        
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            labels = meta.get("labels")
            depths = meta.get("depths")
        
        out_name = f"poincare_{emb_file.stem}.png"
        fig = plot_poincare_disk(
            embeddings,
            labels=labels,
            depths=depths,
            output_path=output_dir / out_name,
            title=f"Poincaré Disk: {emb_file.stem}"
        )
        logger.info(f"  ✓ {out_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save figures"
    )
    parser.add_argument(
        "--dimension-results",
        type=Path,
        default=None,
        help="Path to dimension ablation results JSON"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Path to saved embeddings for Poincaré disk plots"
    )
    parser.add_argument(
        "--h1-only",
        action="store_true",
        help="Generate only H1 figures"
    )
    parser.add_argument(
        "--h2-only",
        action="store_true",
        help="Generate only H2 figures"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(log_dir=args.output_dir / "logs")
    logger = get_logger()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("Generating Figures")
    logger.info("=" * 50)
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Generate figures
    if not args.h2_only:
        generate_h1_figures(args.results_dir, args.output_dir, logger)
    
    if not args.h1_only:
        generate_h2_figures(args.results_dir, args.output_dir, logger)
    
    # Dimension ablation
    dim_path = args.dimension_results or args.results_dir / "dimension_ablation_results.json"
    if dim_path.exists():
        generate_dimension_figures(dim_path, args.output_dir, logger)
    
    # Poincaré disk (if embeddings available)
    emb_dir = args.embeddings_dir or args.results_dir / "embeddings"
    if emb_dir.exists():
        generate_poincare_figures(emb_dir, args.output_dir, logger)
    
    logger.info("=" * 50)
    logger.info("Figure generation complete!")
    logger.info(f"Figures saved to: {args.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
