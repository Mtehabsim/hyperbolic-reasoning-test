#!/usr/bin/env python3
"""
Dimension ablation experiment.

Tests distortion across embedding dimensions: {2, 5, 8, 16, 32}
This is critical for Figure 1 in the paper.

Usage:
    python scripts/run_dimension_ablation.py --cached-activations outputs/activations/deepseek_prontoqa.pt
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes import create_probe, train_probe, evaluate_probe
from src.geometry.metrics import compute_all_metrics
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed


@dataclass
class DimensionResult:
    """Result from dimension ablation."""
    probe_type: str
    dimension: int
    layer: int
    curvature: float
    spearman_rho: float
    avg_distortion: float
    map_at_5: float
    stress: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DimensionAblationExperiment:
    """
    Dimension ablation experiment.
    
    Tests how distortion varies with embedding dimension for both
    Euclidean and Hyperbolic probes. Hyperbolic should be more
    dimension-efficient (lower distortion at smaller dims).
    """
    
    DIMENSIONS = [2, 4, 5, 8, 16, 32]
    CURVATURES = [0.1, 0.3, 0.5, 0.7, 1.0]
    PROBE_TYPES = ["euclidean", "hyperbolic"]
    
    def __init__(
        self,
        input_dim: int,
        layer: int = 23,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.layer = layer
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.logger = get_logger()
        
        self.results: List[DimensionResult] = []
    
    def run_single(
        self,
        probe_type: str,
        dimension: int,
        hidden_states: torch.Tensor,
        target_distances: torch.Tensor,
        device: str = "cpu",
        curvature: float = 1.0,
    ) -> DimensionResult:
        """Run for single dimension, curvature, and probe type."""
        set_seed(self.seed)

        # Create probe (curvature only applies to hyperbolic)
        probe = create_probe(
            probe_type=probe_type,
            input_dim=self.input_dim,
            output_dim=dimension,
            curvature=curvature,
        )

        # Pre-normalize targets for hyperbolic probes so train and evaluate
        # use the same target scale (fixes metric mismatch bug)
        eval_targets = target_distances
        if probe_type == "hyperbolic":
            max_target = target_distances.max()
            if max_target > 0:
                eval_targets = target_distances * (4.0 / max_target)

        probe, _ = train_probe(
            probe=probe,
            hidden_states=hidden_states,
            target_distances=eval_targets,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            device=device,
            verbose=False,
            normalize_targets=False,  # Already normalized above
        )

        # Evaluate with same normalized targets
        metrics = evaluate_probe(probe, hidden_states, eval_targets, device=device)
        
        return DimensionResult(
            probe_type=probe_type,
            dimension=dimension,
            layer=self.layer,
            curvature=curvature,
            spearman_rho=metrics["spearman_rho"],
            avg_distortion=metrics["avg_distortion"],
            map_at_5=metrics["map_at_5"],
            stress=metrics["stress"],
        )
    
    def run_all(
        self,
        hidden_states: torch.Tensor,
        target_distances: torch.Tensor,
        device: str = "cpu",
        dimensions: List[int] = None,
        curvatures: List[float] = None,
        sweep_curvature: bool = False,
    ) -> List[DimensionResult]:
        """Run all dimension x probe_type (x curvature for hyperbolic) combinations.

        Args:
            hidden_states: Input hidden states
            target_distances: Ground truth distances
            device: Compute device
            dimensions: List of dimensions to sweep (default: DIMENSIONS)
            curvatures: List of curvatures to sweep for hyperbolic (default: CURVATURES)
            sweep_curvature: If True, sweep curvatures for hyperbolic probes
        """
        dimensions = dimensions or self.DIMENSIONS
        curvatures = curvatures or self.CURVATURES
        # Track actual dimensions/curvatures used for saving
        self._actual_dimensions = dimensions
        self._actual_curvatures = curvatures
        
        # Count total runs
        euclidean_runs = len(dimensions)
        hyperbolic_runs = len(dimensions) * (len(curvatures) if sweep_curvature else 1)
        total = euclidean_runs + hyperbolic_runs
        pbar = tqdm(total=total, desc="Dimension/curvature ablation")
        
        for dim in dimensions:
            for probe_type in self.PROBE_TYPES:
                # For hyperbolic, optionally sweep curvatures; Euclidean ignores curvature
                curv_list = curvatures if (probe_type == "hyperbolic" and sweep_curvature) else [1.0 if probe_type == "hyperbolic" else 0.0]
                
                for curv in curv_list:
                    self.logger.info(f"Testing {probe_type} d={dim} c={curv}")
                    
                    result = self.run_single(
                        probe_type=probe_type,
                        dimension=dim,
                        hidden_states=hidden_states,
                        target_distances=target_distances,
                        device=device,
                        curvature=curv,
                    )
                    
                    self.results.append(result)
                    self.logger.debug(
                        f"{probe_type} d={dim} c={curv}: "
                        f"rho={result.spearman_rho:.4f}, "
                        f"distortion={result.avg_distortion:.4f}"
                    )
                    
                    pbar.update(1)
        
        pbar.close()
        return self.results
    
    def save_results(self, output_path: Path) -> None:
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment": "dimension_ablation",
            "layer": self.layer,
            "dimensions": getattr(self, '_actual_dimensions', self.DIMENSIONS),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics, aggregating across curvatures if swept."""
        summary = {}
        dimensions = getattr(self, '_actual_dimensions', self.DIMENSIONS)

        for probe_type in self.PROBE_TYPES:
            type_results = [r for r in self.results if r.probe_type == probe_type]

            for dim in dimensions:
                dim_results = [r for r in type_results if r.dimension == dim]
                if dim_results:
                    # Aggregate across curvatures (take mean if multiple)
                    distortions = [r.avg_distortion for r in dim_results]
                    rhos = [r.spearman_rho for r in dim_results]
                    curvatures = [r.curvature for r in dim_results]
                    
                    summary[f"{probe_type}_d{dim}"] = {
                        "distortion": float(np.mean(distortions)),
                        "distortion_std": float(np.std(distortions)) if len(distortions) > 1 else 0.0,
                        "spearman_rho": float(np.mean(rhos)),
                        "spearman_std": float(np.std(rhos)) if len(rhos) > 1 else 0.0,
                        "best_curvature": curvatures[np.argmax(rhos)] if probe_type == "hyperbolic" else None,
                        "n_curvatures": len(dim_results),
                    }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Dimension ablation experiment")
    parser.add_argument(
        "--cached-activations",
        type=Path,
        required=True,
        help="Path to cached activations from extract_all_activations.py"
    )
    parser.add_argument("--layer", type=int, default=23, help="Layer to probe")
    parser.add_argument("--n-epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dimension_ablation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--sweep-curvature",
        action="store_true",
        help="Sweep curvatures [0.5, 1.0, 2.0] for hyperbolic probes"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger()
    set_seed(args.seed)
    
    # Load cached activations
    logger.info(f"Loading cached activations from {args.cached_activations}")
    cache = torch.load(args.cached_activations, map_location="cpu", weights_only=False)
    
    metadata = cache["metadata"]
    logger.info(f"Model: {metadata['model']}, Dataset: {metadata['dataset']}")
    logger.info(f"Samples: {metadata['n_samples']}, Layers: {metadata['n_layers']}")
    
    # Get activations for specified layer
    activations = cache["activations"][args.layer]  # [n_samples, seq_len, d_model]
    
    # Mean pool across sequence length
    hidden_states = activations.mean(dim=1)  # [n_samples, d_model]
    logger.info(f"Hidden states shape: {hidden_states.shape}")
    
    # Build target distances from graph distances
    # Check if we have true cross-sample pairwise distances (binary tree) or need depth fallback (PrOntoQA)
    n_samples = hidden_states.shape[0]
    graph_dists = metadata.get("graph_distances", [])

    # Only use graph_distances if it's a cross-sample matrix matching n_samples x n_samples
    # (PrOntoQA stores per-sample chain distances which are small matrices, not cross-sample)
    use_graph_dists = (graph_dists and graph_dists[0] is not None
                       and hasattr(graph_dists[0], '__len__')
                       and len(graph_dists[0]) == n_samples)
    if use_graph_dists:
        logger.info("Using TRUE pairwise graph distances from dataset")
        target_distances = torch.tensor(graph_dists[0], dtype=torch.float32)
    else:
        # Depth-based pairwise distances (PrOntoQA, ListOps)
        logger.info("Using depth-based pairwise distances")
        depths = np.array(metadata["depths"])
        target_distances = torch.tensor(
            np.abs(depths.reshape(-1, 1) - depths.reshape(1, -1)),
            dtype=torch.float32,
        )
    logger.info(f"Target distances shape: {target_distances.shape}")
    
    # Run experiment
    experiment = DimensionAblationExperiment(
        input_dim=hidden_states.shape[1],
        layer=args.layer,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
    
    results = experiment.run_all(
        hidden_states=hidden_states,
        target_distances=target_distances,
        device=args.device,
        sweep_curvature=args.sweep_curvature,
    )
    
    # Save with model name prefix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = metadata.get('model', 'unknown').replace('/', '_')
    output_file = args.output_dir / f"{model_name}_dimension_ablation_results.json"
    experiment.save_results(output_file)
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    summary = experiment.get_summary()
    logger.info("\n=== Dimension Ablation Summary ===")
    logger.info("Distortion by dimension:")
    for key, val in sorted(summary.items()):
        logger.info(f"  {key}: distortion={val['distortion']:.4f}, rho={val['spearman_rho']:.4f}")


if __name__ == "__main__":
    main()
