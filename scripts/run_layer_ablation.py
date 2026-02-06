#!/usr/bin/env python3
"""
Layer ablation experiment.

Compares layer selection strategies to find optimal probing layers:
1. Fixed sweep (hardcoded layers)
2. Max drift (top-k layers with highest L2 drift)
3. Threshold (layers above 90th percentile drift)

For each strategy, layers are aggregated (averaged) before probe training.
Uses training with early stopping to prevent overfitting.

Usage:
    python scripts/run_layer_ablation.py \
        --cached-activations outputs/activations/deepseek.pt \
        --output-dir outputs/layer_ablation
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.layer_selection import (
    compute_layer_drift,
    select_layers_by_max_drift,
    select_layers_by_threshold,
    aggregate_layer_activations,
)
from src.probes import create_probe
from src.geometry.metrics import compute_all_metrics
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed


@dataclass
class LayerAblationResult:
    """Result from layer ablation."""
    strategy: str
    selected_layers: List[int]
    n_layers_used: int
    spearman_rho: float
    avg_distortion: float
    map_at_5: float
    stress: float
    train_loss_final: float
    epochs_trained: int
    early_stopped: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LayerAblationExperiment:
    """
    Layer ablation experiment.
    
    Compares different layer selection strategies and aggregates
    selected layers before probe training.
    """
    
    # Default fixed layers (for comparison)
    FIXED_LAYERS = [8, 12, 16, 19, 21, 23, 25, 27]
    
    def __init__(
        self,
        input_dim: int,
        probe_type: str = "hyperbolic",
        output_dim: int = 16,
        curvature: float = 1.0,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.probe_type = probe_type
        self.output_dim = output_dim
        self.curvature = curvature
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.seed = seed
        self.logger = get_logger()
        
        self.results: List[LayerAblationResult] = []
    
    def train_with_early_stopping(
        self,
        probe: torch.nn.Module,
        hidden_states: torch.Tensor,
        target_distances: torch.Tensor,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Train probe with early stopping and validation split.
        
        Returns dict with final loss, epochs trained, and early_stopped flag.
        """
        probe = probe.to(device)
        hidden_states = hidden_states.to(device).float()  # Ensure float32
        target_distances = target_distances.to(device).float()
        
        n_samples = hidden_states.shape[0]
        n_val = int(n_samples * self.validation_split)
        n_train = n_samples - n_val
        
        # Shuffle and split
        indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(self.seed))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_h = hidden_states[train_idx]
        val_h = hidden_states[val_idx]
        
        # Target distances are pairwise, so we need to subset carefully
        train_targets = target_distances[train_idx][:, train_idx]
        val_targets = target_distances[val_idx][:, val_idx]
        
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        epochs_trained = 0
        early_stopped = False
        
        for epoch in range(self.n_epochs):
            # Training
            probe.train()
            optimizer.zero_grad()
            
            embeddings = probe(train_h)
            pred_dists = probe.pairwise_distances(embeddings)
            train_loss = torch.nn.functional.mse_loss(pred_dists, train_targets)
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            probe.eval()
            with torch.no_grad():
                val_embeddings = probe(val_h)
                val_pred_dists = probe.pairwise_distances(val_embeddings)
                val_loss = torch.nn.functional.mse_loss(val_pred_dists, val_targets)
            
            scheduler.step(val_loss)
            epochs_trained = epoch + 1
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = probe.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    early_stopped = True
                    break
        
        # Restore best state
        if best_state is not None:
            probe.load_state_dict(best_state)
        
        return {
            "final_loss": best_val_loss.item(),
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
        }
    
    def run_strategy(
        self,
        strategy_name: str,
        selected_layers: List[int],
        layer_activations: Dict[int, torch.Tensor],
        target_distances: torch.Tensor,
        device: str = "cpu",
    ) -> LayerAblationResult:
        """
        Run ablation for a single strategy.
        
        Aggregates selected layers and trains probe.
        """
        self.logger.info(f"Strategy: {strategy_name}, Layers: {selected_layers}")
        
        # Aggregate layers by averaging
        aggregated = aggregate_layer_activations(layer_activations, selected_layers)
        
        # Create probe
        probe = create_probe(
            probe_type=self.probe_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            curvature=self.curvature,
        )
        
        # Train with early stopping
        train_result = self.train_with_early_stopping(
            probe=probe,
            hidden_states=aggregated,
            target_distances=target_distances,
            device=device,
        )
        
        # Evaluate
        probe.eval()
        with torch.no_grad():
            embeddings = probe(aggregated.to(device).float())
            pred_distances = probe.pairwise_distances(embeddings)
            
            metrics = compute_all_metrics(
                pred_distances=pred_distances.cpu(),
                target_distances=target_distances.cpu(),
            )
        
        result = LayerAblationResult(
            strategy=strategy_name,
            selected_layers=selected_layers,
            n_layers_used=len(selected_layers),
            spearman_rho=metrics["spearman_rho"],
            avg_distortion=metrics["avg_distortion"],
            map_at_5=metrics["map_at_5"],
            stress=metrics["stress"],
            train_loss_final=train_result["final_loss"],
            epochs_trained=train_result["epochs_trained"],
            early_stopped=train_result["early_stopped"],
        )
        
        self.results.append(result)
        return result
    
    def run_all(
        self,
        layer_activations: Dict[int, torch.Tensor],
        target_distances: torch.Tensor,
        device: str = "cpu",
        top_k: int = 3,
        percentile: float = 90.0,
    ) -> List[LayerAblationResult]:
        """
        Run all layer selection strategies.
        
        Strategies:
        1. Fixed sweep (hardcoded layers)
        2. Max drift with top-k
        3. Max drift with percentile threshold
        4. Threshold (percentile)
        """
        set_seed(self.seed)
        self.results = []
        
        # Compute drifts
        drifts = compute_layer_drift(layer_activations)
        self.logger.info(f"Layer drifts: {drifts}")
        
        # Strategy 1: Fixed layers
        fixed_available = [l for l in self.FIXED_LAYERS if l in layer_activations]
        if fixed_available:
            self.run_strategy(
                strategy_name="fixed_sweep",
                selected_layers=fixed_available,
                layer_activations=layer_activations,
                target_distances=target_distances,
                device=device,
            )
        
        # Strategy 2: Max drift (top-k)
        max_drift_layers = select_layers_by_max_drift(layer_activations, top_k=top_k, threshold=0.5)
        self.run_strategy(
            strategy_name=f"max_drift_top{top_k}",
            selected_layers=max_drift_layers,
            layer_activations=layer_activations,
            target_distances=target_distances,
            device=device,
        )
        
        # Strategy 3: Threshold (percentile)
        threshold_layers = select_layers_by_threshold(layer_activations, percentile=percentile)
        self.run_strategy(
            strategy_name=f"threshold_p{int(percentile)}",
            selected_layers=threshold_layers,
            layer_activations=layer_activations,
            target_distances=target_distances,
            device=device,
        )
        
        # Strategy 4: Single best layer (highest drift)
        if drifts:
            best_single = max(drifts, key=drifts.get)
            self.run_strategy(
                strategy_name="single_best",
                selected_layers=[best_single],
                layer_activations=layer_activations,
                target_distances=target_distances,
                device=device,
            )
        
        return self.results
    
    def save_results(self, output_path: Path):
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find best strategy
        best = max(self.results, key=lambda r: r.spearman_rho)
        
        data = {
            "experiment": "layer_ablation",
            "probe_type": self.probe_type,
            "output_dim": self.output_dim,
            "curvature": self.curvature,
            "best_strategy": best.strategy,
            "best_layers": best.selected_layers,
            "best_spearman_rho": best.spearman_rho,
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        return data
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.results:
            return {}
        
        best = max(self.results, key=lambda r: r.spearman_rho)
        
        return {
            "best_strategy": best.strategy,
            "best_layers": best.selected_layers,
            "best_spearman_rho": best.spearman_rho,
            "n_strategies_tested": len(self.results),
        }


def main():
    parser = argparse.ArgumentParser(description="Layer ablation experiment")
    parser.add_argument("--cached-activations", type=Path, required=True,
                        help="Path to cached activations (.pt file)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/layer_ablation"),
                        help="Output directory")
    parser.add_argument("--probe-type", type=str, default="hyperbolic",
                        choices=["euclidean", "hyperbolic"],
                        help="Probe type")
    parser.add_argument("--output-dim", type=int, default=16,
                        help="Probe output dimension")
    parser.add_argument("--curvature", type=float, default=1.0,
                        help="Hyperbolic curvature")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Max training epochs")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-k for max drift strategy")
    parser.add_argument("--percentile", type=float, default=90.0,
                        help="Percentile threshold for layer selection")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger()
    set_seed(args.seed)
    
    # Load cached activations
    logger.info(f"Loading activations from {args.cached_activations}")
    data = torch.load(args.cached_activations, map_location="cpu")
    
    # Get activations per layer
    if "activations" in data:
        layer_activations = data["activations"]
    else:
        layer_activations = data
    
    # Convert to dict if needed (handle different cache formats)
    if isinstance(layer_activations, torch.Tensor):
        # Single tensor format - assume it's already pooled per sample
        logger.warning("Single tensor format - treating as single layer")
        layer_activations = {0: layer_activations}
    elif isinstance(layer_activations, dict):
        # Expected format: {layer_idx: Tensor[n_samples, seq_len, d_model] or [n_samples, d_model]}
        # Mean pool if 3D
        pooled = {}
        for layer, acts in layer_activations.items():
            if acts.dim() == 3:
                pooled[layer] = acts.mean(dim=1)  # Pool across sequence
            else:
                pooled[layer] = acts
        layer_activations = pooled
    
    logger.info(f"Loaded {len(layer_activations)} layers")
    
    # Get input dim
    sample_acts = next(iter(layer_activations.values()))
    input_dim = sample_acts.shape[-1]
    n_samples = sample_acts.shape[0]
    
    logger.info(f"Input dim: {input_dim}, Samples: {n_samples}")
    
    # Create target distances - prefer graph_distances over depth fallback
    metadata = data.get("metadata", data)  # Handle both formats
    graph_dists = metadata.get("graph_distances", [])
    
    # Check if we have true pairwise distances (binary tree)
    if graph_dists and graph_dists[0] is not None:
        logger.info("Using TRUE pairwise graph distances from dataset")
        target_distances = torch.tensor(graph_dists[0], dtype=torch.float32)
    else:
        # Fallback to depth-based pairwise distances (PrOntoQA, ListOps)
        logger.info("Using depth-based pairwise distances (fallback)")
        if "depths" in metadata:
            depths = np.array(metadata["depths"])
        elif "depths" in data:
            depths = np.array(data["depths"])
        else:
            logger.warning("No depths found in cache, using random depths for testing")
            depths = np.random.randint(1, 6, size=n_samples)
        
        target_distances = torch.tensor(
            np.abs(depths.reshape(-1, 1) - depths.reshape(1, -1)),
            dtype=torch.float32
        )
    
    # Run experiment
    experiment = LayerAblationExperiment(
        input_dim=input_dim,
        probe_type=args.probe_type,
        output_dim=args.output_dim,
        curvature=args.curvature,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
    
    results = experiment.run_all(
        layer_activations=layer_activations,
        target_distances=target_distances,
        device=args.device,
        top_k=args.top_k,
        percentile=args.percentile,
    )
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment.save_results(args.output_dir / "layer_ablation_results.json")
    
    # Generate visualization
    try:
        from src.analysis.visualization import plot_layer_ablation
        fig = plot_layer_ablation(
            results=[r.to_dict() for r in results],
            output_path=args.output_dir / "layer_ablation_comparison.png",
        )
        logger.info(f"Saved plot to {args.output_dir / 'layer_ablation_comparison.png'}")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")
    
    # Print summary
    summary = experiment.get_summary()
    logger.info(f"\n{'='*50}")
    logger.info(f"LAYER ABLATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Best strategy: {summary['best_strategy']}")
    logger.info(f"Best layers: {summary['best_layers']}")
    logger.info(f"Best Spearman ρ: {summary['best_spearman_rho']:.4f}")
    logger.info(f"{'='*50}")
    
    # Print all results
    logger.info("\nAll Results:")
    for r in results:
        logger.info(f"  {r.strategy}: ρ={r.spearman_rho:.4f}, layers={r.selected_layers}, "
                    f"epochs={r.epochs_trained}, early_stop={r.early_stopped}")


if __name__ == "__main__":
    main()

