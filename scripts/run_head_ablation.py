#!/usr/bin/env python3
"""
Head selection ablation experiment.

Compares head selection modes for attention-weighted token selection:
1. mean: Average across all attention heads
2. max: Use only the highest-attention head
3. threshold: Use heads within 90% of max attention

Uses training with early stopping to prevent overfitting.

Usage:
    python scripts/run_head_ablation.py \
        --cached-activations outputs/activations/deepseek.pt \
        --layer 23 \
        --output-dir outputs/head_ablation
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

from src.model.token_selector import find_attention_weighted_positions
from src.probes import create_probe
from src.geometry.metrics import compute_all_metrics
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed


@dataclass
class HeadAblationResult:
    """Result from head selection ablation."""
    head_mode: str
    layer: int
    spearman_rho: float
    avg_distortion: float
    map_at_5: float
    stress: float
    train_loss_final: float
    epochs_trained: int
    early_stopped: bool
    avg_tokens_selected: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_tokens_with_head_mode(
    hidden_states: torch.Tensor,
    attention_weights: torch.Tensor,
    head_mode: str = "mean",
    top_k: int = 5,
    threshold: float = 0.9,
) -> torch.Tensor:
    """
    Pool hidden states using attention-weighted token selection with specified head mode.
    
    Args:
        hidden_states: [n_samples, seq_len, d_model]
        attention_weights: [n_samples, n_heads, seq_len, seq_len]
        head_mode: "mean", "max", or "threshold"
        top_k: Number of top tokens to select per sample
        threshold: Threshold for head selection (for "threshold" mode)
        
    Returns:
        Pooled hidden states [n_samples, d_model] (normalized)
        Average tokens selected per sample
    """
    n_samples = hidden_states.shape[0]
    d_model = hidden_states.shape[-1]
    pooled = torch.zeros(n_samples, d_model, dtype=hidden_states.dtype)
    tokens_per_sample = []
    
    for i in range(n_samples):
        sample_attn = attention_weights[i]  # [n_heads, seq_len, seq_len]
        sample_hidden = hidden_states[i]  # [seq_len, d_model]
        
        # Get top positions using specified head mode
        positions = find_attention_weighted_positions(
            attention_weights=sample_attn,
            top_k=top_k,
            exclude_first=1,
            exclude_last=1,
            head_selection=head_mode,
            head_threshold=threshold,
        )
        
        if positions:
            # Pool selected positions
            selected = sample_hidden[positions]
            pooled[i] = selected.mean(dim=0)
            tokens_per_sample.append(len(positions))
        else:
            # Fallback to mean pooling
            pooled[i] = sample_hidden.mean(dim=0)
            tokens_per_sample.append(sample_hidden.shape[0])
    
    # CRITICAL: Normalize pooled hidden states to prevent overflow in probe
    # LLM hidden states often have magnitudes in the thousands
    pooled = torch.nn.functional.layer_norm(pooled, [d_model])
    
    return pooled, np.mean(tokens_per_sample)


class HeadAblationExperiment:
    """
    Head selection ablation experiment.
    
    Compares different head aggregation modes for attention-weighted
    token selection.
    """
    
    HEAD_MODES = ["mean", "max", "threshold"]
    
    def __init__(
        self,
        input_dim: int,
        layer: int,
        probe_type: str = "hyperbolic",
        output_dim: int = 16,
        curvature: float = 0.5,  # Default to 0.5 per Phase 3a results
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.layer = layer
        self.probe_type = probe_type
        self.output_dim = output_dim
        self.curvature = curvature
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.seed = seed
        self.logger = get_logger()
        
        self.results: List[HeadAblationResult] = []
    
    def train_with_early_stopping(
        self,
        probe: torch.nn.Module,
        hidden_states: torch.Tensor,
        target_distances: torch.Tensor,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Train probe with early stopping and validation split.
        
        Implementation with:
        - NaN/Inf detection and early termination
        - Type-safe loss extraction
        - Comprehensive logging for debugging
        - Graceful degradation when training fails
        
        Args:
            probe: The probe module to train
            hidden_states: Input hidden states [n_samples, d_model]
            target_distances: Target pairwise distances [n_samples, n_samples]
            device: Device to train on
            
        Returns:
            Dictionary with training results:
                - final_loss: Best validation loss achieved
                - epochs_trained: Number of epochs completed
                - early_stopped: Whether early stopping was triggered
                - training_failed: Whether training failed due to NaN/Inf
        """
        # Move to device
        probe = probe.to(device)
        hidden_states = hidden_states.to(device).float()  # Ensure float32
        target_distances = target_distances.to(device).float()
        
        n_samples = hidden_states.shape[0]
        n_val = max(int(n_samples * self.validation_split), 1)  # At least 1 val sample
        n_train = n_samples - n_val
        
        if n_train < 2:
            self.logger.warning(f"Insufficient samples for training: {n_samples}")
            return {
                "final_loss": float('inf'),
                "epochs_trained": 0,
                "early_stopped": False,
                "training_failed": True,
            }
        
        # Shuffle and split with reproducible seed
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n_samples, generator=generator)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_h = hidden_states[train_idx]
        val_h = hidden_states[val_idx]
        
        train_targets = target_distances[train_idx][:, train_idx]
        val_targets = target_distances[val_idx][:, val_idx]
        
        # Optimizer setup
        optimizer = torch.optim.Adam(
            probe.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6
        )
        
        # Training state
        best_val_loss: float = float('inf')
        patience_counter: int = 0
        best_state: Optional[Dict] = None
        epochs_trained: int = 0
        early_stopped: bool = False
        training_failed: bool = False
        nan_count: int = 0
        max_nan_tolerance: int = 3  # Allow a few NaN epochs before giving up
        
        for epoch in range(self.n_epochs):
            # ========== Training Phase ==========
            probe.train()
            optimizer.zero_grad()
            
            try:
                embeddings = probe(train_h)
                pred_dists = probe.pairwise_distances(embeddings)
                train_loss = torch.nn.functional.mse_loss(pred_dists, train_targets)
            except RuntimeError as e:
                self.logger.error(f"Forward pass failed at epoch {epoch}: {e}")
                training_failed = True
                break
            
            # NaN/Inf check on training loss
            if not torch.isfinite(train_loss):
                nan_count += 1
                self.logger.warning(f"Epoch {epoch}: Non-finite training loss detected ({nan_count}/{max_nan_tolerance})")
                if nan_count >= max_nan_tolerance:
                    self.logger.error("Training failed: Too many NaN/Inf losses")
                    training_failed = True
                    break
                continue  # Skip this epoch, try again
            
            train_loss.backward()
            
            # Gradient clipping with norm check
            grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                self.logger.warning(f"Epoch {epoch}: Non-finite gradient norm, skipping update")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            # ========== Validation Phase ==========
            probe.eval()
            with torch.no_grad():
                try:
                    val_embeddings = probe(val_h)
                    val_pred_dists = probe.pairwise_distances(val_embeddings)
                    val_loss = torch.nn.functional.mse_loss(val_pred_dists, val_targets)
                except RuntimeError as e:
                    self.logger.error(f"Validation forward pass failed at epoch {epoch}: {e}")
                    training_failed = True
                    break
            
            # NaN/Inf check on validation loss
            if not torch.isfinite(val_loss):
                self.logger.warning(f"Epoch {epoch}: Non-finite validation loss, skipping")
                continue
            
            # Extract scalar value safely
            val_loss_scalar = val_loss.item()
            
            scheduler.step(val_loss)
            epochs_trained = epoch + 1
            
            # Early stopping check
            if val_loss_scalar < best_val_loss:
                best_val_loss = val_loss_scalar
                patience_counter = 0
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    early_stopped = True
                    break
        
        # Restore best state if available
        if best_state is not None:
            probe.load_state_dict(best_state)
        elif not training_failed:
            self.logger.warning("No best state saved, using final weights")
        
        return {
            "final_loss": best_val_loss,
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
            "training_failed": training_failed,
        }
    
    def run_mode(
        self,
        head_mode: str,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
        target_distances: torch.Tensor,
        device: str = "cpu",
        top_k: int = 5,
        threshold: float = 0.9,
    ) -> HeadAblationResult:
        """
        Run ablation for a single head selection mode.
        """
        self.logger.info(f"Head mode: {head_mode}")
        
        # Pool using specified head mode
        pooled, avg_tokens = select_tokens_with_head_mode(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            head_mode=head_mode,
            top_k=top_k,
            threshold=threshold,
        )
        
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
            hidden_states=pooled,
            target_distances=target_distances,
            device=device,
        )
        
        # Evaluate
        probe.eval()
        with torch.no_grad():
            embeddings = probe(pooled.to(device).float())
            pred_distances = probe.pairwise_distances(embeddings)
            
            metrics = compute_all_metrics(
                pred_distances=pred_distances.cpu(),
                target_distances=target_distances,
            )
        
        result = HeadAblationResult(
            head_mode=head_mode,
            layer=self.layer,
            spearman_rho=metrics["spearman_rho"],
            avg_distortion=metrics["avg_distortion"],
            map_at_5=metrics["map_at_5"],
            stress=metrics["stress"],
            train_loss_final=train_result["final_loss"],
            epochs_trained=train_result["epochs_trained"],
            early_stopped=train_result["early_stopped"],
            avg_tokens_selected=avg_tokens,
        )
        
        self.results.append(result)
        return result
    
    def run_all(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
        target_distances: torch.Tensor,
        device: str = "cpu",
        top_k: int = 5,
        threshold: float = 0.9,
    ) -> List[HeadAblationResult]:
        """
        Run all head selection modes.
        """
        set_seed(self.seed)
        self.results = []
        
        for mode in self.HEAD_MODES:
            self.run_mode(
                head_mode=mode,
                hidden_states=hidden_states,
                attention_weights=attention_weights,
                target_distances=target_distances,
                device=device,
                top_k=top_k,
                threshold=threshold,
            )
        
        # Also test all_pool (mean across all tokens) as baseline
        self.logger.info("Baseline: all_pool (mean across all tokens)")
        pooled_all = hidden_states.mean(dim=1)
        # Normalize to match other pooling methods
        pooled_all = torch.nn.functional.layer_norm(pooled_all, [pooled_all.shape[-1]])
        
        probe = create_probe(
            probe_type=self.probe_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            curvature=self.curvature,
        )
        
        train_result = self.train_with_early_stopping(
            probe=probe,
            hidden_states=pooled_all,
            target_distances=target_distances,
            device=device,
        )
        
        probe.eval()
        with torch.no_grad():
            embeddings = probe(pooled_all.to(device).float())
            pred_distances = probe.pairwise_distances(embeddings)
            metrics = compute_all_metrics(
                pred_distances=pred_distances.cpu(),
                target_distances=target_distances,
            )
        
        baseline_result = HeadAblationResult(
            head_mode="all_pool",
            layer=self.layer,
            spearman_rho=metrics["spearman_rho"],
            avg_distortion=metrics["avg_distortion"],
            map_at_5=metrics["map_at_5"],
            stress=metrics["stress"],
            train_loss_final=train_result["final_loss"],
            epochs_trained=train_result["epochs_trained"],
            early_stopped=train_result["early_stopped"],
            avg_tokens_selected=float(hidden_states.shape[1]),
        )
        self.results.append(baseline_result)
        
        return self.results
    
    def save_results(self, output_path: Path):
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find best mode
        best = max(self.results, key=lambda r: r.spearman_rho)
        
        data = {
            "experiment": "head_ablation",
            "layer": self.layer,
            "probe_type": self.probe_type,
            "output_dim": self.output_dim,
            "curvature": self.curvature,
            "best_mode": best.head_mode,
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
            "best_mode": best.head_mode,
            "best_spearman_rho": best.spearman_rho,
            "n_modes_tested": len(self.results),
        }


def main():
    parser = argparse.ArgumentParser(description="Head selection ablation experiment")
    parser.add_argument("--cached-activations", type=Path, required=True,
                        help="Path to cached activations (.pt file)")
    parser.add_argument("--layer", type=int, default=23,
                        help="Layer to use for ablation")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/head_ablation"),
                        help="Output directory")
    parser.add_argument("--probe-type", type=str, default="hyperbolic",
                        choices=["euclidean", "hyperbolic"],
                        help="Probe type")
    parser.add_argument("--output-dim", type=int, default=16,
                        help="Probe output dimension")
    parser.add_argument("--curvature", type=float, default=0.5,
                        help="Hyperbolic curvature (default: 0.5 per Phase 3a)")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Max training epochs")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-k tokens to select per sample")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Head threshold for 'threshold' mode")
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
    
    # Get activations for specified layer
    if "activations" in data:
        layer_activations = data["activations"]
    else:
        layer_activations = data
    
    if args.layer not in layer_activations:
        available = list(layer_activations.keys())
        logger.error(f"Layer {args.layer} not found. Available: {available}")
        args.layer = available[-1] if available else 0
        logger.info(f"Using layer {args.layer} instead")
    
    hidden_states = layer_activations[args.layer]
    logger.info(f"Hidden states shape: {hidden_states.shape}")
    
    # Get attention weights
    if "attention" in data and args.layer in data["attention"]:
        attention_weights = data["attention"][args.layer]
        logger.info(f"Attention weights shape: {attention_weights.shape}")
    else:
        # Generate mock attention for testing
        logger.warning("No attention weights found, generating random attention for testing")
        n_samples, seq_len, d_model = hidden_states.shape
        n_heads = 32  # Typical for 7B model
        attention_weights = torch.rand(n_samples, n_heads, seq_len, seq_len)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    # Get input dim
    input_dim = hidden_states.shape[-1]
    n_samples = hidden_states.shape[0]
    
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
            logger.warning("No depths found, using random depths for testing")
            depths = np.random.randint(1, 6, size=n_samples)
        
        target_distances = torch.tensor(
            np.abs(depths.reshape(-1, 1) - depths.reshape(1, -1)),
            dtype=torch.float32
        )
    
    # Run experiment
    experiment = HeadAblationExperiment(
        input_dim=input_dim,
        layer=args.layer,
        probe_type=args.probe_type,
        output_dim=args.output_dim,
        curvature=args.curvature,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
    
    results = experiment.run_all(
        hidden_states=hidden_states,
        attention_weights=attention_weights,
        target_distances=target_distances,
        device=args.device,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment.save_results(args.output_dir / "head_ablation_results.json")
    
    # Generate visualization
    try:
        from src.analysis.visualization import plot_head_ablation
        fig = plot_head_ablation(
            results=[r.to_dict() for r in results],
            output_path=args.output_dir / "head_ablation_comparison.png",
        )
        logger.info(f"Saved plot to {args.output_dir / 'head_ablation_comparison.png'}")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")
    
    # Print summary
    summary = experiment.get_summary()
    logger.info(f"\n{'='*50}")
    logger.info(f"HEAD ABLATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Best head mode: {summary['best_mode']}")
    logger.info(f"Best Spearman ρ: {summary['best_spearman_rho']:.4f}")
    logger.info(f"{'='*50}")
    
    # Print all results
    logger.info("\nAll Results:")
    for r in results:
        logger.info(f"  {r.head_mode}: ρ={r.spearman_rho:.4f}, avg_tokens={r.avg_tokens_selected:.1f}, "
                    f"epochs={r.epochs_trained}, early_stop={r.early_stopped}")


if __name__ == "__main__":
    main()

