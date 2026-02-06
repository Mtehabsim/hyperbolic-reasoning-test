"""
Experiment runner for hierarchy probing (H1: Hyperbolic vs Euclidean).

Tests whether hyperbolic probes preserve reasoning hierarchy better.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.base import Dataset
from src.geometry.metrics import compute_all_metrics
from src.probes import create_probe, EuclideanPairwiseProbe, HyperbolicPairwiseProbe
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed


@dataclass
class ProbeTrainingConfig:
    """Configuration for probe training."""
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    output_dim: int = 16
    curvature: float = 1.0
    mdr_max_norm: float = 15.0
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10


@dataclass 
class ExperimentResult:
    """Result from a single experiment run."""
    probe_type: str
    layer: int
    spearman_rho: float
    avg_distortion: float
    map_at_5: float
    stress: float
    train_loss_final: float
    epochs_trained: int
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HierarchyExperiment:
    """
    H1 Experiment: Compare Euclidean vs Hyperbolic probes.
    
    For each layer, train both probe types and compare metrics.
    """
    
    def __init__(
        self,
        input_dim: int,
        layers: List[int],
        config: Optional[ProbeTrainingConfig] = None,
        device: str = "cuda",
        seed: int = 42,
        save_embeddings: bool = False,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize experiment.
        
        Args:
            input_dim: Hidden state dimension (d_model)
            layers: List of layer indices to probe
            config: Training configuration
            device: Device for training
            seed: Random seed
            save_embeddings: If True, save embeddings for post-hoc visualization
            output_dir: Directory to save embeddings (required if save_embeddings=True)
        """
        self.input_dim = input_dim
        self.layers = layers
        self.config = config or ProbeTrainingConfig()
        self.device = device
        self.seed = seed
        self.save_embeddings = save_embeddings
        self.output_dir = Path(output_dir) if output_dir else None
        self.logger = get_logger()
        
        self.results: List[ExperimentResult] = []
    
    def train_probe(
        self,
        probe: nn.Module,
        hidden_states: torch.Tensor,
        target_distances: torch.Tensor,
        probe_type: str,
        layer: int,
    ) -> Tuple[nn.Module, float, int]:
        """
        Train a probe on hidden states.
        
        Args:
            probe: Probe module
            hidden_states: [n_samples, d_model]
            target_distances: [n_samples, n_samples]
            probe_type: Type name for logging
            layer: Layer index for logging
            
        Returns:
            (trained_probe, final_loss, epochs_trained)
        """
        probe = probe.to(self.device)
        hidden_states = hidden_states.to(self.device)
        target_distances = target_distances.to(self.device)
        
        optimizer = optim.Adam(
            probe.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        best_loss = float('inf')
        patience_counter = 0
        epochs_trained = 0
        
        for epoch in range(self.config.epochs):
            probe.train()
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = probe(hidden_states)
            pred_distances = probe.pairwise_distances(embeddings)
            
            # Loss
            loss = probe.distortion_loss(pred_distances, target_distances)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epochs_trained = epoch + 1
            current_loss = loss.item()
            
            # Early stopping
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.debug(f"Early stopping at epoch {epoch+1}")
                break
        
        return probe, best_loss, epochs_trained
    
    def evaluate_probe(
        self,
        probe: nn.Module,
        hidden_states: torch.Tensor,
        target_distances: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate a trained probe.
        
        Returns metrics dictionary.
        """
        probe.eval()
        hidden_states = hidden_states.to(self.device)
        target_distances = target_distances.to(self.device)
        
        with torch.no_grad():
            embeddings = probe(hidden_states)
            pred_distances = probe.pairwise_distances(embeddings)
        
        metrics = compute_all_metrics(
            pred_distances.cpu(),
            target_distances.cpu(),
            k=5,
        )
        
        return metrics
    
    def run_layer(
        self,
        layer: int,
        hidden_states: torch.Tensor,
        target_distances: torch.Tensor,
    ) -> List[ExperimentResult]:
        """
        Run experiment for a single layer.
        
        Trains and evaluates both Euclidean and Hyperbolic probes.
        """
        set_seed(self.seed)
        
        results = []
        
        for probe_type in ["euclidean", "hyperbolic", "lorentz"]:
            self.logger.info(f"Training {probe_type} probe for layer {layer}")
            
            # Create probe
            probe = create_probe(
                probe_type=probe_type,
                input_dim=self.input_dim,
                output_dim=self.config.output_dim,
                curvature=self.config.curvature,
                mdr_max_norm=self.config.mdr_max_norm,
            )
            
            # Train
            probe, final_loss, epochs = self.train_probe(
                probe=probe,
                hidden_states=hidden_states,
                target_distances=target_distances,
                probe_type=probe_type,
                layer=layer,
            )
            
            # Evaluate and get embeddings
            probe.eval()
            with torch.no_grad():
                embeddings = probe(hidden_states.to(self.device))
                pred_distances = probe.pairwise_distances(embeddings)
            
            metrics = compute_all_metrics(
                pred_distances.cpu(),
                target_distances.cpu() if torch.is_tensor(target_distances) else target_distances,
                k=5,
            )
            
            # Save embeddings for post-hoc visualization (Poincaré disk plots)
            if self.save_embeddings and self.output_dir is not None:
                import numpy as np
                self.output_dir.mkdir(parents=True, exist_ok=True)
                embed_path = self.output_dir / f"embeddings_layer{layer}_{probe_type}.npy"
                np.save(embed_path, embeddings.cpu().numpy())
                self.logger.info(f"Saved embeddings to {embed_path}")
            
            result = ExperimentResult(
                probe_type=probe_type,
                layer=layer,
                spearman_rho=metrics["spearman_rho"],
                avg_distortion=metrics["avg_distortion"],
                map_at_5=metrics["map_at_5"],
                stress=metrics["stress"],
                train_loss_final=final_loss,
                epochs_trained=epochs,
                config=asdict(self.config),
            )
            
            results.append(result)
            self.results.append(result)
            
            self.logger.info(
                f"Layer {layer} {probe_type}: "
                f"rho={result.spearman_rho:.4f}, "
                f"distortion={result.avg_distortion:.4f}"
            )
        
        return results
    
    def run_all_layers(
        self,
        activations_per_layer: Dict[int, torch.Tensor],
        target_distances: torch.Tensor,
    ) -> List[ExperimentResult]:
        """
        Run experiment across all layers.
        
        Args:
            activations_per_layer: Dict mapping layer -> hidden states [n, d]
            target_distances: Ground truth distances [n, n]
            
        Returns:
            All experiment results
        """
        for layer in self.layers:
            if layer not in activations_per_layer:
                self.logger.warning(f"Layer {layer} not found in activations")
                continue
            
            hidden_states = activations_per_layer[layer]
            self.run_layer(layer, hidden_states, target_distances)
        
        return self.results
    
    def save_results(self, output_path: Path) -> None:
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment": "hierarchy_h1",
            "config": asdict(self.config),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        euclidean = [r for r in self.results if r.probe_type == "euclidean"]
        hyperbolic = [r for r in self.results if r.probe_type == "hyperbolic"]
        
        def avg(lst, key):
            vals = [getattr(r, key) for r in lst]
            return sum(vals) / len(vals) if vals else 0
        
        return {
            "euclidean_avg_rho": avg(euclidean, "spearman_rho"),
            "hyperbolic_avg_rho": avg(hyperbolic, "spearman_rho"),
            "euclidean_avg_distortion": avg(euclidean, "avg_distortion"),
            "hyperbolic_avg_distortion": avg(hyperbolic, "avg_distortion"),
            "improvement_rho": avg(hyperbolic, "spearman_rho") - avg(euclidean, "spearman_rho"),
            "n_layers_tested": len(self.layers),
        }
