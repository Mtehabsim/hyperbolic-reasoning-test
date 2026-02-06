"""
Token ablation experiment (H2: Thinking tokens vs all-token pooling).

Tests whether reasoning concentrates at sparse "thinking tokens".
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.model.token_selector import TokenSelector, find_thinking_positions
from src.probes import create_probe
from src.geometry.metrics import compute_all_metrics
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed


@dataclass
class TokenAblationResult:
    """Result from token ablation experiment."""
    selection_method: str
    layer: int
    probe_type: str
    spearman_rho: float
    avg_distortion: float
    n_tokens_used: float  # Average tokens used
    improvement_vs_all: float
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TokenAblationExperiment:
    """
    H2 Experiment: Compare token selection strategies.
    
    Tests:
    - thinking_tokens: Select only reasoning-related tokens
    - all_pool: Mean pool all tokens
    - last_token: Use only last token
    - attention_weighted: Weight by attention
    """
    
    METHODS = ["thinking_tokens", "attention_weighted", "all_pool", "last_token", "random"]
    
    def __init__(
        self,
        input_dim: int,
        layers: List[int],
        probe_type: str = "hyperbolic",
        output_dim: int = 16,
        curvature: float = 0.5,
        device: str = "cuda",
        seed: int = 42,
    ):
        """
        Initialize experiment.
        
        Args:
            input_dim: Hidden state dimension
            layers: Layers to test
            probe_type: Probe type to use
            output_dim: Probe output dimension
            device: Device
            seed: Random seed
        """
        self.input_dim = input_dim
        self.layers = layers
        self.probe_type = probe_type
        self.output_dim = output_dim
        self.curvature = curvature
        self.device = device
        self.seed = seed
        self.logger = get_logger()
        
        self.results: List[TokenAblationResult] = []
    
    def pool_activations(
        self,
        hidden_states: torch.Tensor,
        tokens_list: List[List[str]],
        method: str,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Pool hidden states using specified method.
        
        Args:
            hidden_states: [n_samples, seq_len, d_model]
            tokens_list: List of token lists per sample
            method: Selection method
            attention_weights: Optional attention for attention-based
            
        Returns:
            (pooled_states [n_samples, d_model], avg_tokens_used)
        """
        selector = TokenSelector(method=method, fallback="last_token")
        
        pooled = []
        total_tokens = 0
        
        for i, (hidden, tokens) in enumerate(zip(hidden_states, tokens_list)):
            attn = attention_weights[i] if attention_weights is not None else None
            
            pooled_hidden, positions, _ = selector.select(
                tokens=tokens,
                hidden_states=hidden,
                attention_weights=attn,
            )
            
            pooled.append(pooled_hidden)
            total_tokens += len(positions)
        
        avg_tokens = total_tokens / len(hidden_states)
        pooled_tensor = torch.stack(pooled)
        
        # DEBUG: Check for NaNs
        if torch.isnan(pooled_tensor).any():
            self.logger.error(f"NaNs detected in pooled embeddings for method {method}!")
            self.logger.error(f"Pooled shape: {pooled_tensor.shape}")
            self.logger.error(f"Sample NaNs: {torch.isnan(pooled_tensor).sum()}")
            # Check individual samples
            for i, p in enumerate(pooled):
                if torch.isnan(p).any():
                    self.logger.error(f"Sample {i} has NaNs. Tokens used: {len(tokens_list[i])}")
                    break
        
        # Log stats
        norms = torch.norm(pooled_tensor, dim=1)
        self.logger.info(f"Method {method}: Pooled stats - Norm: {norms.mean():.2f} +/- {norms.std():.2f}, Max: {pooled_tensor.abs().max():.2f}")
        
        return pooled_tensor, avg_tokens

    
    def run_method(
        self,
        method: str,
        layer: int,
        hidden_states: torch.Tensor,
        tokens_list: List[List[str]],
        target_distances: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> TokenAblationResult:
        """
        Run experiment for a single method.
        """
        set_seed(self.seed)
        
        # Pool activations
        pooled, avg_tokens = self.pool_activations(
            hidden_states=hidden_states,
            tokens_list=tokens_list,
            method=method,
            attention_weights=attention_weights,
        )
        
        probe = create_probe(
            probe_type=self.probe_type,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            curvature=self.curvature,
        ).to(self.device)
        
        pooled = pooled.to(self.device)
        target_distances = target_distances.to(self.device)
        
        # Training with early stopping
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        n_epochs = 100
        early_stopping_patience = 10
        
        for epoch in range(n_epochs):
            probe.train()
            optimizer.zero_grad()
            embeddings = probe(pooled)
            pred_dist = probe.pairwise_distances(embeddings)
            loss = probe.distortion_loss(pred_dist, target_distances)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_loss = loss.item()
            scheduler.step(current_loss)
            
            # Early stopping
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                break
        
        # Restore best weights
        if best_state is not None:
            probe.load_state_dict(best_state)
        
        # Evaluate
        probe.eval()
        with torch.no_grad():
            embeddings = probe(pooled)
            pred_dist = probe.pairwise_distances(embeddings)
        
        metrics = compute_all_metrics(pred_dist.cpu(), target_distances.cpu())
        
        return TokenAblationResult(
            selection_method=method,
            layer=layer,
            probe_type=self.probe_type,
            spearman_rho=metrics["spearman_rho"],
            avg_distortion=metrics["avg_distortion"],
            n_tokens_used=avg_tokens,
            improvement_vs_all=0.0,  # Computed later
            config={"output_dim": self.output_dim, "layer": layer},
        )
    
    def run_layer(
        self,
        layer: int,
        hidden_states: torch.Tensor,
        tokens_list: List[List[str]],
        target_distances: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> List[TokenAblationResult]:
        """Run all methods for a layer."""
        layer_results = []
        
        for method in self.METHODS:
            self.logger.info(f"Testing method '{method}' for layer {layer}")
            
            result = self.run_method(
                method=method,
                layer=layer,
                hidden_states=hidden_states,
                tokens_list=tokens_list,
                target_distances=target_distances,
                attention_weights=attention_weights,
            )
            
            layer_results.append(result)
            self.results.append(result)
        
        # Compute improvement vs all_pool
        all_pool_rho = next(
            (r.spearman_rho for r in layer_results if r.selection_method == "all_pool"),
            0.0
        )
        
        for r in layer_results:
            r.improvement_vs_all = r.spearman_rho - all_pool_rho
        
        return layer_results
    
    def save_results(self, output_path: Path) -> None:
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment": "token_ablation_h2",
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        by_method = {}
        for r in self.results:
            if r.selection_method not in by_method:
                by_method[r.selection_method] = []
            by_method[r.selection_method].append(r.spearman_rho)
        
        return {
            method: sum(vals) / len(vals) if vals else 0
            for method, vals in by_method.items()
        }
