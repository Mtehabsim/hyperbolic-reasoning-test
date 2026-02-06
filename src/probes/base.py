"""
Base probe classes for distance preservation.

Defines the interface for Euclidean and hyperbolic probes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseProbe(nn.Module, ABC):
    """
    Abstract base class for distance-preserving probes.
    
    Probes learn to map high-dimensional hidden states to a lower-dimensional
    space while preserving pairwise distances.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        name: str = "base_probe",
    ):
        """
        Initialize probe.
        
        Args:
            input_dim: Dimension of input hidden states
            output_dim: Dimension of output embeddings
            name: Probe name for logging
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map hidden states to embedding space.
        
        Args:
            x: Hidden states [batch, input_dim]
            
        Returns:
            Embeddings [batch, output_dim]
        """
        pass
    
    @abstractmethod
    def pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances in embedding space.
        
        Args:
            z: Embeddings [batch, output_dim]
            
        Returns:
            Distance matrix [batch, batch]
        """
        pass
    
    def distortion_loss(
        self,
        pred_distances: torch.Tensor,
        target_distances: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute distortion loss for training.
        
        Loss = sum((pred - target)^2) / sum(target^2)  (normalized)
        or
        Loss = mean((pred - target)^2)  (unnormalized)
        
        Args:
            pred_distances: Predicted distances [n, n]
            target_distances: Target distances [n, n]
            normalize: Whether to normalize by target distances
            
        Returns:
            Scalar loss
        """
        diff_sq = (pred_distances - target_distances) ** 2
        
        if normalize:
            # Stress-like normalized loss
            target_sq = target_distances ** 2 + 1e-8
            return diff_sq.sum() / target_sq.sum()
        else:
            return diff_sq.mean()
    
    def ranking_loss(
        self,
        pred_distances: torch.Tensor,
        target_distances: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute ranking-based loss.
        
        Penalizes violations of relative ordering of distances.
        
        Args:
            pred_distances: Predicted distances [n, n]
            target_distances: Target distances [n, n]
            margin: Margin for ranking violations
            
        Returns:
            Scalar loss
        """
        n = pred_distances.shape[0]
        
        # Sample triplets for efficiency
        loss = torch.tensor(0.0, device=pred_distances.device)
        count = 0
        
        for i in range(min(n, 32)):  # Limit for efficiency
            for j in range(i + 1, min(n, 32)):
                for k in range(j + 1, min(n, 32)):
                    # Target ordering
                    t_ij = target_distances[i, j]
                    t_ik = target_distances[i, k]
                    t_jk = target_distances[j, k]
                    
                    p_ij = pred_distances[i, j]
                    p_ik = pred_distances[i, k]
                    p_jk = pred_distances[j, k]
                    
                    # Check if ordering is violated
                    if t_ij < t_ik:
                        loss = loss + torch.relu(p_ij - p_ik + margin)
                    else:
                        loss = loss + torch.relu(p_ik - p_ij + margin)
                    
                    count += 1
        
        return loss / max(count, 1)


class PairwiseProbe(BaseProbe):
    """
    Probe trained to preserve pairwise distances.
    
    Uses a linear projection followed by distance computation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True,
        learnable_scale: bool = True,
        name: str = "pairwise_probe",
    ):
        """
        Initialize pairwise probe.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            use_bias: Use bias in linear layer
            learnable_scale: Learn a scale factor for distances
            name: Probe name
        """
        super().__init__(input_dim, output_dim, name)
        
        self.projection = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.learnable_scale = learnable_scale
        
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights orthogonally for better distance preservation."""
        nn.init.orthogonal_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses."""
        pass
