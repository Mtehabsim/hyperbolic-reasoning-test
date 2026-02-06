"""
Euclidean probes for distance preservation.

Implements linear probes with L2 distance computation in Euclidean space.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import PairwiseProbe
from src.geometry.euclidean import pairwise_l2_distance, pairwise_cosine_distance


class EuclideanLinearProbe(nn.Module):
    """
    Simple linear probe for depth/class prediction.
    
    Maps hidden states to a scalar or class logits.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize linear probe.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (1 for regression, n_classes for classification)
            hidden_dim: Optional hidden layer dimension
        """
        super().__init__()
        
        if hidden_dim is not None:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.model = nn.Linear(input_dim, output_dim)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class EuclideanPairwiseProbe(PairwiseProbe):
    """
    Pairwise distance probe using Euclidean (L2) distances.
    
    Projects hidden states to a lower dimension and computes L2 distances.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        use_bias: bool = True,
        learnable_scale: bool = True,
    ):
        """
        Initialize Euclidean pairwise probe.
        
        Args:
            input_dim: Input dimension
            output_dim: Embedding dimension
            use_bias: Use bias in projection
            learnable_scale: Learn a scale factor
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=use_bias,
            learnable_scale=learnable_scale,
            name="euclidean_pairwise",
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to Euclidean embedding space.
        
        Args:
            x: Hidden states [batch, input_dim]
            
        Returns:
            Euclidean embeddings [batch, output_dim]
        """
        return self.projection(x)
    
    def pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise L2 distances.
        
        Args:
            z: Embeddings [batch, output_dim]
            
        Returns:
            Distance matrix [batch, batch]
        """
        distances = pairwise_l2_distance(z)
        return self.scale * distances


class EuclideanCosineProbe(PairwiseProbe):
    """
    Pairwise probe using cosine distances.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        use_bias: bool = True,
        learnable_scale: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=use_bias,
            learnable_scale=learnable_scale,
            name="cosine_pairwise",
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to embedding space."""
        return self.projection(x)
    
    def pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine distances."""
        distances = pairwise_cosine_distance(z)
        return self.scale * distances
