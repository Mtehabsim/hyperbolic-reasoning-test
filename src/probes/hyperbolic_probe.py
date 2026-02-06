"""
Hyperbolic probes for distance preservation.

Implements probes that map to Poincaré ball or Lorentz (hyperboloid) manifold
for better preservation of hierarchical structure.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import PairwiseProbe
from src.geometry.hyperbolic import (
    PoincareBall,
    LorentzModel,
    maximum_distance_rescaling,
    EPS,
)


class HyperbolicPairwiseProbe(PairwiseProbe):
    """
    Pairwise probe using Poincaré ball distances.
    
    Projects hidden states to Poincaré ball and computes hyperbolic distances.
    Uses MDR rescaling for numerical stability.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        curvature: float = 1.0,
        mdr_max_norm: float = 15.0,
        use_bias: bool = True,
        learnable_scale: bool = True,
    ):
        """
        Initialize Poincaré probe.
        
        Args:
            input_dim: Input dimension
            output_dim: Embedding dimension (Poincaré ball)
            curvature: Curvature magnitude (c > 0)
            mdr_max_norm: Max norm for MDR rescaling
            use_bias: Use bias in projection
            learnable_scale: Learn a scale factor
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=use_bias,
            learnable_scale=learnable_scale,
            name="poincare_pairwise",
        )
        
        self.curvature = curvature
        self.mdr_max_norm = mdr_max_norm
        self.manifold = PoincareBall(curvature=curvature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map hidden states to Poincaré ball.
        
        1. Linear projection to tangent space at origin
        2. MDR rescaling for numerical stability
        3. Exponential map to Poincaré ball
        
        Args:
            x: Hidden states [batch, input_dim]
            
        Returns:
            Points on Poincaré ball [batch, output_dim]
        """
        # Project to tangent space
        v = self.projection(x)  # [batch, output_dim]
        
        # MDR rescaling
        v = maximum_distance_rescaling(v, max_norm=self.mdr_max_norm)
        
        # Exponential map to Poincaré ball
        z = self.manifold.expmap0(v)
        
        # Ensure we're inside the ball
        z = self.manifold.project(z)
        
        return z
    
    def pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Poincaré distances.
        
        Args:
            z: Points on Poincaré ball [batch, output_dim]
            
        Returns:
            Distance matrix [batch, batch]
        """
        distances = self.manifold.distance(z, z)
        return self.scale * distances


class LorentzProbe(PairwiseProbe):
    """
    Pairwise probe using Lorentz (hyperboloid) model.
    
    More stable than Poincaré for high dimensions (d >= 5).
    Maps to the upper sheet of the hyperboloid.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        curvature: float = 1.0,
        mdr_max_norm: float = 15.0,
        use_bias: bool = True,
        learnable_scale: bool = True,
    ):
        """
        Initialize Lorentz probe.
        
        Args:
            input_dim: Input dimension
            output_dim: Space dimension (hyperboloid is in R^{output_dim+1})
            curvature: Curvature magnitude
            mdr_max_norm: Max norm for MDR
            use_bias: Use bias
            learnable_scale: Learn scale factor
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=use_bias,
            learnable_scale=learnable_scale,
            name="lorentz_pairwise",
        )
        
        self.curvature = curvature
        self.mdr_max_norm = mdr_max_norm
        self.manifold = LorentzModel(curvature=curvature)
        
        # Store actual hyperboloid dimension (d+1)
        self.hyperboloid_dim = output_dim + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map hidden states to hyperboloid.
        
        1. Linear projection to tangent space
        2. MDR rescaling  
        3. Exponential map to hyperboloid
        
        Args:
            x: Hidden states [batch, input_dim]
            
        Returns:
            Points on hyperboloid [batch, output_dim + 1]
        """
        # Project to tangent space
        v = self.projection(x)  # [batch, output_dim]
        
        # MDR rescaling
        v = maximum_distance_rescaling(v, max_norm=self.mdr_max_norm)
        
        # Exponential map to hyperboloid
        z = self.manifold.expmap0(v)  # [batch, output_dim + 1]
        
        # Project onto hyperboloid (ensure constraint satisfied)
        z = self.manifold.project(z)
        
        return z
    
    def pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Lorentz distances.
        
        Args:
            z: Points on hyperboloid [batch, output_dim + 1]
            
        Returns:
            Distance matrix [batch, batch]
        """
        distances = self.manifold.distance(z, z)
        return self.scale * distances



class RobustHyperbolicProbe(PairwiseProbe):
    """
    Robust hyperbolic probe matching src_old methodology.
    
    Includes:
    - LayerNorm (critical for stability)
    - Spectral normalization
    - Sigmoid scaling
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        curvature: float = 1.0,
        use_layer_norm: bool = True,
    ):
        """
        Initialize robust probe.
        
        Args:
            input_dim: Input dimension
            output_dim: Embedding dimension
            curvature: Curvature (c)
            use_layer_norm: Whether to use LayerNorm
        """
        # Initialize parent with dummy values, we'll override projection
        super().__init__(input_dim, output_dim, use_bias=False, learnable_scale=False)
        
        self.curvature = curvature
        self.manifold = PoincareBall(curvature=curvature)
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.register_buffer('input_scale', torch.tensor(0.01))
            
        # Override projection with Spectral Norm linear layer
        self.projection = nn.utils.spectral_norm(nn.Linear(input_dim, output_dim, bias=False))
        
        # Learnable log scale for converting to hyperbolic radius
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization and scaling."""
        if self.use_layer_norm:
            x = self.input_norm(x)
        else:
            x = x * self.input_scale
            
        v = self.projection(x)
        
        # Scale to ensure points are within reasonable radius
        scale = torch.sigmoid(self.log_scale) * 0.5
        v = v * scale
        
        # Expmap to Poincare Ball
        z = self.manifold.expmap0(v)
        
        return z

    def pairwise_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances."""
        return self.manifold.distance(z, z)


def create_probe(
    probe_type: str,
    input_dim: int,
    output_dim: int = 16,
    curvature: float = 1.0,
    mdr_max_norm: float = 15.0,
) -> PairwiseProbe:
    """
    Factory function to create probes by type.
    
    Args:
        probe_type: 'euclidean', 'hyperbolic'/'poincare', or 'lorentz'
        input_dim: Input dimension
        output_dim: Output dimension
        curvature: Curvature for hyperbolic probes
        mdr_max_norm: MDR max norm
        
    Returns:
        Probe instance
    """
    from .euclidean_probe import EuclideanPairwiseProbe
    
    if probe_type in ("euclidean", "l2"):
        return EuclideanPairwiseProbe(input_dim=input_dim, output_dim=output_dim)
    
    elif probe_type in ("hyperbolic", "poincare", "poincaré"):
        # Use RobustHyperbolicProbe by default for stability
        return RobustHyperbolicProbe(
            input_dim=input_dim,
            output_dim=output_dim,
            curvature=curvature,
        )
    
    elif probe_type == "lorentz":
        return LorentzProbe(
            input_dim=input_dim,
            output_dim=output_dim,
            curvature=curvature,
            mdr_max_norm=mdr_max_norm,
        )
    
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
