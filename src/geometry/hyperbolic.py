"""
Hyperbolic geometry utilities.

Implements Poincaré ball and Lorentz (hyperboloid) models with:
- Exponential/logarithmic maps
- Distance computations  
- Maximum Distance Rescaling (MDR) for numerical stability
- Centroid computations

All operations are numerically stable for float32.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# Small epsilon for numerical stability
EPS = 1e-8


# =============================================================================
# Maximum Distance Rescaling (MDR)
# =============================================================================

def maximum_distance_rescaling(
    x: torch.Tensor,
    max_norm: float = 15.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Maximum Distance Rescaling for numerical stability.
    
    Rescales vectors so that maximum norm is bounded, preventing
    overflow in exponential map and distance computations.
    
    From: "Numerical stability in hyperbolic geometry" recommendations.
    
    Args:
        x: Input tensor
        max_norm: Maximum allowed norm
        dim: Dimension to compute norm over
        
    Returns:
        Rescaled tensor with norms clamped to max_norm
    """
    norms = torch.norm(x, p=2, dim=dim, keepdim=True)
    scale = torch.clamp(norms / max_norm, min=1.0)
    return x / scale


def clip_by_norm(
    x: torch.Tensor,
    max_norm: float,
    dim: int = -1,
) -> torch.Tensor:
    """
    Clip tensor by norm (alternative to MDR).
    
    Args:
        x: Input tensor
        max_norm: Maximum norm
        dim: Dimension
        
    Returns:
        Clipped tensor
    """
    norms = torch.norm(x, p=2, dim=dim, keepdim=True)
    desired = torch.clamp(norms, max=max_norm)
    return x * (desired / (norms + EPS))


# =============================================================================
# Poincaré Ball Model
# =============================================================================

class PoincareBall:
    """
    Poincaré ball model of hyperbolic space.
    
    The Poincaré ball is the unit ball B^n = {x ∈ R^n : ||x|| < 1}
    with the Riemannian metric g_x = (2/(1-||x||^2))^2 * I.
    """
    
    def __init__(self, curvature: float = 1.0):
        """
        Initialize Poincaré ball.
        
        Args:
            curvature: Negative curvature magnitude (c > 0, actual curvature = -c)
        """
        self.c = curvature
        self.sqrt_c = curvature ** 0.5
    
    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor at point x."""
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        return 2.0 / (1.0 - self.c * x_norm_sq + EPS)
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.
        
        x ⊕ y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / 
                (1 + 2c<x,y> + c²||x||²||y||²)
        """
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * self.c * xy + self.c * y_norm_sq) * x + \
              (1 - self.c * x_norm_sq) * y
        denom = 1 + 2 * self.c * xy + self.c ** 2 * x_norm_sq * y_norm_sq
        
        return num / (denom + EPS)
    
    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from origin (tangent space at 0) to Poincaré ball.
        
        exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
        
        Args:
            v: Tangent vector at origin
            
        Returns:
            Point on Poincaré ball
        """
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=EPS)
        
        # tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v
        scale = torch.tanh(self.sqrt_c * v_norm) / (self.sqrt_c * v_norm)
        
        return scale * v
    
    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map from Poincaré ball to tangent space at origin.
        
        log_0(x) = arctanh(sqrt(c) * ||x||) * x / (sqrt(c) * ||x||)
        
        Args:
            x: Point on Poincaré ball
            
        Returns:
            Tangent vector at origin
        """
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=EPS, max=1.0 - EPS)
        
        # arctanh(sqrt(c) * ||x||) / (sqrt(c) * ||x||) * x
        scale = torch.arctanh(self.sqrt_c * x_norm) / (self.sqrt_c * x_norm + EPS)
        
        return scale * x
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Poincaré distance between two points.
        
        d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||−x ⊕ y||)
        
        Args:
            x: First point [n, d] or [d]
            y: Second point [m, d] or [d]
            
        Returns:
            Distance scalar or matrix
        """
        # Handle broadcasting for pairwise distances
        if x.dim() == 2 and y.dim() == 2:
            # Pairwise: [n, d] vs [m, d] -> [n, m]
            n, d = x.shape
            m = y.shape[0]
            
            x_exp = x.unsqueeze(1)  # [n, 1, d]
            y_exp = y.unsqueeze(0)  # [1, m, d]
            
            diff = self.mobius_add(-x_exp, y_exp)  # [n, m, d]
            diff_norm = torch.norm(diff, p=2, dim=-1)  # [n, m]
        else:
            diff = self.mobius_add(-x, y)
            diff_norm = torch.norm(diff, p=2, dim=-1)
        
        diff_norm = torch.clamp(diff_norm, min=EPS, max=1.0 / self.sqrt_c - EPS)
        
        dist = (2.0 / self.sqrt_c) * torch.arctanh(self.sqrt_c * diff_norm)
        
        return dist
    
    def project(self, x: torch.Tensor, max_radius: float = 1.0 - 1e-5) -> torch.Tensor:
        """
        Project point onto Poincaré ball (clamp to radius).
        
        Args:
            x: Point (possibly outside ball)
            max_radius: Maximum radius (< 1)
            
        Returns:
            Projected point inside ball
        """
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        clamped_norm = torch.clamp(x_norm, max=max_radius)
        return x * (clamped_norm / (x_norm + EPS))


def poincare_pairwise_distance(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Compute pairwise Poincaré distances.
    
    Args:
        x: Points [n, d]
        y: Points [m, d] (optional, defaults to x)
        curvature: Curvature magnitude
        
    Returns:
        Distance matrix [n, m] or [n, n]
    """
    ball = PoincareBall(curvature=curvature)
    
    if y is None:
        y = x
    
    return ball.distance(x, y)


# =============================================================================
# Lorentz (Hyperboloid) Model
# =============================================================================

class LorentzModel:
    """
    Lorentz (hyperboloid) model of hyperbolic space.
    
    The hyperboloid is H^n = {x ∈ R^{n+1} : <x, x>_L = -1/c, x_0 > 0}
    where <x, y>_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n
    """
    
    def __init__(self, curvature: float = 1.0):
        """
        Initialize Lorentz model.
        
        Args:
            curvature: Curvature magnitude (c > 0)
        """
        self.c = curvature
        self.sqrt_c = curvature ** 0.5
    
    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Minkowski inner product: <x, y>_L = -x_0*y_0 + sum(x_i*y_i)
        """
        # Time component (first dim)
        time = -x[..., :1] * y[..., :1]
        # Space components
        space = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
        return time + space
    
    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Minkowski norm: sqrt(|<x, x>_L|)"""
        inner = self.minkowski_inner(x, x)
        return torch.sqrt(torch.abs(inner) + EPS)
    
    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from tangent space at origin to hyperboloid.
        
        The origin on the hyperboloid is (1/sqrt(c), 0, 0, ..., 0).
        
        Args:
            v: Tangent vector [batch, d] (Euclidean, d-dimensional)
            
        Returns:
            Point on hyperboloid [batch, d+1]
        """
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=EPS)
        
        # Time component
        t = torch.cosh(self.sqrt_c * v_norm) / self.sqrt_c  # [batch, 1]
        
        # Space components
        s = torch.sinh(self.sqrt_c * v_norm) / (self.sqrt_c * v_norm) * v  # [batch, d]
        
        return torch.cat([t, s], dim=-1)
    
    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map from hyperboloid to tangent space at origin.
        
        Args:
            x: Point on hyperboloid [batch, d+1]
            
        Returns:
            Tangent vector [batch, d]
        """
        t = x[..., :1]  # Time component [batch, 1]
        s = x[..., 1:]  # Space components [batch, d]
        
        s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        s_norm = torch.clamp(s_norm, min=EPS)
        
        # acosh(sqrt(c) * t) / (sqrt(c) * ||s||) * s
        theta = torch.acosh(torch.clamp(self.sqrt_c * t, min=1.0 + EPS))
        scale = theta / (self.sqrt_c * s_norm + EPS)
        
        return scale * s
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Lorentz distance between two points.
        
        d(x, y) = (1/sqrt(c)) * acosh(-c * <x, y>_L)
        
        Args:
            x: Points [n, d+1]
            y: Points [m, d+1]
            
        Returns:
            Distance matrix [n, m]
        
        Note:
            Uses aggressive clamping (min=1.0 + 1e-5) to prevent NaN from acosh.
            The domain of acosh is [1, inf), but floating point errors can produce
            values slightly below 1.0, especially on GPU with mixed precision.
        """
        if x.dim() == 2 and y.dim() == 2:
            # Pairwise computation
            # Time: -x_0 * y_0
            time = -x[:, :1] @ y[:, :1].T  # [n, m]
            # Space: sum of x_i * y_i
            space = x[:, 1:] @ y[:, 1:].T  # [n, m]
            inner = time + space
        else:
            inner = self.minkowski_inner(x, y)
        
        # acosh(-c * inner) / sqrt(c)
        # Use aggressive clamping to prevent NaN - 1e-5 is safe margin for float32
        arg = torch.clamp(-self.c * inner, min=1.0 + 1e-5, max=1e6)
        dist = torch.acosh(arg) / self.sqrt_c
        
        # Handle any remaining NaN/inf by replacing with 0
        dist = torch.where(torch.isfinite(dist), dist, torch.zeros_like(dist))
        
        return dist.squeeze()
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point onto hyperboloid.
        
        Args:
            x: Point [batch, d+1]
            
        Returns:
            Projected point on hyperboloid
        """
        space = x[..., 1:]
        space_norm_sq = (space ** 2).sum(dim=-1, keepdim=True)
        
        # t = sqrt(||s||^2 + 1/c)
        t = torch.sqrt(space_norm_sq + 1.0 / self.c)
        
        return torch.cat([t, space], dim=-1)


def lorentz_pairwise_distance(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Compute pairwise Lorentz distances.
    
    Args:
        x: Points [n, d+1] on hyperboloid
        y: Points [m, d+1] (optional)
        curvature: Curvature magnitude
        
    Returns:
        Distance matrix
    """
    model = LorentzModel(curvature=curvature)
    
    if y is None:
        y = x
    
    return model.distance(x, y)
