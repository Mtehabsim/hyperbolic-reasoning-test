"""
Euclidean geometry utilities for distance computation and embedding.

Provides pairwise distance computations and PCA projection.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def pairwise_l2_distance(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute pairwise L2 (Euclidean) distances.
    
    Args:
        x: Tensor of shape [n, d]
        y: Optional tensor of shape [m, d]. If None, compute x vs x.
        
    Returns:
        Distance matrix of shape [n, m] or [n, n]
    """
    if y is None:
        y = x
    
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [n, 1]
    y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)  # [m, 1]
    
    dist_sq = x_norm_sq + y_norm_sq.T - 2 * x @ y.T
    
    # Clamp to avoid numerical issues with sqrt
    dist_sq = torch.clamp(dist_sq, min=0.0)
    
    # Add epsilon to avoid infinite gradient at 0
    return torch.sqrt(dist_sq + 1e-12)


def pairwise_cosine_distance(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute pairwise cosine distances (1 - cosine_similarity).
    
    Args:
        x: Tensor of shape [n, d]
        y: Optional tensor of shape [m, d]. If None, compute x vs x.
        
    Returns:
        Distance matrix of shape [n, m] or [n, n]
    """
    if y is None:
        y = x
    
    # Normalize
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    
    # Cosine similarity
    cos_sim = x_norm @ y_norm.T
    
    # Cosine distance = 1 - similarity
    return 1 - cos_sim


def pairwise_squared_distance(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances.
    
    Args:
        x: Tensor of shape [n, d]
        y: Optional tensor of shape [m, d]. If None, compute x vs x.
        
    Returns:
        Squared distance matrix of shape [n, m] or [n, n]
    """
    if y is None:
        y = x
    
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)
    
    dist_sq = x_norm_sq + y_norm_sq.T - 2 * x @ y.T
    
    return torch.clamp(dist_sq, min=0.0)


def pca_projection(
    x: torch.Tensor,
    n_components: int,
    center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project data using PCA.
    
    Args:
        x: Data tensor of shape [n, d]
        n_components: Number of components to keep
        center: Whether to center the data
        
    Returns:
        (projected_data, components, mean)
    """
    n, d = x.shape
    
    # Center data
    if center:
        mean = x.mean(dim=0, keepdim=True)
        x_centered = x - mean
    else:
        mean = torch.zeros(1, d, device=x.device, dtype=x.dtype)
        x_centered = x
    
    # SVD
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    
    # Take top components
    k = min(n_components, d, n)
    components = Vh[:k]  # [k, d]
    projected = x_centered @ components.T  # [n, k]
    
    return projected, components, mean.squeeze(0)


def project_to_pca(
    x: torch.Tensor,
    components: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """
    Project new data using existing PCA components.
    
    Args:
        x: New data of shape [n, d]
        components: PCA components of shape [k, d]
        mean: Mean vector of shape [d]
        
    Returns:
        Projected data of shape [n, k]
    """
    x_centered = x - mean.unsqueeze(0)
    return x_centered @ components.T


def normalize_embeddings(x: torch.Tensor, p: float = 2, dim: int = -1) -> torch.Tensor:
    """
    Normalize embeddings to unit norm.
    
    Args:
        x: Embedding tensor
        p: Norm order (default: 2 for L2)
        dim: Dimension to normalize
        
    Returns:
        Normalized tensor
    """
    return F.normalize(x, p=p, dim=dim)
