"""
Geometric metrics for evaluating distance preservation.

Provides metrics for comparing predicted distances against ground truth:
- Spearman correlation
- Distortion (average/max stretch)
- Mean Average Precision at K
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy import stats


def compute_spearman(
    pred_distances: torch.Tensor,
    target_distances: torch.Tensor,
    flatten: bool = True,
) -> Dict[str, float]:
    """
    Compute Spearman rank correlation between predicted and target distances.
    
    Args:
        pred_distances: Predicted distance matrix [n, n] or [n, m]
        target_distances: Ground truth distance matrix [n, n] or [n, m]
        flatten: If True, flatten matrices to 1D
        
    Returns:
        Dictionary with 'rho' (correlation) and 'p_value'
    """
    pred = pred_distances.detach().cpu().numpy()
    target = target_distances.detach().cpu().numpy() if torch.is_tensor(target_distances) else target_distances
    
    if flatten:
        pred = pred.flatten()
        target = target.flatten()
    
    # Remove infinities and NaNs
    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    
    if len(pred) < 3:
        return {"rho": 0.0, "p_value": 1.0}
    
    rho, p_value = stats.spearmanr(pred, target)
    
    return {
        "rho": float(rho),
        "p_value": float(p_value),
    }


def compute_distortion(
    pred_distances: torch.Tensor,
    target_distances: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute distortion metrics.
    
    Distortion measures how well pairwise distances are preserved:
    - Average distortion: mean |pred/target - 1|
    - Max distortion: max |pred/target - 1|
    - Contraction: fraction where pred < target
    - Expansion: fraction where pred > target
    
    Args:
        pred_distances: Predicted distances
        target_distances: Target distances
        eps: Small constant to avoid division by zero
        
    Returns:
        Dictionary with distortion metrics
    """
    pred = pred_distances.detach().cpu().numpy().flatten()
    target = target_distances if isinstance(target_distances, np.ndarray) else target_distances.detach().cpu().numpy()
    target = target.flatten()
    
    # Filter out self-distances (zero) and invalid values
    mask = (target > eps) & np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    
    if len(pred) == 0:
        return {
            "avg_distortion": 0.0,
            "max_distortion": 0.0,
            "contraction_ratio": 0.0,
            "expansion_ratio": 0.0,
        }
    
    # Ratio of pred to target
    ratios = pred / (target + eps)
    
    # Distortion = |ratio - 1|
    distortions = np.abs(ratios - 1.0)
    
    return {
        "avg_distortion": float(np.mean(distortions)),
        "max_distortion": float(np.max(distortions)),
        "contraction_ratio": float(np.mean(ratios < 1.0)),
        "expansion_ratio": float(np.mean(ratios > 1.0)),
    }


def compute_map_at_k(
    pred_distances: torch.Tensor,
    target_distances: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute Mean Average Precision at K.
    
    For each point, checks if the K nearest neighbors in predicted space
    match the K nearest neighbors in target space.
    
    Args:
        pred_distances: [n, n] predicted distance matrix
        target_distances: [n, n] target distance matrix
        k: Number of nearest neighbors to consider
        
    Returns:
        MAP@K score (0 to 1)
    """
    pred = pred_distances.detach().cpu().numpy()
    target = target_distances if isinstance(target_distances, np.ndarray) else target_distances.detach().cpu().numpy()
    
    n = pred.shape[0]
    if n <= k:
        k = n - 1
    
    if k <= 0:
        return 0.0
    
    ap_scores = []
    
    for i in range(n):
        # Get K nearest neighbors (excluding self)
        pred_row = pred[i].copy()
        target_row = target[i].copy()
        
        pred_row[i] = np.inf  # Exclude self
        target_row[i] = np.inf
        
        pred_knn = set(np.argsort(pred_row)[:k])
        target_knn = set(np.argsort(target_row)[:k])
        
        # Precision = intersection / k
        precision = len(pred_knn & target_knn) / k
        ap_scores.append(precision)
    
    return float(np.mean(ap_scores))


def compute_stress(
    pred_distances: torch.Tensor,
    target_distances: torch.Tensor,
) -> float:
    """
    Compute Kruskal's stress (MDS quality metric).
    
    Stress = sqrt(sum((pred - target)^2) / sum(target^2))
    
    Args:
        pred_distances: Predicted distances
        target_distances: Target distances
        
    Returns:
        Stress value (lower is better, 0 is perfect)
    """
    pred = pred_distances.detach().cpu().numpy().flatten()
    target = target_distances if isinstance(target_distances, np.ndarray) else target_distances.detach().cpu().numpy()
    target = target.flatten()
    
    mask = np.isfinite(pred) & np.isfinite(target) & (target > 0)
    pred = pred[mask]
    target = target[mask]
    
    if len(pred) == 0 or np.sum(target ** 2) == 0:
        return 0.0
    
    numerator = np.sum((pred - target) ** 2)
    denominator = np.sum(target ** 2)
    
    return float(np.sqrt(numerator / denominator))


def compute_all_metrics(
    pred_distances: torch.Tensor,
    target_distances: torch.Tensor,
    k: int = 5,
) -> Dict[str, float]:
    """
    Compute all distance preservation metrics.
    
    Args:
        pred_distances: Predicted distances
        target_distances: Target distances
        k: K for MAP@K
        
    Returns:
        Dictionary with all metrics
    """
    spearman = compute_spearman(pred_distances, target_distances)
    distortion = compute_distortion(pred_distances, target_distances)
    map_k = compute_map_at_k(pred_distances, target_distances, k=k)
    stress = compute_stress(pred_distances, target_distances)
    
    return {
        "spearman_rho": spearman["rho"],
        "spearman_p": spearman["p_value"],
        **distortion,
        f"map_at_{k}": map_k,
        "stress": stress,
    }
