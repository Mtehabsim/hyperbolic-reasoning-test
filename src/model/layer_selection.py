"""
Dynamic layer selection utilities.

Implements layer selection strategies based on drift.
"""

from typing import Dict, List, Optional, Tuple

import torch


def compute_layer_drift(
    layer_activations: Dict[int, torch.Tensor],
) -> Dict[int, float]:
    """
    Compute L2 drift between consecutive layers.
    
    Args:
        layer_activations: Dict mapping layer -> activations [n_samples, d_model]
        
    Returns:
        Dict mapping layer -> drift score (L2 norm of diff from prev layer)
    """
    layers = sorted(layer_activations.keys())
    drifts = {}
    
    for i in range(1, len(layers)):
        prev_layer = layers[i - 1]
        curr_layer = layers[i]
        
        prev_acts = layer_activations[prev_layer]
        curr_acts = layer_activations[curr_layer]
        
        # Mean L2 diff across samples
        diff = curr_acts - prev_acts
        drift = diff.norm(dim=-1).mean().item()
        
        drifts[curr_layer] = drift
    
    return drifts


def select_layers_by_max_drift(
    layer_activations: Dict[int, torch.Tensor],
    top_k: int = 3,
    threshold: float = 0.1,
) -> List[int]:
    """
    Select top-k layers with highest L2 drift (unsupervised).
    
    If top layers have similar drift scores (within threshold), aggregate them.
    From LSD methodology + CLAP insight (multi-layer better than single-layer).
    
    Args:
        layer_activations: Dict mapping layer -> activations
        top_k: Number of top layers to consider
        threshold: Threshold for including layers (fraction of max drift)
        
    Returns:
        List of selected layer indices
    """
    drifts = compute_layer_drift(layer_activations)
    
    if not drifts:
        # Fallback to middle layer if only one layer
        return list(layer_activations.keys())[:1]
    
    # Sort by drift value (descending)
    sorted_layers = sorted(drifts.keys(), key=lambda l: drifts[l], reverse=True)
    
    # Get top-k
    top_layers = sorted_layers[:min(top_k, len(sorted_layers))]
    
    if not top_layers:
        return list(layer_activations.keys())[-1:]  # Last layer fallback
    
    # Filter to only include layers within threshold of max
    max_drift = drifts[top_layers[0]]
    close_layers = [l for l in top_layers if drifts[l] >= max_drift * (1 - threshold)]
    
    return sorted(close_layers)  # Return in ascending layer order


def aggregate_layer_activations(
    layer_activations: Dict[int, torch.Tensor],
    selected_layers: List[int],
) -> torch.Tensor:
    """
    Average activations from selected layers.
    
    Args:
        layer_activations: Dict mapping layer -> activations
        selected_layers: List of layer indices to aggregate
        
    Returns:
        Aggregated activations [n_samples, d_model]
    """
    if not selected_layers:
        raise ValueError("selected_layers cannot be empty")
    
    stacked = torch.stack([layer_activations[l] for l in selected_layers])
    return stacked.mean(dim=0)


def select_layers_by_threshold(
    layer_activations: Dict[int, torch.Tensor],
    percentile: float = 90.0,
) -> List[int]:
    """
    Select layers where drift is above a percentile threshold.
    
    Args:
        layer_activations: Dict mapping layer -> activations
        percentile: Percentile threshold (e.g., 90 = top 10%)
        
    Returns:
        List of selected layer indices
    """
    drifts = compute_layer_drift(layer_activations)
    
    if not drifts:
        return list(layer_activations.keys())[:1]
    
    drift_values = list(drifts.values())
    threshold = torch.quantile(
        torch.tensor(drift_values), 
        percentile / 100.0
    ).item()
    
    selected = [l for l, d in drifts.items() if d >= threshold]
    
    return sorted(selected) if selected else [max(drifts, key=drifts.get)]
