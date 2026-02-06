#!/usr/bin/env python3
"""
Compute layer-wise activation statistics for anisotropy analysis.

This script computes:
1. Mean and std of activation norms across samples
2. Singular value spectrum (concentration in top-k)
3. Participation ratio (effective dimensionality / anisotropy metric)
4. Mean pairwise Euclidean distances

These metrics help explain WHY Euclidean probes fail at late layers:
- Hypothesis: Late layers become anisotropic (collapsed to low-dim subspace)
- This breaks Euclidean distance assumptions but hyperbolic can still embed

Usage:
    python scripts/compute_layer_statistics.py \
        --cached-activations outputs/activations/deepseek_prontoqa.pt \
        --output outputs/layer_statistics_deepseek_prontoqa.json

Output format:
    {
        "model": "deepseek_7b",
        "dataset": "prontoqa",
        "layers": {
            "8": {
                "mean_norm": 12.34,
                "std_norm": 1.56,
                "top10_sv_ratio": 0.45,
                "top50_sv_ratio": 0.85,
                "participation_ratio": 156.7,
                "effective_dim": 89.2,
                "mean_pairwise_dist": 8.92,
                "std_pairwise_dist": 2.13,
                "condition_number": 234.5
            },
            ...
        }
    }
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed


def compute_anisotropy_metrics(
    activations: torch.Tensor,
    max_samples_for_svd: int = 1000,
    max_samples_for_pairwise: int = 500,
) -> Dict[str, float]:
    """
    Compute anisotropy and distribution metrics for activations.
    
    Args:
        activations: [n_samples, hidden_dim] or [n_samples, seq_len, hidden_dim] tensor
        max_samples_for_svd: Max samples for SVD (memory efficiency)
        max_samples_for_pairwise: Max samples for pairwise distances
        
    Returns:
        Dictionary of computed metrics
    """
    # Handle 3D activations by mean-pooling over sequence dimension
    if activations.dim() == 3:
        # Shape: [n_samples, seq_len, hidden_dim] -> [n_samples, hidden_dim]
        activations = activations.mean(dim=1)
    
    n_samples, hidden_dim = activations.shape
    device = activations.device
    
    # Move to float32 for numerical stability
    activations = activations.float()
    
    # 1. Norm statistics
    norms = torch.norm(activations, dim=-1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    min_norm = norms.min().item()
    max_norm = norms.max().item()
    
    # 2. Center activations for SVD (important for anisotropy analysis)
    centered = activations - activations.mean(dim=0, keepdim=True)
    
    # 3. Singular value decomposition (subsample if needed)
    if n_samples > max_samples_for_svd:
        indices = torch.randperm(n_samples)[:max_samples_for_svd]
        svd_input = centered[indices]
    else:
        svd_input = centered
    
    try:
        # Use SVD on centered data
        U, S, Vh = torch.linalg.svd(svd_input, full_matrices=False)
        S = S.float()
        
        # Normalize singular values
        S_sum = S.sum()
        if S_sum > 0:
            S_norm = S / S_sum
            
            # Top-k singular value ratios (concentration)
            top10_ratio = S[:10].sum().item() / S_sum.item() if len(S) >= 10 else 1.0
            top50_ratio = S[:50].sum().item() / S_sum.item() if len(S) >= 50 else 1.0
            top100_ratio = S[:100].sum().item() / S_sum.item() if len(S) >= 100 else 1.0
            
            # Participation ratio: 1 / sum(p_i^2) where p_i = s_i / sum(s)
            # High = isotropic (many directions), Low = anisotropic (few directions)
            participation_ratio = 1.0 / (S_norm ** 2).sum().item()
            
            # Effective dimensionality (exponential of entropy)
            # More robust to noise than participation ratio
            S_norm_clipped = S_norm[S_norm > 1e-10]
            entropy = -(S_norm_clipped * torch.log(S_norm_clipped)).sum().item()
            effective_dim = np.exp(entropy)
            
            # Condition number (ratio of largest to smallest singular value)
            # High condition number = ill-conditioned = anisotropic
            if S[-1] > 1e-10:
                condition_number = (S[0] / S[-1]).item()
            else:
                condition_number = float('inf')
        else:
            top10_ratio = top50_ratio = top100_ratio = 1.0
            participation_ratio = 1.0
            effective_dim = 1.0
            condition_number = float('inf')
            
    except Exception as e:
        warnings.warn(f"SVD failed: {e}, using fallback values")
        top10_ratio = top50_ratio = top100_ratio = 1.0
        participation_ratio = 1.0
        effective_dim = 1.0
        condition_number = float('inf')
    
    # 4. Pairwise Euclidean distances (subsample for efficiency)
    if n_samples > max_samples_for_pairwise:
        indices = torch.randperm(n_samples)[:max_samples_for_pairwise]
        dist_input = activations[indices]
    else:
        dist_input = activations
    
    # Compute pairwise distances efficiently using broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    dist_norms = torch.norm(dist_input, dim=-1)
    dist_sq = dist_norms.unsqueeze(0)**2 + dist_norms.unsqueeze(1)**2
    dist_sq = dist_sq - 2 * torch.mm(dist_input, dist_input.t())
    dist_sq = torch.clamp(dist_sq, min=0)  # Numerical stability
    distances = torch.sqrt(dist_sq)
    
    # Get upper triangular (unique pairs)
    n_dist = dist_input.size(0)
    mask = torch.triu(torch.ones(n_dist, n_dist, dtype=torch.bool, device=device), diagonal=1)
    pairwise_dists = distances[mask]
    
    mean_pairwise_dist = pairwise_dists.mean().item()
    std_pairwise_dist = pairwise_dists.std().item()
    min_pairwise_dist = pairwise_dists.min().item()
    max_pairwise_dist = pairwise_dists.max().item()
    
    # 5. Isotropy score (1 - angular concentration around mean direction)
    # Per "All-but-the-Top" paper: measure how concentrated embeddings are
    mean_direction = activations.mean(dim=0)
    mean_direction_norm = torch.norm(mean_direction)
    if mean_direction_norm > 1e-10:
        mean_direction = mean_direction / mean_direction_norm
        # Cosine similarity to mean direction
        cos_sims = torch.mv(activations / norms.unsqueeze(1).clamp(min=1e-10), mean_direction)
        mean_cos_sim = cos_sims.mean().item()
        isotropy_score = 1.0 - abs(mean_cos_sim)  # 1 = isotropic, 0 = all same direction
    else:
        mean_cos_sim = 0.0
        isotropy_score = 1.0
    
    return {
        # Norm statistics
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "min_norm": min_norm,
        "max_norm": max_norm,
        
        # Singular value analysis
        "top10_sv_ratio": top10_ratio,
        "top50_sv_ratio": top50_ratio,
        "top100_sv_ratio": top100_ratio,
        "participation_ratio": participation_ratio,
        "effective_dim": effective_dim,
        "condition_number": condition_number if condition_number != float('inf') else -1,
        
        # Pairwise distance statistics
        "mean_pairwise_dist": mean_pairwise_dist,
        "std_pairwise_dist": std_pairwise_dist,
        "min_pairwise_dist": min_pairwise_dist,
        "max_pairwise_dist": max_pairwise_dist,
        
        # Isotropy
        "mean_cos_sim_to_centroid": mean_cos_sim,
        "isotropy_score": isotropy_score,
    }


def load_cached_activations(path: Path) -> Dict[str, Any]:
    """Load cached activations from .pt file."""
    logger = get_logger()
    logger.info(f"Loading activations from {path}")
    
    data = torch.load(path, map_location="cpu")
    
    # Handle different cache formats
    if isinstance(data, dict):
        if "activations" in data:
            return data
        elif "layer_activations" in data:
            return {"activations": data["layer_activations"], "metadata": data.get("metadata", {})}
    
    raise ValueError(f"Unknown activation cache format: {type(data)}")


def main():
    parser = argparse.ArgumentParser(description="Compute layer-wise activation statistics")
    parser.add_argument(
        "--cached-activations",
        type=str,
        required=True,
        help="Path to cached activations (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to analyze (default: all available)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max samples per layer for analysis (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger()
    set_seed(args.seed)
    
    # Load activations
    cache_path = Path(args.cached_activations)
    if not cache_path.exists():
        raise FileNotFoundError(f"Activations not found: {cache_path}")
    
    cache_data = load_cached_activations(cache_path)
    activations = cache_data["activations"]
    metadata = cache_data.get("metadata", {})
    
    # Parse model/dataset from filename if not in metadata
    model_name = metadata.get("model", cache_path.stem.split("_")[0])
    dataset_name = metadata.get("dataset", cache_path.stem.split("_")[-1].replace(".pt", ""))
    
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")
    logger.info(f"Available layers: {sorted(activations.keys())}")
    
    # Select layers to analyze
    if args.layers:
        layers = [l for l in args.layers if l in activations or str(l) in activations]
    else:
        # Convert all keys to int and sort
        layers = sorted([int(k) for k in activations.keys()])
    
    logger.info(f"Analyzing layers: {layers}")
    
    # Compute statistics for each layer
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "n_samples_analyzed": args.max_samples,
        "seed": args.seed,
        "layers": {},
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    for layer in tqdm(layers, desc="Analyzing layers"):
        layer_key = layer if layer in activations else str(layer)
        layer_acts = activations[layer_key]
        
        # Convert to tensor if needed
        if isinstance(layer_acts, np.ndarray):
            layer_acts = torch.from_numpy(layer_acts)
        
        # Subsample if needed
        n_samples = layer_acts.size(0)
        if n_samples > args.max_samples:
            indices = torch.randperm(n_samples)[:args.max_samples]
            layer_acts = layer_acts[indices]
        
        # Move to GPU for faster computation
        layer_acts = layer_acts.to(device)
        
        logger.info(f"Layer {layer}: shape {layer_acts.shape}")
        
        # Compute metrics
        metrics = compute_anisotropy_metrics(
            layer_acts,
            max_samples_for_svd=min(1000, layer_acts.size(0)),
            max_samples_for_pairwise=min(500, layer_acts.size(0)),
        )
        
        results["layers"][str(layer)] = metrics
        
        # Log key metrics
        logger.info(
            f"  L{layer}: norm={metrics['mean_norm']:.2f}±{metrics['std_norm']:.2f}, "
            f"eff_dim={metrics['effective_dim']:.1f}, "
            f"part_ratio={metrics['participation_ratio']:.1f}, "
            f"isotropy={metrics['isotropy_score']:.3f}"
        )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("LAYER STATISTICS SUMMARY")
    print("=" * 80)
    print(f"{'Layer':>6} {'MeanNorm':>10} {'StdNorm':>10} {'EffDim':>10} {'PartRatio':>12} {'Isotropy':>10} {'Top10SV':>10}")
    print("-" * 80)
    
    for layer in sorted([int(l) for l in results["layers"].keys()]):
        m = results["layers"][str(layer)]
        print(
            f"{layer:>6} "
            f"{m['mean_norm']:>10.2f} "
            f"{m['std_norm']:>10.2f} "
            f"{m['effective_dim']:>10.1f} "
            f"{m['participation_ratio']:>12.1f} "
            f"{m['isotropy_score']:>10.3f} "
            f"{m['top10_sv_ratio']:>10.3f}"
        )
    
    print("=" * 80)
    print("\nInterpretation:")
    print("- Decreasing EffDim/PartRatio at late layers = increasing anisotropy")
    print("- High Top10SV ratio = variance concentrated in few directions")
    print("- Low isotropy = embeddings clustered around mean direction")
    print("- These explain Euclidean probe degradation at late layers")


if __name__ == "__main__":
    main()
