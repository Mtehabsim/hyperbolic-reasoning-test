#!/usr/bin/env python3
"""
Single-output regression probing for Binary Tree dataset.

For Binary Tree, each sample asks "what's the distance between node A and node B?"
We probe: can the model's activation predict the actual distance?
This is single-output regression, not pairwise distance matching.
"""

import argparse
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class LinearRegressor(nn.Module):
    """Simple linear regression probe."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_regressor(activations: torch.Tensor, targets: torch.Tensor, 
                    epochs: int = 100, lr: float = 0.001):
    """Train a simple regression probe."""
    # Split 80/20
    n = len(activations)
    n_train = int(0.8 * n)
    perm = torch.randperm(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    
    X_train, y_train = activations[train_idx], targets[train_idx]
    X_test, y_test = activations[test_idx], targets[test_idx]
    
    # Normalize activations
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Model
    model = LinearRegressor(activations.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_norm)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_norm).numpy()
    
    rho, p_value = spearmanr(test_preds, y_test.numpy())
    mse = np.mean((test_preds - y_test.numpy()) ** 2)
    
    return {
        'spearman_rho': float(rho),
        'p_value': float(p_value),
        'mse': float(mse),
        'train_loss': float(loss.item())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[8, 12, 16, 19, 21, 23, 25, 27])
    args = parser.parse_args()
    
    # Load activations
    logger.info(f"Loading activations from {args.activations}")
    act_data = torch.load(args.activations)
    
    # Load ground-truth distances
    logger.info(f"Loading data from {args.data}")
    with open(args.data, 'r') as f:
        data = json.load(f)
    
    # Extract distances as targets
    distances = torch.tensor([s['distance'] for s in data], dtype=torch.float32)
    n_samples = len(distances)
    logger.info(f"Loaded {n_samples} samples with distances range [{distances.min():.0f}, {distances.max():.0f}]")
    
    # Run regression for each layer
    results = {'layers': {}}
    
    for layer in args.layers:
        if layer not in act_data['activations']:
            logger.warning(f"Layer {layer} not in activations, skipping")
            continue
        
        acts = act_data['activations'][layer]  # [n_samples, seq_len, hidden_dim]
        pooled = acts.mean(dim=1).float()  # [n_samples, hidden_dim]
        
        # Align sizes
        n_acts = pooled.shape[0]
        n = min(n_acts, n_samples)
        pooled = pooled[:n]
        dist_targets = distances[:n]
        
        logger.info(f"Layer {layer}: {n} samples, activation shape {pooled.shape}")
        
        # Train regression probe
        result = train_regressor(pooled, dist_targets)
        logger.info(f"Layer {layer}: ρ={result['spearman_rho']:.4f}, MSE={result['mse']:.4f}")
        
        results['layers'][layer] = result
    
    # Summary
    rhos = [results['layers'][l]['spearman_rho'] for l in args.layers if l in results['layers']]
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Mean Spearman ρ: {np.mean(rhos):.4f}")
    logger.info(f"Max Spearman ρ: {np.max(rhos):.4f} at layer {args.layers[np.argmax(rhos)]}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
