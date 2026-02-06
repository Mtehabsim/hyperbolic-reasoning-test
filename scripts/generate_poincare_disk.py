#!/usr/bin/env python3
"""Generate Poincaré disk visualization comparing hyperbolic and Euclidean probe embeddings."""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.decomposition import PCA

from src.probes.hyperbolic_probe import create_probe


def load_layer_activations(path, layer=27):
    """Load activations for a single layer, mean-pooled across sequence."""
    print(f"Loading activations from {path} (layer {layer})...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    acts = data["activations"][layer]
    if acts.dim() == 3:
        acts = acts.mean(dim=1)
    metadata = data.get("metadata", {})
    print(f"  Shape: {acts.shape}, dtype: {acts.dtype}")
    return acts.float(), metadata


def get_depths(metadata, n_samples):
    """Reconstruct per-sample reasoning depths for PrOntoQA.

    PrOntoQA: 1000 samples, depths 1-5, 200 per depth, generated in order.
    """
    if "depths" in metadata:
        return np.array(metadata["depths"])
    # Default: uniform 200 per depth
    depths = np.repeat(np.arange(1, 6), n_samples // 5)
    if len(depths) < n_samples:
        depths = np.concatenate([depths, np.full(n_samples - len(depths), 5)])
    return depths[:n_samples]


def pairwise_depth_distances(depths):
    """Absolute pairwise depth differences."""
    return torch.tensor(
        np.abs(depths.reshape(-1, 1) - depths.reshape(1, -1)),
        dtype=torch.float32,
    )


def train_probe(probe, hidden_states, targets, device, n_epochs=100, lr=1e-3):
    """Train a probe with early stopping and stress-normalized loss."""
    probe = probe.to(device).float()
    h = hidden_states.to(device).float()
    t = targets.to(device).float()

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    best_loss, patience, best_state = float("inf"), 0, None

    for epoch in range(n_epochs):
        probe.train()
        optimizer.zero_grad()

        z = probe(h)
        pred = probe.pairwise_distances(z)

        diff_sq = (pred - t) ** 2
        loss = diff_sq.sum() / (t ** 2 + 1e-8).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience = 0
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                print(f"  Early stop at epoch {epoch + 1}, loss={best_loss:.4f}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1}: loss={loss.item():.4f}")

    if best_state:
        probe.load_state_dict(best_state)
    return probe


def extract_embeddings(probe, hidden_states, device):
    """Get probe output embeddings."""
    probe.eval()
    with torch.no_grad():
        z = probe(hidden_states.to(device).float())
    return z.cpu().numpy()


def plot_poincare_disk(ax, pts, depths, title):
    """Plot 2D points on the Poincaré disk with depth coloring."""
    # Unit circle boundary
    circle = Circle((0, 0), 1.0, fill=False, color="#333333", linewidth=1.2,
                     linestyle="-", zorder=1)
    ax.add_patch(circle)

    # Rescale into disk if needed
    norms = np.linalg.norm(pts, axis=1)
    if norms.max() > 0.98:
        pts = pts * (0.95 / norms.max())

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=1, vmax=5)
    scatter = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=depths, cmap=cmap, norm=norm,
        s=15, alpha=0.75, edgecolors="none", zorder=2,
    )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("PCA 1", fontsize=8)
    ax.set_ylabel("PCA 2", fontsize=8)
    ax.tick_params(labelsize=7)
    # Light grid
    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)
    return scatter


def plot_euclidean(ax, pts, depths, title):
    """Plot 2D Euclidean scatter with depth coloring."""
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=1, vmax=5)
    scatter = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=depths, cmap=cmap, norm=norm,
        s=15, alpha=0.75, edgecolors="none",
    )
    # Use equal lim range (centered) so shape is comparable to Poincare disk
    max_abs = max(np.abs(pts).max(axis=0))
    margin = max_abs * 1.1
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("PCA 1", fontsize=8)
    ax.set_ylabel("PCA 2", fontsize=8)
    ax.tick_params(labelsize=7)
    # Light grid
    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)
    return scatter


def main():
    parser = argparse.ArgumentParser(description="Generate Poincare disk visualization")
    parser.add_argument("--activations", type=str, default="outputs/activations/deepseek_prontoqa.pt",
                        help="Path to cached activations file")
    parser.add_argument("--layer", type=int, default=27, help="Layer to visualize")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load data ---
    acts, meta = load_layer_activations(args.activations, layer=args.layer)
    n, d = acts.shape
    depths = get_depths(meta, n)
    targets = pairwise_depth_distances(depths)
    print(f"Samples: {n}, input dim: {d}, depth range: {depths.min()}-{depths.max()}")

    # --- Train hyperbolic probe ---
    print("\n[1/2] Training hyperbolic probe (d=5, c=0.5)...")
    hyp_probe = create_probe("hyperbolic", input_dim=d, output_dim=5, curvature=0.5)
    hyp_probe = train_probe(hyp_probe, acts, targets, device)
    hyp_emb = extract_embeddings(hyp_probe, acts, device)

    # --- Train Euclidean probe ---
    print("\n[2/2] Training Euclidean probe (d=5)...")
    euc_probe = create_probe("euclidean", input_dim=d, output_dim=5)
    euc_probe = train_probe(euc_probe, acts, targets, device)
    euc_emb = extract_embeddings(euc_probe, acts, device)

    # --- Project to 2D ---
    print("\nProjecting to 2D (PCA)...")
    hyp_2d = PCA(n_components=2).fit_transform(hyp_emb)
    euc_2d = PCA(n_components=2).fit_transform(euc_emb)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(wspace=0.35, bottom=0.18)

    s1 = plot_poincare_disk(ax1, hyp_2d, depths, "Hyperbolic Probe (L27)")
    s2 = plot_euclidean(ax2, euc_2d, depths, "Euclidean Probe (L27)")

    # Horizontal colorbar below both plots, properly separated
    cbar_ax = fig.add_axes([0.25, 0.06, 0.50, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(s1, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Reasoning Depth", fontsize=10)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.ax.tick_params(labelsize=8)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_base = os.path.join(out_dir, "..", "poincare_disk_visualization")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_base}.pdf/.png")


if __name__ == "__main__":
    main()
