"""
Probe training utilities.

Provides training loop and evaluation functions for all probe types.
Features: early stopping, LR scheduling, checkpointing.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.geometry.metrics import compute_all_metrics
from src.utils.logging import get_logger
from src.utils.wandb_logger import WandBLogger


def train_probe(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    target_distances: torch.Tensor,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 10,
    device: Optional[str] = None,
    verbose: bool = True,
    # Advanced options
    val_split: float = 0.0,
    use_lr_scheduler: bool = True,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    grad_clip_norm: float = 1.0,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_name: Optional[str] = None,
    wandb_logger: Optional[WandBLogger] = None,
    log_samples: bool = False,
    # Target distance normalization for hyperbolic stability
    normalize_targets: bool = False,
    target_max_distance: float = 4.0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a probe on hidden states to preserve distances.
    
    Args:
        probe: Probe module (Euclidean or Hyperbolic)
        hidden_states: [n_samples, d_model]
        target_distances: [n_samples, n_samples]
        n_epochs: Training epochs
        learning_rate: Learning rate
        weight_decay: L2 regularization
        early_stopping_patience: Stop if no improvement
        device: Device to train on
        verbose: Show progress bar
        val_split: Fraction for validation (0.0 = no validation)
        use_lr_scheduler: Use ReduceLROnPlateau scheduler
        lr_patience: Patience for LR reduction
        lr_factor: Factor to reduce LR by
        min_lr: Minimum learning rate
        grad_clip_norm: Max gradient norm for clipping
        checkpoint_dir: Directory to save checkpoints
        checkpoint_name: Name for checkpoint file
        wandb_logger: Optional W&B logger for experiment tracking
        log_samples: Whether to log per-sample predictions (at end of training)
        normalize_targets: If True, normalize target distances to [0, target_max_distance]
                          CRITICAL for hyperbolic probes when target range > 4-5 units
        target_max_distance: Maximum target distance after normalization (default 4.0
                            is stable for Poincaré ball with curvature=1.0)
        
    Returns:
        (trained_probe, training_history)
    """
    logger = get_logger()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    probe = probe.to(device)
    hidden_states = hidden_states.to(device).float()  # Ensure float32 for probe compatibility
    target_distances = target_distances.to(device).float()
    
    # Target distance normalization for hyperbolic stability
    if normalize_targets:
        max_target = target_distances.max()
        if max_target > 0:
            target_distances = target_distances * (target_max_distance / max_target)
            logger.info(f"Normalized target distances from [0, {max_target:.2f}] to [0, {target_max_distance:.2f}]")
    
    # Validation split (if requested)
    n_samples = hidden_states.shape[0]
    if val_split > 0 and n_samples > 10:
        n_val = max(int(n_samples * val_split), 2)
        perm = torch.randperm(n_samples)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        
        train_hidden = hidden_states[train_idx]
        train_targets = target_distances[train_idx][:, train_idx]
        val_hidden = hidden_states[val_idx]
        val_targets = target_distances[val_idx][:, val_idx]
        logger.debug(f"Split: {len(train_idx)} train, {len(val_idx)} val")
    else:
        train_hidden = hidden_states
        train_targets = target_distances
        val_hidden = val_targets = None
    
    optimizer = optim.Adam(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # LR scheduler for better convergence
    scheduler = None
    if use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor,
            patience=lr_patience, min_lr=min_lr
        )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "best_loss": float('inf'),
        "best_epoch": 0,
    }
    
    best_state = None
    patience_counter = 0
    
    iterator = range(n_epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"Training {probe.name}")
    
    for epoch in iterator:
        probe.train()
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = probe(train_hidden)
        pred_distances = probe.pairwise_distances(embeddings)
        
        # Loss
        loss = probe.distortion_loss(pred_distances, train_targets)
        
        # Backward
        loss.backward()
        
        # Gradient clipping for stability
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=grad_clip_norm)
        
        optimizer.step()
        
        # Track metrics
        current_loss = loss.item()
        history["train_loss"].append(current_loss)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        
        # W&B logging
        if wandb_logger:
            wandb_logger.log({
                "train_loss": current_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)
        
        # Validation loss (if applicable)
        val_loss = None
        if val_hidden is not None:
            probe.eval()
            with torch.no_grad():
                val_emb = probe(val_hidden)
                val_pred = probe.pairwise_distances(val_emb)
                val_loss = probe.distortion_loss(val_pred, val_targets).item()
            history["val_loss"].append(val_loss)
            if wandb_logger:
                wandb_logger.log({"val_loss": val_loss}, step=epoch)
            probe.train()
        
        # LR scheduler step
        metric_for_scheduler = val_loss if val_loss is not None else current_loss
        if scheduler is not None:
            scheduler.step(metric_for_scheduler)
        
        if verbose:
            postfix = {"loss": f"{current_loss:.4f}"}
            if val_loss is not None:
                postfix["val"] = f"{val_loss:.4f}"
            iterator.set_postfix(postfix)
        
        # Early stopping on validation or train loss
        stopping_metric = val_loss if val_loss is not None else current_loss
        if stopping_metric < history["best_loss"] - 1e-6:
            history["best_loss"] = stopping_metric
            history["best_epoch"] = epoch
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            logger.debug(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best weights
    if best_state is not None:
        probe.load_state_dict(best_state)
    
    history["epochs_trained"] = epoch + 1
    history["final_lr"] = optimizer.param_groups[0]["lr"]
    
    # Save checkpoint if requested
    if checkpoint_dir is not None and best_state is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        name = checkpoint_name or f"{probe.name}_best.pt"
        checkpoint_path = checkpoint_dir / name
        torch.save({
            "probe_state_dict": best_state,
            "history": history,
            "config": {
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    return probe, history


def evaluate_probe(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    target_distances: torch.Tensor,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained probe.
    
    Returns metrics including Spearman rho, distortion, MAP@K.
    """
    if device is None:
        device = next(probe.parameters()).device
    
    probe.eval()
    hidden_states = hidden_states.to(device).float()  # Ensure float32
    target_distances = target_distances.to(device).float()
    
    with torch.no_grad():
        embeddings = probe(hidden_states)
        pred_distances = probe.pairwise_distances(embeddings)
    
    metrics = compute_all_metrics(
        pred_distances.cpu(),
        target_distances.cpu(),
        k=5,
    )
    
    return metrics


def train_and_evaluate(
    probe: nn.Module,
    train_hidden: torch.Tensor,
    train_distances: torch.Tensor,
    test_hidden: Optional[torch.Tensor] = None,
    test_distances: Optional[torch.Tensor] = None,
    **train_kwargs,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train and evaluate a probe.
    
    Args:
        probe: Probe module
        train_hidden: Training hidden states
        train_distances: Training target distances
        test_hidden: Optional test hidden states
        test_distances: Optional test target distances
        **train_kwargs: Additional arguments for train_probe
        
    Returns:
        (trained_probe, results_dict)
    """
    # Train
    probe, history = train_probe(
        probe=probe,
        hidden_states=train_hidden,
        target_distances=train_distances,
        **train_kwargs,
    )
    
    # Evaluate on training data
    train_metrics = evaluate_probe(probe, train_hidden, train_distances)
    
    results = {
        "training": history,
        "train_metrics": train_metrics,
    }
    
    # Evaluate on test data if provided
    if test_hidden is not None and test_distances is not None:
        test_metrics = evaluate_probe(probe, test_hidden, test_distances)
        results["test_metrics"] = test_metrics
    
    return probe, results
