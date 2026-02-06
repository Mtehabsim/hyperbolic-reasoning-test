"""
Reproducibility utilities for deterministic experiments.

Provides seed management for PyTorch, NumPy, and Python random modules,
plus CUDA determinism settings.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled: bool = True) -> None:
    """
    Enable or disable deterministic operations in PyTorch.
    
    Note: Deterministic mode may reduce performance.
    
    Args:
        enabled: Whether to enable deterministic mode
    """
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """
    Complete reproducibility setup.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode
    """
    set_seed(seed)
    set_deterministic(deterministic)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Explicit device string, or None for auto-detection
        
    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
