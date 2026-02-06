"""
Model utilities for activation extraction and token selection.
"""

from .loader import load_model, load_model_config, unload_model, get_model_info
from .hooks import (
    extract_activations,
    extract_activations_with_generation,
    extract_activations_with_positions,
    extract_all_layers,
    ActivationCache,
)
from .token_selector import TokenSelector, find_thinking_positions, find_attention_weighted_positions
from .layer_selection import (
    compute_layer_drift,
    select_layers_by_max_drift,
    aggregate_layer_activations,
    select_layers_by_threshold,
)

__all__ = [
    "load_model",
    "load_model_config",
    "unload_model",
    "get_model_info",
    "extract_activations",
    "extract_activations_with_generation",
    "extract_activations_with_positions",
    "extract_all_layers",
    "ActivationCache",
    "TokenSelector",
    "find_thinking_positions",
    "find_attention_weighted_positions",
    "compute_layer_drift",
    "select_layers_by_max_drift",
    "aggregate_layer_activations",
    "select_layers_by_threshold",
]
