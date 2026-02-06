"""
Token selection strategies for reasoning analysis.

Implements multiple strategies for selecting which tokens to use
when pooling hidden states for probe training.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.config import get_config_dir
from src.utils.logging import get_logger


# Default thinking tokens (from config)
# Based on "Demystifying Reasoning Dynamics with Mutual Information" (arXiv:2506.02867)
# These tokens correspond to MI peaks - information bottleneck points during reasoning
DEFAULT_THINKING_TOKENS = [
    # Reflective/pause (high MI peaks per paper)
    "wait", "hmm", "hm", "Hmm", "Wait",
    # Transition markers
    "okay", "ok", "well", "so", "therefore", "Thus", "Therefore",
    "thus", "hence", "now", "Hence",
    # Self-correction (critical for reasoning)
    "actually", "Actually", "but", "But", "however", "although",
    "but wait", "wait,",
    # Planning/structure
    "let me", "Let me", "first", "First", "next", "then", "finally",
    "step", "Step",
    # Conclusion/inference
    "means", "implies", "shows", "proves", "conclude", "Conclude",
    # Additional from paper (consider, note, important)
    "consider", "note", "Note", "importantly", "crucially",
    "specifically", "essentially", "alternatively",
    "interestingly", "Interestingly",
    # Verification tokens
    "check", "verify", "confirm", "double-check",
]



def load_thinking_tokens(config: Optional[DictConfig] = None) -> List[str]:
    """Load thinking tokens from config or use defaults."""
    if config is None:
        try:
            config = OmegaConf.load(get_config_dir() / "config.yaml")
        except Exception:
            return DEFAULT_THINKING_TOKENS.copy()
    
    return list(config.get("token_selection", {}).get("thinking_tokens", DEFAULT_THINKING_TOKENS))


def find_thinking_positions(
    tokens: List[str],
    thinking_tokens: Optional[List[str]] = None,
    case_insensitive: bool = True,
) -> List[int]:
    """
    Find positions of thinking tokens in a token list.
    
    Args:
        tokens: List of token strings
        thinking_tokens: List of thinking token patterns
        case_insensitive: Match case-insensitively
        
    Returns:
        List of token positions (0-indexed)
    """
    if thinking_tokens is None:
        thinking_tokens = load_thinking_tokens()
    
    positions = []
    
    for i, token in enumerate(tokens):
        # Clean token (remove special chars, whitespace)
        clean_token = token.strip().lower() if case_insensitive else token.strip()
        clean_token = clean_token.lstrip("Ġ").lstrip("▁")  # Remove tokenizer prefixes
        
        for think_token in thinking_tokens:
            pattern = think_token.lower() if case_insensitive else think_token
            
            # Check for match (substring or full)
            if pattern in clean_token or clean_token.startswith(pattern):
                positions.append(i)
                break
    
    return positions


def find_attention_weighted_positions(
    attention_weights: torch.Tensor,
    top_k: int = 5,
    exclude_first: int = 1,
    exclude_last: int = 1,
    head_selection: str = "mean",
    head_threshold: float = 0.9,
) -> List[int]:
    """
    Find positions with highest attention weights.
    
    Args:
        attention_weights: Attention weights [n_heads, seq_len, seq_len] or [seq_len, seq_len]
        top_k: Number of top positions to return
        exclude_first: Exclude first N positions (usually BOS/system)
        exclude_last: Exclude last N positions
        head_selection: How to combine heads:
            - "mean": Average across all heads
            - "max": Take max-attention head only
            - "threshold": Take heads within threshold of max (reduces damping)
        head_threshold: For "threshold" mode, fraction of max head attention to include
        
    Returns:
        List of position indices
    """
    # Handle head aggregation
    if attention_weights.dim() == 3:
        n_heads = attention_weights.shape[0]
        
        if head_selection == "max":
            # Take the head with highest total attention
            head_totals = attention_weights.sum(dim=(1, 2))  # [n_heads]
            max_head_idx = head_totals.argmax()
            attention_weights = attention_weights[max_head_idx]  # [seq_len, seq_len]
            
        elif head_selection == "threshold":
            # Take heads within threshold of max attention
            head_totals = attention_weights.sum(dim=(1, 2))  # [n_heads]
            max_total = head_totals.max()
            active_heads = head_totals >= max_total * head_threshold
            if active_heads.sum() > 0:
                attention_weights = attention_weights[active_heads].mean(dim=0)
            else:
                attention_weights = attention_weights[head_totals.argmax()]
                
        else:  # "mean"
            attention_weights = attention_weights.mean(dim=0)  # [seq_len, seq_len]
    
    # Sum attention received by each token (columns)
    token_importance = attention_weights.sum(dim=0)  # [seq_len]
    
    # Mask excluded positions
    seq_len = len(token_importance)
    mask = torch.ones(seq_len, dtype=torch.bool)
    if exclude_first > 0:
        mask[:exclude_first] = False
    if exclude_last > 0:
        mask[-exclude_last:] = False
    
    # Apply mask
    token_importance[~mask] = float("-inf")
    
    # Get top-k positions
    _, top_indices = torch.topk(token_importance, min(top_k, mask.sum().item()))
    
    return sorted(top_indices.tolist())



class TokenSelector:
    """
    Token selection with fallback chain.
    
    Supports multiple selection methods:
    - thinking_tokens: Select tokens matching thinking patterns
    - attention_weighted: Select high-attention tokens
    - all_pool: Use all tokens (mean pooling)
    - last_token: Use only the last token
    - random: Random selection
    """
    
    def __init__(
        self,
        method: str = "thinking_tokens",
        fallback: str = "attention_weighted",
        min_tokens: int = 2,
        thinking_tokens: Optional[List[str]] = None,
        config: Optional[DictConfig] = None,
    ):
        """
        Initialize token selector.
        
        Args:
            method: Primary selection method
            fallback: Fallback method if primary fails
            min_tokens: Minimum tokens required before fallback
            thinking_tokens: Override thinking tokens list
            config: Full configuration
        """
        self.method = method
        self.fallback = fallback
        self.min_tokens = min_tokens
        self.thinking_tokens = thinking_tokens or load_thinking_tokens(config)
        self.logger = get_logger()
        
        # Valid methods
        self.valid_methods = {
            "thinking_tokens",
            "attention_weighted",
            "all_pool",
            "last_token",
            "random",
        }
        
        if method not in self.valid_methods:
            raise ValueError(f"Unknown method: {method}. Valid: {self.valid_methods}")
    
    def select(
        self,
        tokens: List[str],
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        seq_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[int], str]:
        """
        Select tokens and return pooled hidden states.
        
        Args:
            tokens: List of token strings
            hidden_states: Hidden states [seq_len, d_model]
            attention_weights: Optional attention weights for attention-based selection
            seq_length: Actual sequence length (excluding padding)
            
        Returns:
            (pooled_hidden, selected_positions, method_used)
        """
        if seq_length is None:
            seq_length = len(tokens)
        
        # Trim to actual length
        hidden_states = hidden_states[:seq_length]
        tokens = tokens[:seq_length]
        
        # Try primary method
        positions, method_used = self._select_positions(
            tokens=tokens,
            attention_weights=attention_weights,
            method=self.method,
        )
        
        # Fallback if insufficient
        if len(positions) < self.min_tokens and self.fallback:
            self.logger.debug(
                f"Method {self.method} returned {len(positions)} positions, "
                f"falling back to {self.fallback}"
            )
            positions, method_used = self._select_positions(
                tokens=tokens,
                attention_weights=attention_weights,
                method=self.fallback,
            )
        
        # Ultimate fallback: last token
        if len(positions) == 0:
            positions = [seq_length - 1]
            method_used = "last_token"
        
        # Pool hidden states at selected positions
        selected_hidden = hidden_states[positions]  # [n_positions, d_model]
        pooled = selected_hidden.mean(dim=0)  # [d_model]
        
        return pooled, positions, method_used
    
    def _select_positions(
        self,
        tokens: List[str],
        attention_weights: Optional[torch.Tensor],
        method: str,
    ) -> Tuple[List[int], str]:
        """Apply a selection method."""
        if method == "thinking_tokens":
            positions = find_thinking_positions(tokens, self.thinking_tokens)
            return positions, method
        
        elif method == "attention_weighted":
            if attention_weights is None:
                return [], method
            positions = find_attention_weighted_positions(attention_weights)
            return positions, method
        
        elif method == "all_pool":
            return list(range(len(tokens))), method
        
        elif method == "last_token":
            return [len(tokens) - 1] if tokens else [], method
        
        elif method == "random":
            import random
            n = min(5, len(tokens))
            positions = random.sample(range(len(tokens)), n)
            return sorted(positions), method
        
        return [], method
    
    @classmethod
    def from_config(cls, config: DictConfig) -> "TokenSelector":
        """Create from configuration."""
        ts_config = config.get("token_selection", {})
        return cls(
            method=ts_config.get("method", "thinking_tokens"),
            fallback=ts_config.get("fallback", "attention_weighted"),
            min_tokens=ts_config.get("min_tokens", 2),
            config=config,
        )
