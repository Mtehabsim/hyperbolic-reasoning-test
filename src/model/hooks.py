"""
Activation extraction hooks for TransformerLens models.

Provides utilities for extracting hidden states at specific layers
during model forward passes.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from src.utils.logging import get_logger


def extract_activations(
    model,
    prompts: List[str],
    layers: Union[int, List[int]],
    tokenizer=None,
    batch_size: int = 1,
    max_length: int = 512,
    device: Optional[str] = None,
    return_attention: bool = False,
    show_progress: bool = True,
    cached_activations_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Extract hidden state activations from specified layers.
    
    Automatically dispatches to TransformerLens or HuggingFace extraction
    based on model type.
    
    Args:
        model: HookedTransformer or HuggingFace model
        prompts: List of input prompts
        layers: Layer index or list of layer indices to extract
        tokenizer: Tokenizer (uses model.tokenizer if None)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device for computation
        return_attention: If True, also return attention patterns
        show_progress: Show progress bar
        
    Returns:
        Dictionary with:
            - 'activations': Dict[layer_idx, Tensor[n_samples, seq_len, d_model]]
            - 'tokens': List of token lists
            - 'attention': Dict[layer_idx, Tensor] (if return_attention=True)
    """
    logger = get_logger()
    
    # CACHE LOADING: If cached path exists, load and return early
    if cached_activations_path is not None:
        cache_path = Path(cached_activations_path)
        if cache_path.exists():
            logger.info(f"Loading cached activations from {cache_path}")
            cached_data = torch.load(cache_path, map_location='cpu')
            
            # Convert layer indices to strings if needed (for compatibility)
            if 'activations' in cached_data:
                # Already in expected format
                return cached_data
            elif 'hidden_states' in cached_data:
                # Old format: convert to expected format
                hidden_states = cached_data['hidden_states']
                n_layers = hidden_states.shape[1] if hidden_states.ndim > 2 else 1
                
                # Reshape to layer-indexed dict
                activations = {}
                for i, layer_idx in enumerate(layers if isinstance(layers, list) else [layers]):
                    if i < hidden_states.shape[1]:
                        activations[layer_idx] = hidden_states[:, i, :, :]
                
                return {
                    'activations': activations,
                    'tokens': cached_data.get('tokens', []),
                    'attention': cached_data.get('attention', {}),
                    'metadata': cached_data.get('metadata', {}),
                }
            else:
                logger.warning(f"Unknown cache format, attempting direct access")
                return cached_data
    
    # Check if this is a TransformerLens model
    is_transformer_lens = getattr(model, '_is_transformer_lens', hasattr(model, 'run_with_cache'))
    
    if is_transformer_lens:
        return _extract_activations_transformer_lens(
            model=model,
            prompts=prompts,
            layers=layers,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            return_attention=return_attention,
            show_progress=show_progress,
        )
    else:
        return _extract_activations_huggingface(
            model=model,
            prompts=prompts,
            layers=layers,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            return_attention=return_attention,
            show_progress=show_progress,
        )


def extract_activations_with_generation(
    model,
    prompts: List[str],
    layers: Union[int, List[int]],
    tokenizer=None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    max_length: int = 1024,
    device: Optional[str] = None,
    return_attention: bool = False,
    show_progress: bool = True,
    stop_tokens: Optional[List[str]] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate reasoning traces and extract activations from full sequences.
    
    This is essential for reasoning models like DeepSeek-R1 where thinking
    tokens only appear in the generated response, not the input prompt.
    
    Args:
        model: HookedTransformer or HuggingFace model
        prompts: List of input prompts
        layers: Layer index or list of layer indices to extract
        tokenizer: Tokenizer (uses model.tokenizer if None)
        batch_size: Batch size for processing (set to 1 for generation)
        max_new_tokens: Maximum tokens to generate per prompt
        max_length: Maximum total sequence length (input + generated)
        device: Device for computation
        return_attention: If True, also return attention patterns
        show_progress: Show progress bar
        stop_tokens: List of tokens to stop generation at (e.g., ["</think>"])
        temperature: Generation temperature (0.0 for greedy)
        
    Returns:
        Dictionary with:
            - 'activations': Dict[layer_idx, Tensor[n_samples, seq_len, d_model]]
            - 'tokens': List of token lists (including generated tokens)
            - 'attention': Dict[layer_idx, Tensor] (if return_attention=True)
            - 'generated_text': List of generated strings
            - 'thinking_positions': List of thinking token positions per sample
    """
    logger = get_logger()
    
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided or available on model")
    
    if isinstance(layers, int):
        layers = [layers]
    
    if device is None:
        device = next(model.parameters()).device
    
    # Default stop tokens for DeepSeek-R1 style reasoning
    if stop_tokens is None:
        stop_tokens = ["</think>", "<|endofthink|>", "<|end|>"]
    
    # Check if model supports generation
    is_transformer_lens = getattr(model, '_is_transformer_lens', hasattr(model, 'run_with_cache'))
    
    logger.info(f"Generating reasoning traces for {len(prompts)} prompts, then extracting from layers: {layers}")
    
    # Results storage
    all_activations = {layer: [] for layer in layers}
    all_attention = {layer: [] for layer in layers} if return_attention else None
    all_tokens = []
    all_generated_text = []
    all_thinking_positions = []
    
    # Define thinking token patterns for position tracking
    thinking_patterns = [
        "wait", "hmm", "hm", "Hmm", "Wait",
        "okay", "ok", "well", "so", "therefore", "Thus", "Therefore",
        "actually", "Actually", "but", "But", "however",
        "let me", "Let me", "first", "First", "step", "Step",
        "means", "implies", "shows", "proves", "conclude",
    ]
    
    # Process one at a time for generation (batch_size=1 for generation stability)
    iterator = range(len(prompts))
    if show_progress:
        iterator = tqdm(iterator, desc="Generating + extracting")
    
    for i in iterator:
        prompt = prompts[i]
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length // 2,  # Leave room for generation
        ).to(device)
        
        input_len = inputs.input_ids.shape[1]
        
        # Generate reasoning trace
        with torch.no_grad():
            if is_transformer_lens:
                # TransformerLens generation - different API than HuggingFace
                # TransformerLens.generate() signature: generate(input, max_new_tokens, stop_at_eos, ...)
                try:
                    generated_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        stop_at_eos=True,
                        temperature=temperature if temperature > 0 else 1.0,
                        verbose=False,
                    )
                except TypeError:
                    # Fallback for older TransformerLens versions
                    generated_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        stop_at_eos=True,
                    )
            else:
                # HuggingFace generation
                generated_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        
        # Decode generated text
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        generated_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
        all_generated_text.append(generated_text)
        
        # Get tokens for the full sequence
        tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
        all_tokens.append(tokens)
        
        # Find thinking token positions
        thinking_positions = []
        for pos, token in enumerate(tokens):
            clean_token = token.strip().lower().lstrip("Ġ").lstrip("▁")
            for pattern in thinking_patterns:
                if pattern.lower() in clean_token or clean_token.startswith(pattern.lower()):
                    thinking_positions.append(pos)
                    break
        all_thinking_positions.append(thinking_positions)
        
        # Now extract activations from the full generated sequence
        if is_transformer_lens:
            # Build hook names
            hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
            if return_attention:
                attn_hook_names = [f"blocks.{layer}.attn.hook_attn" for layer in layers]
                hook_names.extend(attn_hook_names)
            
            # Clone generated_ids to avoid inference mode issue
            # TransformerLens generate returns inference tensors that can't be used with run_with_cache
            generated_ids_clone = generated_ids.clone().detach()
            
            # Run with cache on full sequence
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    generated_ids_clone,
                    names_filter=hook_names,
                    return_type="logits",
                )
            
            # Extract activations
            for layer in layers:
                hook_name = f"blocks.{layer}.hook_resid_post"
                acts = cache[hook_name]  # [1, seq_len, d_model]
                all_activations[layer].append(acts.cpu())
                
                if return_attention:
                    attn_hook = f"blocks.{layer}.attn.hook_attn"
                    if attn_hook in cache:
                        all_attention[layer].append(cache[attn_hook].cpu())
            
            del cache
        else:
            # HuggingFace with hooks
            activations_cache = {}
            
            def make_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    activations_cache[layer_idx] = hidden.detach().cpu()
                return hook
            
            # Register hooks
            handles = []
            for layer in layers:
                handle = model.model.layers[layer].register_forward_hook(make_hook(layer))
                handles.append(handle)
            
            # Forward pass
            _ = model(generated_ids)
            
            # Collect activations
            for layer in layers:
                all_activations[layer].append(activations_cache[layer])
            
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Pad activations to same sequence length
    def pad_and_cat(tensors: list, dim: int = 0) -> torch.Tensor:
        if not tensors:
            return torch.tensor([])
        
        max_seq_len = max(t.shape[1] for t in tensors)
        d_model = tensors[0].shape[-1]
        
        padded = []
        for t in tensors:
            if t.shape[1] < max_seq_len:
                pad_size = max_seq_len - t.shape[1]
                padding = torch.zeros(t.shape[0], pad_size, d_model, dtype=t.dtype)
                t = torch.cat([t, padding], dim=1)
            padded.append(t)
        
        return torch.cat(padded, dim=dim)
    
    result = {
        "activations": {
            layer: pad_and_cat(all_activations[layer], dim=0)
            for layer in layers
        },
        "tokens": all_tokens,
        "layers": layers,
        "n_samples": len(prompts),
        "generated_text": all_generated_text,
        "thinking_positions": all_thinking_positions,
    }
    
    if return_attention:
        result["attention"] = {
            layer: pad_and_cat(all_attention[layer], dim=0) if all_attention[layer] else None
            for layer in layers
        }
    
    logger.info(f"Generated {len(prompts)} traces, avg thinking tokens: "
                f"{sum(len(p) for p in all_thinking_positions) / len(prompts):.1f}")
    
    return result


def _extract_activations_transformer_lens(
    model,
    prompts: List[str],
    layers: Union[int, List[int]],
    tokenizer=None,
    batch_size: int = 1,
    max_length: int = 512,
    device: Optional[str] = None,
    return_attention: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Extract activations using TransformerLens run_with_cache."""
    logger = get_logger()
    
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided or available on model")
    
    if isinstance(layers, int):
        layers = [layers]
    
    if device is None:
        device = next(model.parameters()).device
    
    # Build hook names for TransformerLens
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    if return_attention:
        attn_hook_names = [f"blocks.{layer}.attn.hook_attn" for layer in layers]
        hook_names.extend(attn_hook_names)
    
    logger.info(f"Extracting activations from {len(prompts)} prompts, layers: {layers}")
    
    # Results storage
    all_activations = {layer: [] for layer in layers}
    all_attention = {layer: [] for layer in layers} if return_attention else None
    all_tokens = []
    
    # Process in batches
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting activations")
    
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        # Store tokens
        for i in range(len(batch_prompts)):
            seq_len = inputs.attention_mask[i].sum().item()
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[i][:seq_len])
            all_tokens.append(tokens)
        
        # Run forward pass with cache
        with torch.no_grad():
            _, cache = model.run_with_cache(
                inputs.input_ids,
                names_filter=hook_names,
                return_type="logits",
            )
        
        # Extract activations
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            acts = cache[hook_name]  # [batch, seq_len, d_model]
            
            # Mask padding
            mask = inputs.attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            acts = acts * mask
            
            all_activations[layer].append(acts.cpu())
            
            if return_attention:
                attn_hook = f"blocks.{layer}.attn.hook_attn"
                if attn_hook in cache:
                    all_attention[layer].append(cache[attn_hook].cpu())
        
        # Clear cache
        del cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Pad activations to same sequence length before concatenating
    # Find max sequence length across all batches for each layer
    def pad_and_cat(tensors: list, dim: int = 0) -> torch.Tensor:
        """Pad tensors to same seq_len (dim=1) then concatenate along dim=0."""
        if not tensors:
            return torch.tensor([])
        
        # Find max seq_len
        max_seq_len = max(t.shape[1] for t in tensors)
        d_model = tensors[0].shape[-1]
        
        padded = []
        for t in tensors:
            if t.shape[1] < max_seq_len:
                # Pad along seq_len dimension
                pad_size = max_seq_len - t.shape[1]
                padding = torch.zeros(t.shape[0], pad_size, d_model, dtype=t.dtype)
                t = torch.cat([t, padding], dim=1)
            padded.append(t)
        
        return torch.cat(padded, dim=dim)
    
    # Concatenate results with padding
    result = {
        "activations": {
            layer: pad_and_cat(all_activations[layer], dim=0)
            for layer in layers
        },
        "tokens": all_tokens,
        "layers": layers,
        "n_samples": len(prompts),
    }
    
    if return_attention:
        result["attention"] = {
            layer: pad_and_cat(all_attention[layer], dim=0) if all_attention[layer] else None
            for layer in layers
        }
    
    return result


def _extract_activations_huggingface(
    model,
    prompts: List[str],
    layers: Union[int, List[int]],
    tokenizer=None,
    batch_size: int = 1,
    max_length: int = 512,
    device: Optional[str] = None,
    return_attention: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Extract activations using HuggingFace register_forward_hook.
    
    This is a fallback for models not supported by TransformerLens.
    """
    logger = get_logger()
    
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for HuggingFace models")
    
    if isinstance(layers, int):
        layers = [layers]
    
    if device is None:
        device = next(model.parameters()).device
    
    logger.info(f"Extracting activations (HuggingFace) from {len(prompts)} prompts, layers: {layers}")
    
    # Find the transformer layers module
    # Common patterns: model.model.layers, model.transformer.h, model.gpt_neox.layers
    layers_module = None
    for attr_path in ['model.layers', 'transformer.h', 'gpt_neox.layers', 'model.decoder.layers']:
        try:
            obj = model
            for attr in attr_path.split('.'):
                obj = getattr(obj, attr)
            layers_module = obj
            logger.info(f"Found layers at: {attr_path}")
            break
        except AttributeError:
            continue
    
    if layers_module is None:
        raise ValueError("Could not find transformer layers in model. Please check model architecture.")
    
    # Results storage
    all_activations = {layer: [] for layer in layers}
    all_tokens = []
    
    # Process in batches
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting activations (HF)")
    
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        # Store tokens
        for i in range(len(batch_prompts)):
            seq_len = inputs.attention_mask[i].sum().item()
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[i][:int(seq_len)])
            all_tokens.append(tokens)
        
        # Storage for this batch
        batch_activations = {layer: None for layer in layers}
        hooks = []
        
        # Register hooks
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is usually a tuple (hidden_states, ...)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                batch_activations[layer_idx] = hidden_states.detach().cpu()
            return hook_fn
        
        for layer_idx in layers:
            if layer_idx < len(layers_module):
                hook = layers_module[layer_idx].register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Collect activations
        for layer in layers:
            if batch_activations[layer] is not None:
                # Mask padding
                mask = inputs.attention_mask.unsqueeze(-1).cpu()  # [batch, seq_len, 1]
                acts = batch_activations[layer] * mask
                all_activations[layer].append(acts)
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate results
    result = {
        "activations": {
            layer: torch.cat(all_activations[layer], dim=0) if all_activations[layer] else None
            for layer in layers
        },
        "tokens": all_tokens,
        "layers": layers,
        "n_samples": len(prompts),
    }
    
    if return_attention:
        result["attention"] = {layer: None for layer in layers}  # Not implemented for HF fallback
        logger.warning("Attention extraction not implemented for HuggingFace fallback")
    
    return result


def extract_all_layers(
    model,
    prompts: List[str],
    tokenizer=None,
    batch_size: int = 1,
    max_length: int = 512,
    device: Optional[str] = None,
    show_progress: bool = True,
    pool_tokens: bool = False,
) -> Dict[str, Any]:
    """
    Extract activations from ALL layers in a single forward pass.
    
    This is critical for efficient layer ablation - avoids n_layers forward passes.
    
    Args:
        model: HookedTransformer model
        prompts: List of input prompts
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        device: Device
        show_progress: Show progress bar
        pool_tokens: If True, mean pool across tokens
        
    Returns:
        Dictionary with:
            - 'activations': Dict[layer_idx, Tensor]
            - 'tokens': List of token lists
            - 'metadata': Model info
    """
    logger = get_logger()
    
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
    
    if device is None:
        device = next(model.parameters()).device
    
    n_layers = model.cfg.n_layers
    all_layers = list(range(n_layers))
    
    # Build hook names for all layers
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in all_layers]
    
    logger.info(f"Extracting ALL {n_layers} layers from {len(prompts)} prompts")
    
    # Storage
    all_activations = {layer: [] for layer in all_layers}
    all_tokens = []
    hidden_norms = []  # Track for logging
    
    # Process in batches
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Extracting {n_layers} layers")
    
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        # Store tokens
        for i in range(len(batch_prompts)):
            seq_len = inputs.attention_mask[i].sum().item()
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[i][:int(seq_len)])
            all_tokens.append(tokens)
        
        # Forward with cache for all layers
        with torch.no_grad():
            _, cache = model.run_with_cache(
                inputs.input_ids,
                names_filter=lambda name: name in hook_names,
                return_type="logits",
            )
        
        # Extract each layer
        mask = inputs.attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        
        for layer in all_layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            acts = cache[hook_name] * mask  # Mask padding
            
            if pool_tokens:
                # Mean pool across tokens (excluding padding)
                seq_lens = inputs.attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
                pooled = acts.sum(dim=1) / seq_lens.squeeze(-1)
                all_activations[layer].append(pooled.cpu())
            else:
                all_activations[layer].append(acts.cpu())
        
        # Track norms for logging
        final_layer_acts = cache[f"blocks.{n_layers-1}.hook_resid_post"]
        norms = final_layer_acts.norm(dim=-1).mean(dim=1).cpu().tolist()
        hidden_norms.extend(norms)
        
        del cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate
    for layer in all_layers:
        all_activations[layer] = torch.cat(all_activations[layer], dim=0)
    
    result = {
        "activations": all_activations,
        "tokens": all_tokens,
        "n_layers": n_layers,
        "d_model": model.cfg.d_model,
        "n_samples": len(prompts),
        "hidden_norms": hidden_norms,
    }
    
    logger.info(f"Extracted: {n_layers} layers, {len(prompts)} samples")
    
    return result



def extract_activations_with_positions(
    model,
    prompts: List[str],
    layer: int,
    positions: List[List[int]],
    tokenizer=None,
    batch_size: int = 1,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Extract activations at specific token positions.
    
    Args:
        model: HookedTransformer model
        prompts: List of input prompts
        layer: Layer to extract from
        positions: List of position indices per sample (list of lists)
        tokenizer: Tokenizer
        batch_size: Batch size
        device: Device
        
    Returns:
        Tensor of shape [n_samples, max_positions, d_model]
    """
    result = extract_activations(
        model=model,
        prompts=prompts,
        layers=[layer],
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    
    acts = result["activations"][layer]  # [n_samples, seq_len, d_model]
    
    # Extract at specified positions
    max_positions = max(len(p) for p in positions)
    d_model = acts.shape[-1]
    
    extracted = torch.zeros(len(prompts), max_positions, d_model)
    
    for i, pos_list in enumerate(positions):
        for j, pos in enumerate(pos_list):
            if pos < acts.shape[1]:
                extracted[i, j] = acts[i, pos]
    
    return extracted


class ActivationCache:
    """
    Disk cache for model activations.
    
    Caches activations to avoid recomputation.
    """
    
    def __init__(self, cache_dir: Union[str, Path], version: str = "v1"):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache files
            version: Cache version (invalidate old caches)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.logger = get_logger()
    
    def _get_cache_key(
        self,
        model_name: str,
        dataset_name: str,
        layers: List[int],
        sample_ids: Optional[List[str]] = None,
    ) -> str:
        """Generate cache key."""
        key_data = {
            "version": self.version,
            "model": model_name,
            "dataset": dataset_name,
            "layers": sorted(layers),
            "sample_ids": sample_ids,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cache file."""
        return self.cache_dir / f"{cache_key}.pt"
    
    def exists(
        self,
        model_name: str,
        dataset_name: str,
        layers: List[int],
        sample_ids: Optional[List[str]] = None,
    ) -> bool:
        """Check if cache exists."""
        key = self._get_cache_key(model_name, dataset_name, layers, sample_ids)
        return self._get_cache_path(key).exists()
    
    def load(
        self,
        model_name: str,
        dataset_name: str,
        layers: List[int],
        sample_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load from cache if exists."""
        key = self._get_cache_key(model_name, dataset_name, layers, sample_ids)
        path = self._get_cache_path(key)
        
        if not path.exists():
            return None
        
        self.logger.info(f"Loading activations from cache: {path}")
        return torch.load(path, weights_only=False)
    
    def save(
        self,
        data: Dict[str, Any],
        model_name: str,
        dataset_name: str,
        layers: List[int],
        sample_ids: Optional[List[str]] = None,
    ) -> Path:
        """Save to cache."""
        key = self._get_cache_key(model_name, dataset_name, layers, sample_ids)
        path = self._get_cache_path(key)
        
        # Add metadata
        data["_cache_meta"] = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "layers": layers,
            "version": self.version,
        }
        
        torch.save(data, path)
        self.logger.info(f"Saved activations to cache: {path}")
        return path
