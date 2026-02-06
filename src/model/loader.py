"""
Model loading utilities.

Supports loading models with TransformerLens for activation extraction.
For models not directly supported (like DeepSeek), we load the HuggingFace
weights first and wrap them in a compatible TransformerLens architecture.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.config import get_config_dir, load_config
from src.utils.logging import get_logger


# Model type aliases
ModelType = Any  # HookedTransformer or PreTrainedModel
TokenizerType = Any  # Tokenizer


def load_model_config(model_name: str) -> DictConfig:
    """
    Load model configuration from YAML.
    
    Args:
        model_name: Name of model (e.g., 'deepseek_7b', 'qwen_7b')
        
    Returns:
        Model configuration
    """
    config_path = get_config_dir() / "model" / f"{model_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Model config not found: {config_path}")
    return OmegaConf.load(config_path)


def load_model(
    model_name: str,
    device: Optional[str] = None,
    use_transformer_lens: bool = True,
    torch_dtype: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[ModelType, TokenizerType]:
    """
    Load a model and tokenizer by name.
    
    For models not natively supported by TransformerLens (like DeepSeek),
    we load the HuggingFace weights and wrap them using a compatible
    architecture (Qwen for DeepSeek).
    
    Args:
        model_name: Model name ('deepseek_7b' or 'qwen_7b')
        device: Device to load on (auto-detected if None)
        use_transformer_lens: If True, wrap with HookedTransformer
        torch_dtype: Override dtype ('float16', 'bfloat16', 'float32')
        cache_dir: HuggingFace cache directory
        
    Returns:
        (model, tokenizer) tuple
    """
    logger = get_logger()
    
    # Load model config
    config = load_model_config(model_name)
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine dtype
    if torch_dtype is None:
        torch_dtype = config.loading.get("torch_dtype", "float16")
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)
    
    logger.info(f"Loading model: {config.display_name}")
    logger.info(f"Device: {device}, dtype: {torch_dtype}")
    
    hf_model_id = config.hf_model_id
    
    # Check if we need to use an alias (for unsupported models like DeepSeek)
    tl_alias = config.get("hf_base_id", None)  # e.g., "Qwen/Qwen2.5-7B-Instruct" for DeepSeek
    
    if use_transformer_lens:
        if tl_alias:
            # Load HuggingFace weights first, then wrap with TL alias architecture
            logger.info(f"Loading weights from {hf_model_id}, wrapping as {tl_alias}")
            model, tokenizer = _load_with_hf_and_wrap(
                hf_model_id=hf_model_id,
                tl_alias=tl_alias,
                device=device,
                dtype=dtype,
                config=config,
                cache_dir=cache_dir,
            )
        else:
            # Direct TransformerLens load
            model, tokenizer = _load_with_transformer_lens(
                config=config,
                device=device,
                dtype=dtype,
                cache_dir=cache_dir,
            )
    else:
        model, tokenizer = _load_with_transformers(
            hf_model_id=hf_model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=config.loading.get("trust_remote_code", True),
            cache_dir=cache_dir,
        )
    
    logger.info(f"Model loaded successfully. Layers: {_get_n_layers(model)}")
    return model, tokenizer


def _load_with_hf_and_wrap(
    hf_model_id: str,
    tl_alias: str,
    device: str,
    dtype: torch.dtype,
    config: DictConfig,
    cache_dir: Optional[str] = None,
) -> Tuple[ModelType, TokenizerType]:
    """
    Load HuggingFace model weights and wrap with TransformerLens.
    
    This is the key technique for loading DeepSeek and other models
    not directly supported by TransformerLens.
    
    The approach:
    1. Load HuggingFace model weights to CPU
    2. Load the TransformerLens-compatible architecture (e.g., Qwen)
    3. Pass hf_model to from_pretrained to transfer weights
    """
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger = get_logger()
    
    # Step 1: Load HuggingFace model to CPU first (saves GPU memory)
    logger.info(f"Downloading {hf_model_id} weights to CPU...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=dtype,
        trust_remote_code=config.loading.get("trust_remote_code", True),
        device_map="cpu",  # Load to CPU first to avoid OOM
        cache_dir=cache_dir,
    )
    
    # PATCH: Ensure config has rope_theta (needed for TransformerLens)
    # Use deepcopy to ensure we have a clean object we can modify safely
    import copy
    patched_config = copy.deepcopy(hf_model.config)
    
    if not hasattr(patched_config, "rope_theta"):
        logger.warning(f"Config type {type(patched_config)} missing 'rope_theta', patching...")
        setattr(patched_config, "rope_theta", 1000000.0)
    
    # Verify patch worked
    if hasattr(patched_config, "rope_theta"):
        logger.info(f"Verified 'rope_theta' is present in patched config: {patched_config.rope_theta}")
    else:
        logger.error("Failed to patch 'rope_theta'!")
    
    # Step 2: Load tokenizer from the actual model
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        trust_remote_code=config.loading.get("trust_remote_code", True),
        cache_dir=cache_dir,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 3: Wrap with TransformerLens using the alias architecture
    logger.info(f"Wrapping weights in HookedTransformer ({tl_alias}) on {device}...")
    
    tl_config = config.get("transformer_lens", {})
    
    hooked_model = HookedTransformer.from_pretrained(
        tl_alias,  # Use the alias (e.g., Qwen) for architecture
        hf_model=hf_model,  # Pass the actual weights
        hf_config=patched_config, # Explicitly pass the DEEPCOPIED and PATCHED config
        tokenizer=tokenizer,
        device=device,
        fold_ln=tl_config.get("fold_ln", False),
        fold_value_biases=tl_config.get("fold_value_biases", False),
        center_writing_weights=tl_config.get("center_writing_weights", False),
        center_unembed=tl_config.get("center_unembed", False),
    )
    
    # Free CPU memory from original HF model
    del hf_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info("✅ Model wrapped successfully!")
    
    return hooked_model, tokenizer


def _load_with_transformer_lens(
    config: DictConfig,
    device: str,
    dtype: torch.dtype,
    cache_dir: Optional[str] = None,
) -> Tuple[ModelType, TokenizerType]:
    """Load model directly with TransformerLens (native support)."""
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    
    hf_model_id = config.hf_model_id
    tl_config = config.get("transformer_lens", {})
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        trust_remote_code=config.loading.get("trust_remote_code", True),
        cache_dir=cache_dir,
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load with TransformerLens
    model = HookedTransformer.from_pretrained(
        hf_model_id,
        device=device,
        dtype=dtype,
        fold_ln=tl_config.get("fold_ln", False),
        fold_value_biases=tl_config.get("fold_value_biases", False),
        center_writing_weights=tl_config.get("center_writing_weights", False),
        center_unembed=tl_config.get("center_unembed", False),
        cache_dir=cache_dir,
    )
    
    model.tokenizer = tokenizer
    
    return model, tokenizer


def _load_with_transformers(
    hf_model_id: str,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[ModelType, TokenizerType]:
    """Load model directly with HuggingFace Transformers (no TL wrapper)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    
    if device != "cuda":
        model = model.to(device)
    
    return model, tokenizer


def _get_n_layers(model: ModelType) -> int:
    """Get number of layers in model."""
    if hasattr(model, "cfg"):  # HookedTransformer
        return model.cfg.n_layers
    elif hasattr(model, "config"):  # HuggingFace
        return getattr(model.config, "num_hidden_layers", 
                      getattr(model.config, "n_layer", -1))
    return -1


def get_d_model(model: ModelType) -> int:
    """Get model hidden dimension."""
    if hasattr(model, "cfg"):  # HookedTransformer
        return model.cfg.d_model
    elif hasattr(model, "config"):  # HuggingFace
        return getattr(model.config, "hidden_size", 
                      getattr(model.config, "d_model", -1))
    return -1


def unload_model(model: ModelType) -> None:
    """
    Unload model and free memory.
    
    Args:
        model: Model to unload
    """
    logger = get_logger()
    
    del model
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Model unloaded and memory cleared")


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get model information without loading.
    
    Args:
        model_name: Model name
        
    Returns:
        Dictionary with model info
    """
    config = load_model_config(model_name)
    
    return {
        "name": config.name,
        "display_name": config.display_name,
        "type": config.type,
        "hf_model_id": config.hf_model_id,
        "tl_alias": config.get("hf_base_id", None),
        "n_layers": config.architecture.n_layers,
        "d_model": config.architecture.d_model,
        "n_heads": config.architecture.n_heads,
    }
