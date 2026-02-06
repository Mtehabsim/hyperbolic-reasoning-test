#!/usr/bin/env python3
"""
Extract activations for ALL layers in a single forward pass.

This is critical for efficient layer ablation experiments - avoids
8x forward passes by caching all layers at once.

Usage:
    # CPU mode (for testing/local dev)
    python scripts/extract_all_activations.py --model qwen_7b --dataset prontoqa --cpu
    
    # GPU mode
    python scripts/extract_all_activations.py --model deepseek_7b --dataset prontoqa
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import Dataset, PrOntoQAGenerator, ListOpsGenerator, BinaryTreeGenerator
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed


def extract_all_layers_cpu(
    prompts: List[str],
    n_layers: int = 28,
    d_model: int = 3584,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Mock extraction for CPU testing.
    
    Generates random activations with correct shapes for testing
    the rest of the pipeline without GPU or transformers.
    """
    logger = get_logger()
    logger.info("Running CPU mock extraction - generating random activations")
    
    all_activations = {layer: [] for layer in range(n_layers)}
    all_tokens = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Mock extraction")):
        # Simple word tokenization (no HuggingFace dependency)
        words = prompt.split()[:max_length]
        token_strs = [f"token_{j}" for j in range(len(words))]
        all_tokens.append(token_strs)
        
        seq_len = len(words)
        
        # Generate random activations for each layer (mock)
        for layer in range(n_layers):
            # Random activation with realistic norm
            activation = torch.randn(1, seq_len, d_model) * 0.1
            all_activations[layer].append(activation)
    
    # Concatenate per layer (pad to max seq len for simplicity)
    max_seq = max(all_activations[0][i].shape[1] for i in range(len(prompts)))
    for layer in range(n_layers):
        padded = []
        for act in all_activations[layer]:
            if act.shape[1] < max_seq:
                padding = torch.zeros(1, max_seq - act.shape[1], d_model)
                act = torch.cat([act, padding], dim=1)
            padded.append(act)
        all_activations[layer] = torch.cat(padded, dim=0)
    
    return {
        "activations": all_activations,
        "tokens": all_tokens,
    }



def extract_all_layers_gpu(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 1,
    max_length: int = 512,
    save_attention: bool = False,
) -> Dict[str, Any]:
    """
    Extract all layers in single forward pass using TransformerLens.
    
    Caches all layer activations to avoid multiple forward passes.
    Optionally extracts attention patterns for attention-weighted token selection.
    
    Args:
        model: TransformerLens HookedTransformer model
        tokenizer: Tokenizer
        prompts: List of prompts to process
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        save_attention: Whether to also save attention patterns
        
    Returns:
        Dict with 'activations', 'tokens', and optionally 'attention'
    """
    logger = get_logger()
    n_layers = model.cfg.n_layers
    
    all_activations = {layer: [] for layer in range(n_layers)}
    all_attention = {layer: [] for layer in range(n_layers)} if save_attention else None
    all_tokens = []
    
    # Hook names for all residual stream outputs
    names_filter = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    
    # Add attention pattern hooks if saving attention
    if save_attention:
        attention_hooks = [f"blocks.{layer}.attn.hook_pattern" for layer in range(n_layers)]
        names_filter.extend(attention_hooks)
        logger.info(f"Saving attention patterns for {n_layers} layers")
    
    logger.info(f"Extracting {n_layers} layers for {len(prompts)} samples...")
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(model.cfg.device) for k, v in inputs.items()}
        
        # Forward with cache
        with torch.no_grad():
            _, cache = model.run_with_cache(
                inputs["input_ids"],
                names_filter=lambda name: name in names_filter,
            )
        
        # Extract activations per layer
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_resid_post"
            activations = cache[hook_name].cpu()
            all_activations[layer].append(activations)
            
            # Extract attention patterns if requested
            if save_attention:
                attn_hook = f"blocks.{layer}.attn.hook_pattern"
                if attn_hook in cache:
                    attn_pattern = cache[attn_hook].cpu()  # [batch, n_heads, seq, seq]
                    all_attention[layer].append(attn_pattern)
        
        # Store tokens
        for ids in inputs["input_ids"]:
            token_strs = tokenizer.convert_ids_to_tokens(ids.cpu().tolist())
            all_tokens.append(token_strs)
        
        # Clear cache to free memory
        del cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate batches with padding
    for layer in range(n_layers):
        layer_acts = all_activations[layer]
        if not layer_acts:
            continue
            
        # Find max length in this batch of activations
        max_len = max(act.shape[1] for act in layer_acts)
        
        # Pad each activation tensor to max_len
        padded_acts = []
        for act in layer_acts:
            # act shape: [batch, seq_len, d_model]
            curr_len = act.shape[1]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                # Pad second dimension (sequence length)
                # F.pad format: (left, right, top, bottom, front, back)
                # specific for 3D tensor: (0, 0, 0, pad_len, 0, 0)
                # But since it's [batch, seq, d_model], we want to pad seq
                # This depends on dimension order.
                # TransformerLens activations are [batch, seq, d_model]
                # F.pad operates on last dim first.
                # So (0, 0) for d_model, (0, pad_len) for seq_len, (0, 0) for batch
                padded = torch.nn.functional.pad(act, (0, 0, 0, pad_len))
                padded_acts.append(padded)
            else:
                padded_acts.append(act)
        
        # Concatenate this layer and immediately clear the list to free memory
        all_activations[layer] = torch.cat(padded_acts, dim=0)
        del padded_acts
        del layer_acts
        gc.collect()
    
    # Concatenate attention patterns if saved (one layer at a time)
    if save_attention:
        for layer in range(n_layers):
            attn_acts = all_attention[layer]
            if not attn_acts:
                continue
            # Attention patterns may have different seq lengths, pad them
            max_len = max(act.shape[2] for act in attn_acts)  # seq dimension
            padded_attn = []
            for act in attn_acts:
                curr_len = act.shape[2]
                if curr_len < max_len:
                    pad_len = max_len - curr_len
                    # Pad both seq dimensions (query and key)
                    padded = torch.nn.functional.pad(act, (0, pad_len, 0, pad_len))
                    padded_attn.append(padded)
                else:
                    padded_attn.append(act)
            all_attention[layer] = torch.cat(padded_attn, dim=0)
            del padded_attn
            del attn_acts
            gc.collect()
    
    result = {
        "activations": all_activations,
        "tokens": all_tokens,
    }
    
    if save_attention:
        result["attention"] = all_attention
        logger.info(f"Extracted attention patterns for {n_layers} layers")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract all-layer activations")
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen_7b",
        choices=["deepseek_7b", "qwen_7b", "gpt2"],
        help="Model to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="prontoqa",
        choices=["prontoqa", "listops", "binary_tree", "mock"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: outputs/activations/{model}_{dataset}.pt)"
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (mock extraction)")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/data"), help="Directory containing datasets")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--save-attention", action="store_true", help="Also save attention patterns (for H2/head ablation)")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    logger = get_logger()
    
    # Output path
    if args.output is None:
        args.output = Path(f"outputs/activations/{args.model}_{args.dataset}.pt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate/load dataset
    logger.info(f"Generating {args.dataset} dataset...")
    
    logger.info(f"Loading/Generating {args.dataset} dataset...")
    
    # Try to load from data-dir first
    dataset_path = args.data_dir / f"{args.dataset}_test.json"
    if dataset_path.exists():
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = Dataset.load(dataset_path)
        # Verify sample count
        if len(dataset) < args.n_samples and args.n_samples != 100: # 100 is default
             logger.warning(f"Loaded dataset has {len(dataset)} samples, but asked for {args.n_samples}")
    else:
        logger.info(f"Dataset not found at {dataset_path}, generating...")
        if args.dataset == "prontoqa":
            gen = PrOntoQAGenerator(seed=args.seed)
            n_samples = args.n_samples if args.n_samples > 100 else 1000 # Default to 1000 if not specified
            dataset = gen.generate(n_samples=n_samples) # Use generate method directly
        elif args.dataset == "binary_tree":
            gen = BinaryTreeGenerator(seed=args.seed, tree_depth=5)
            dataset = gen.generate_dataset(n_samples=args.n_samples)
        else:
            gen = ListOpsGenerator(seed=args.seed)
            dataset = gen.generate(n_samples=args.n_samples)
    
    prompts = [s.prompt for s in dataset.samples]
    logger.info(f"Dataset: {len(prompts)} samples")
    
    # Determine mode
    use_gpu = torch.cuda.is_available() and not args.cpu
    
    if use_gpu:
        logger.info("Using GPU extraction with TransformerLens")
        
        # Import and load model
        from src.model.loader import load_model
        
        # load_model expects model_name (e.g., 'deepseek_7b'), not config path
        model, tokenizer = load_model(args.model, device="cuda")
        
        result = extract_all_layers_gpu(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            save_attention=args.save_attention,
        )
        
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        
        # Cleanup model to free GPU memory before saving
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    else:
        logger.info("Using CPU mock extraction (for testing)")
        
        # Mock values
        n_layers = 28
        d_model = 3584
        
        result = extract_all_layers_cpu(
            prompts=prompts,
            n_layers=n_layers,
            d_model=d_model,
            max_length=args.max_length,
        )


    
    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_samples": len(prompts),
        "sample_ids": [s.id for s in dataset.samples],
        "depths": [s.depth for s in dataset.samples],
        "labels": [s.label for s in dataset.samples],
        "graph_distances": [
            s.graph_distances.tolist() if s.graph_distances is not None else None
            for s in dataset.samples
        ],
        "node_ids": [s.node_ids for s in dataset.samples],
        "timestamp": datetime.now().isoformat(),
        "cpu_mode": not use_gpu,
        "attention_saved": args.save_attention and "attention" in result,
    }
    
    # Memory-efficient incremental save
    # Save each layer separately, then combine metadata
    logger.info("Saving layers incrementally (memory-efficient mode)...")
    
    # Create output directory for per-layer files
    output_dir = args.output.parent / f"{args.output.stem}_layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save activations one layer at a time
    activations = result["activations"]
    for layer in range(n_layers):
        layer_data = activations[layer]
        layer_path = output_dir / f"layer_{layer:02d}.pt"
        torch.save(layer_data, layer_path)
        # Free memory immediately
        activations[layer] = None
        del layer_data
        gc.collect()
        if layer % 7 == 0:
            logger.info(f"  Saved layer {layer}/{n_layers-1}")
    
    # Save attention if present (one layer at a time)
    if args.save_attention and "attention" in result:
        logger.info("Saving attention patterns...")
        attention = result["attention"]
        for layer in range(n_layers):
            if attention[layer] is not None:
                attn_path = output_dir / f"attention_{layer:02d}.pt"
                torch.save(attention[layer], attn_path)
                attention[layer] = None
                gc.collect()
    
    # Save tokens and metadata
    torch.save({
        "tokens": result["tokens"],
        "metadata": metadata,
        "layer_files": [f"layer_{i:02d}.pt" for i in range(n_layers)],
    }, output_dir / "metadata.pt")
    
    # Now combine into single file if memory allows
    logger.info("Combining layers into single file...")
    try:
        combined_activations = {}
        for layer in range(n_layers):
            layer_path = output_dir / f"layer_{layer:02d}.pt"
            combined_activations[layer] = torch.load(layer_path)
        
        save_data = {
            "activations": combined_activations,
            "tokens": result["tokens"],
            "metadata": metadata,
        }
        
        if args.save_attention and metadata["attention_saved"]:
            combined_attention = {}
            for layer in range(n_layers):
                attn_path = output_dir / f"attention_{layer:02d}.pt"
                if attn_path.exists():
                    combined_attention[layer] = torch.load(attn_path)
            save_data["attention"] = combined_attention
        
        torch.save(save_data, args.output)
        logger.info(f"Saved combined file to {args.output}")
        
        # Clean up layer files
        import shutil
        shutil.rmtree(output_dir)
        
    except Exception as e:
        logger.warning(f"Could not combine into single file (likely OOM): {e}")
        logger.info(f"Layer files saved to {output_dir}/ - use these directly")
        # Update output path to point to directory
        args.output = output_dir / "metadata.pt"
    
    logger.info("\n=== Extraction Summary ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {len(prompts)}")
    logger.info(f"Layers: {n_layers}")
    logger.info(f"d_model: {d_model}")
    logger.info(f"Attention saved: {args.save_attention and metadata['attention_saved']}")
    logger.info(f"Mode: {'GPU' if use_gpu else 'CPU (mock)'}")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
