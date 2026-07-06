#!/usr/bin/env python3
"""
Main experiment runner.

Runs the complete experiment pipeline for hyperbolic probing experiments.

Usage:
    python run_experiments.py --experiment h1 --model deepseek
    python run_experiments.py --experiment h2 --model qwen
    python run_experiments.py --experiment all --model all
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data import generate_prontoqa_datasets, generate_listops_datasets, Dataset
from src.model import load_model, extract_activations, extract_activations_with_generation, TokenSelector
from src.experiments import HierarchyExperiment, TokenAblationExperiment, ProbeTrainingConfig
from src.analysis import compare_probe_types, compute_all_statistics, bonferroni_correction, plot_euclidean_vs_hyperbolic, plot_token_ablation
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed
from src.utils.config import load_config
import numpy as np


def make_json_serializable(obj):
    """Convert any object to JSON serializable format, handling numpy/circular refs."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)  # Convert objects to string representation
    return obj


def _tree_distance_from_label_paths(label_paths):
    """Shared-prefix tree distance between samples' taxonomy label-paths.

    For paths p_i, p_j (root->leaf lists of node ids), distance =
    (depth_i - shared) + (depth_j - shared), where shared = length of the common
    prefix. Same-branch/sibling leaves are close; cross-branch leaves are far.
    This is the BRANCHING-tree target for H1.5, replacing H1's 1-D
    |depth_i - depth_j| ruler. Returns an (n, n) numpy matrix.
    """
    import numpy as np
    n = len(label_paths)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        pi = label_paths[i]
        for j in range(n):
            pj = label_paths[j]
            shared = 0
            for a, b in zip(pi, pj):
                if a == b:
                    shared += 1
                else:
                    break
            d[i, j] = (len(pi) - shared) + (len(pj) - shared)
    return d


def run_h1_experiment(model_name: str, config: dict, output_dir: Path, cached_activations_path: Path = None,
                      target_mode: str = "depth", out_prefix: str = "h1") -> dict:
    """
    Run H1 / H1.5: Hierarchy encoding experiment.

    Compares Euclidean vs Hyperbolic probes for distance preservation.

    ``target_mode`` selects what the probe regresses onto -- the ONLY difference
    between H1 and H1.5 (everything else -- extraction, probes, metrics, stats,
    plots -- is shared):
      - "depth"    : H1. Target = |depth_i - depth_j| (a 1-D reasoning-depth ruler;
                     Raj's setup). A line fits Euclidean and hyperbolic equally.
      - "taxonomy" : H1.5. Target = shared-prefix TREE distance from each sample's
                     metadata["label_path"] (a branching harm taxonomy). This is
                     where hyperbolic geometry can genuinely win, if the hierarchy
                     is encoded. Falls back to depth if no label_path is present.
    ``out_prefix`` names the output files (h1_* vs h15_*).
    """
    logger = get_logger()
    logger.info(f"=== H1 Experiment: {model_name} ===")
    
    # Load model (skip if using cached activations)
    if cached_activations_path and cached_activations_path.exists():
        logger.info(f"Using cached activations from {cached_activations_path}")
        model = None
        tokenizer = None
        # Load metadata later
    else:
        logger.info("Loading model...")
        model, tokenizer = load_model(model_name)
    
    # Determine dataset to use (default: prontoqa)
    dataset_name = config.get("dataset", "prontoqa")
    logger.info(f"Using dataset: {dataset_name}")
    
    # Load or generate dataset based on config
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == "binarytree":
        from src.data import generate_binary_tree_datasets
        # Check for existing binary tree data that matches cached activations
        # The cached activations were extracted using data/binarytree_depth5.json
        existing_data_path = Path("data/binarytree_depth5.json")
        if cached_activations_path and existing_data_path.exists():
            # Use existing data that matches cached activations
            logger.info(f"Using existing binary tree data from {existing_data_path}")
            test_path = existing_data_path
            train_path = existing_data_path  # Same file for train/test in this format
        else:
            # Generate new data (only when not using cached activations)
            train_path = data_dir / "binary_tree_train.json"
            test_path = data_dir / "binary_tree_test.json"
            if not train_path.exists():
                logger.info("Generating Binary Tree dataset...")
                generate_binary_tree_datasets(data_dir, n_samples=1000, tree_depth=5)
    elif dataset_name == "listops":
        from src.data import generate_listops_datasets
        train_path = data_dir / "listops_train.json"
        test_path = data_dir / "listops_test.json"
        if not train_path.exists():
            logger.info("Generating ListOps dataset...")
            generate_listops_datasets(data_dir, n_train=500, n_test=500)
    elif dataset_name == "ailuminate":
        # AILuminate harm taxonomy (H1.5 target lives here). CRITICAL: without this
        # branch the code fell through to the prontoqa `else`, loading PrOntoQA
        # labels while the cached .pt held AILuminate activations -> every
        # activation paired with the WRONG sample -> rho~0 by construction (this
        # was the real cause of the rho~0 "result", NOT the placeholder prompts).
        from src.data import generate_ailuminate_datasets
        train_path = data_dir / "ailuminate_train.json"
        test_path = data_dir / "ailuminate_test.json"
        if not test_path.exists():
            logger.info("Generating AILuminate dataset...")
            # 1000 test to match the cached activations extracted with n_test=1000.
            generate_ailuminate_datasets(data_dir, n_test=1000, n_train=1000, seed=config.get("seed", 42))
    else:  # default: prontoqa
        train_path = data_dir / "prontoqa_train.json"
        test_path = data_dir / "prontoqa_test.json"
        if not train_path.exists():
            logger.info("Generating PrOntoQA dataset...")
            # Use 500+500=1000 test samples to match cached activations (extracted with 1000)
            generate_prontoqa_datasets(data_dir, n_test_true=500, n_test_false=500)
    
    # Load dataset - handle different formats
    raw_binarytree_data = None
    import numpy as np  # Import early for binary tree processing
    if dataset_name == "binarytree" and str(test_path).endswith("binarytree_depth5.json"):
        # Load raw JSON for binary tree pairwise distance format
        # NOTE: do NOT `import json` here -- a local import makes `json` a
        # function-local name for the WHOLE function, so on the non-binarytree
        # path (prontoqa/ailuminate) the later json.dump at save time raised
        # UnboundLocalError. The module-level `import json` (top of file) covers us.
        with open(test_path, "r") as f:
            raw_binarytree_data = json.load(f)
        logger.info(f"Loaded {len(raw_binarytree_data)} samples from binary tree data")
        # Create prompts list for activation extraction
        prompts = [s["prompt"] for s in raw_binarytree_data]
        # For binary tree, use actual pairwise distances directly
        distances = np.array([s["distance"] for s in raw_binarytree_data])
        n = len(distances)
    else:
        # Standard dataset loading
        test_dataset = Dataset.load(test_path)
        logger.info(f"Loaded {len(test_dataset)} test samples from {dataset_name}")
        prompts = [s.prompt for s in test_dataset]
        distances = None  # Compute from depth later
        n = len(test_dataset)
    
    # Extract activations
    logger.info("Extracting activations...")
    layers = config.get("layers", [8, 12, 16, 19, 21, 23, 25, 27])
    
    activation_result = extract_activations(
        model=model,
        prompts=prompts,
        layers=layers,
        tokenizer=tokenizer,
        batch_size=config.get("batch_size", 1),
        cached_activations_path=cached_activations_path,
    )

    # Get target distances from dataset
    import torch
    import numpy as np

    # ALIGNMENT GUARD: the taxonomy/depth target is built from `test_dataset`, but
    # the cached activations were extracted from whatever dataset order extraction
    # used. If the dataset json was regenerated with a different shuffle (or the
    # wrong dataset was loaded entirely), activations pair with the WRONG labels
    # -> rho~0. The cached .pt metadata carries the exact `sample_ids` in
    # activation-row order; reorder test_dataset to match it so pairing is correct
    # regardless of shuffle/regeneration. (No-op for binarytree raw path.)
    if raw_binarytree_data is None and cached_activations_path is not None:
        cached_meta = (activation_result or {}).get("metadata", {}) or {}
        cached_ids = cached_meta.get("sample_ids")
        ds_ids = [s.id for s in test_dataset]
        if cached_ids and set(cached_ids) == set(ds_ids) and list(cached_ids) != list(ds_ids):
            logger.warning("Reordering test_dataset to match cached-activation sample order "
                           "(shuffle/regeneration mismatch detected).")
            by_id = {s.id: s for s in test_dataset.samples}
            test_dataset.samples = [by_id[i] for i in cached_ids]
        elif cached_ids and set(cached_ids) != set(ds_ids):
            logger.error("MISALIGNMENT: cached-activation sample_ids do NOT match the loaded "
                         f"dataset '{dataset_name}' ({len(set(cached_ids) & set(ds_ids))}/"
                         f"{len(ds_ids)} overlap). Activations and labels are from different "
                         "datasets -> results will be meaningless. Re-extract with the SAME "
                         "dataset, or point --data-dir at the extracted dataset json.")
        prompts = [s.prompt for s in test_dataset]

    if raw_binarytree_data is not None:
        # Binary tree: use actual distances as 1D target (not pairwise matrix)
        # Since each sample IS a node pair, we use distances directly
        # For pairwise probing: |distance_i - distance_j| between samples
        target_distances = torch.tensor(
            np.abs(distances.reshape(-1, 1) - distances.reshape(1, -1)),
            dtype=torch.float32
        )
    elif target_mode == "taxonomy":
        # H1.5: BRANCHING taxonomy tree distance from each sample's label_path.
        label_paths = [list(s.metadata.get("label_path", [s.depth])) for s in test_dataset]
        have_paths = any(len(p) > 1 for p in label_paths)
        if not have_paths:
            logger.warning("target_mode='taxonomy' but no metadata['label_path'] found "
                           "(len>1); falling back to depth ruler. Use a taxonomy dataset "
                           "(e.g. ailuminate) for H1.5.")
            depths = np.array([s.depth for s in test_dataset])
            target_distances = torch.tensor(
                np.abs(depths.reshape(-1, 1) - depths.reshape(1, -1)),
                dtype=torch.float32,
            )
        else:
            logger.info(f"H1.5 taxonomy target: {len(set(tuple(p) for p in label_paths))} "
                        f"distinct label-paths (branching tree distance)")
            target_distances = torch.tensor(
                _tree_distance_from_label_paths(label_paths),
                dtype=torch.float32,
            )
    else:
        # H1: PrOntoQA/ListOps use depth as a 1-D proxy.
        depths = np.array([s.depth for s in test_dataset])
        if len(np.unique(depths)) < 2:
            logger.warning(
                f"H1 depth target is DEGENERATE: all {len(depths)} samples have "
                f"depth={depths[0]}, so |depth_i-depth_j|=0 everywhere and the probe "
                f"fits noise. Dataset '{dataset_name}' has no depth variation "
                f"(flat taxonomy?) -- use --experiment h1.5 (taxonomy tree target) instead."
            )
        # Target: Cross-sample depth differences as hierarchy proxy
        target_distances = torch.tensor(
            np.abs(depths.reshape(-1, 1) - depths.reshape(1, -1)),
            dtype=torch.float32
        )
    
    # Prepare probe config (filter/map keys to match ProbeTrainingConfig)
    probe_config = config.get("probe", {}).copy()
    # Map 'embedding_dim' -> 'output_dim'
    if "embedding_dim" in probe_config:
        probe_config["output_dim"] = probe_config.pop("embedding_dim")
    # Remove keys not in ProbeTrainingConfig
    probe_config.pop("type", None)  # probe type handled separately
    
    # Mean pool activations for each sample and convert to float32
    activations_per_layer = {}
    for layer in layers:
        acts = activation_result["activations"][layer]  # [n_samples, seq_len, d_model]
        pooled = acts.mean(dim=1).float()  # [n_samples, d_model] - convert to float32
        activations_per_layer[layer] = pooled
    
    # Get input_dim from activations (not model, which may be None for cached)
    first_layer = layers[0]
    input_dim = activations_per_layer[first_layer].shape[-1]
    n_activation_samples = activations_per_layer[first_layer].shape[0]
    n_target_samples = target_distances.shape[0]
    
    # Align sample counts (cached activations may have different size than fresh dataset)
    if n_activation_samples != n_target_samples:
        logger.warning(f"Sample count mismatch: activations={n_activation_samples}, targets={n_target_samples}")
        n_samples = min(n_activation_samples, n_target_samples)
        logger.info(f"Using first {n_samples} samples for alignment")
        
        # Slice activations
        for layer in layers:
            activations_per_layer[layer] = activations_per_layer[layer][:n_samples]
        
        # Slice target distances
        target_distances = target_distances[:n_samples, :n_samples]
    
    # Run hierarchy experiment
    # Device: honor config, but fall back to CPU when CUDA is unavailable so the
    # pipeline runs on a laptop (probe training is small). DGX keeps cuda.
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available -> running probes on CPU")
        device = "cpu"

    experiment = HierarchyExperiment(
        input_dim=input_dim,
        layers=layers,
        config=ProbeTrainingConfig(**probe_config),
        seed=config.get("seed", 42),
        device=device,
    )
    
    results = experiment.run_all_layers(activations_per_layer, target_distances)
    
    # Save results (out_prefix = "h1" for depth / "h15" for taxonomy)
    experiment.save_results(output_dir / f"{out_prefix}_results.json")

    # Generate plots
    result_dicts = [r.to_dict() for r in results]
    fig = plot_euclidean_vs_hyperbolic(result_dicts, output_dir / f"{out_prefix}_comparison.png")
    
    # Extract scores for comprehensive statistical analysis
    euclidean_scores = [r["spearman_rho"] for r in result_dicts if r["probe_type"] == "euclidean"]
    hyperbolic_scores = [r["spearman_rho"] for r in result_dicts if r["probe_type"] == "hyperbolic"]
    lorentz_scores = [r["spearman_rho"] for r in result_dicts if r["probe_type"] == "lorentz"]
    
    # Compute comprehensive statistics with bootstrap CIs
    full_stats = compute_all_statistics(
        euclidean_scores=euclidean_scores,
        hyperbolic_scores=hyperbolic_scores,
        metric_name="spearman_rho",
        n_bootstrap=1000,
    )
    
    # Apply Bonferroni correction for layer-wise comparisons
    layer_p_values = []
    for layer in layers:
        layer_results = [r for r in result_dicts if r["layer"] == layer]
        euc_rho = next((r["spearman_rho"] for r in layer_results if r["probe_type"] == "euclidean"), None)
        hyp_rho = next((r["spearman_rho"] for r in layer_results if r["probe_type"] == "hyperbolic"), None)
        if euc_rho is not None and hyp_rho is not None:
            # Simple bootstrap test for each layer
            layer_p_values.append(full_stats["tests"].get("mann_whitney", {}).get("p_value", 1.0))
    
    corrected_p = bonferroni_correction(layer_p_values) if layer_p_values else []
    full_stats["bonferroni_corrected_p"] = corrected_p
    
    # Log summary
    logger.info(f"{out_prefix.upper()} Results Summary (target_mode={target_mode}):")
    logger.info(f"  Euclidean avg ρ: {full_stats['descriptive']['euclidean_mean']:.4f} (95% CI: {full_stats['descriptive']['euclidean_ci_95']})")
    logger.info(f"  Hyperbolic avg ρ: {full_stats['descriptive']['hyperbolic_mean']:.4f} (95% CI: {full_stats['descriptive']['hyperbolic_ci_95']})")
    logger.info(f"  Improvement: {full_stats['descriptive']['percent_improvement']:.1f}% (95% CI for diff: {full_stats['descriptive']['diff_ci_95']})")
    logger.info(f"  Effect size (Cohen's d): {full_stats['effect_size']['cohens_d']:.3f}")
    if "mann_whitney" in full_stats["tests"]:
        logger.info(f"  Mann-Whitney U p-value: {full_stats['tests']['mann_whitney']['p_value']:.4f}")
    logger.info(f"  Significant: {full_stats['significant']}")
    
    # Save extended results with statistics
    extended_results = {
        out_prefix: result_dicts,
        "statistics": full_stats,
        "lorentz_scores": lorentz_scores,
        "metadata": {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "layers": layers,
            "n_samples": target_distances.shape[0],
            "target_mode": target_mode,
            "dataset": dataset_name,
        }
    }

    with open(output_dir / f"{out_prefix}_full_results.json", "w") as f:
        # Convert all non-serializable types recursively (fixes circular reference)
        serializable_results = make_json_serializable(extended_results)
        json.dump(serializable_results, f, indent=2)

    return extended_results


def run_h15_experiment(model_name: str, config: dict, output_dir: Path, cached_activations_path: Path = None) -> dict:
    """H1.5: same as H1 but regress onto the BRANCHING taxonomy tree distance
    (from metadata['label_path']) instead of the 1-D depth ruler. Thin wrapper --
    all logic is shared with run_h1_experiment via target_mode."""
    return run_h1_experiment(
        model_name, config, output_dir, cached_activations_path,
        target_mode="taxonomy", out_prefix="h15",
    )


def run_h2_experiment(model_name: str, config: dict, output_dir: Path, generate_cot: bool = True) -> dict:
    """
    Run H2: Token selection ablation experiment.
    
    Tests whether reasoning concentrates at "thinking tokens".
    
    Args:
        model_name: Model to use
        config: Configuration dictionary
        output_dir: Output directory
        generate_cot: If True, generate reasoning traces with CoT prompting.
                     This is REQUIRED for thinking token detection in reasoning models.
    """
    logger = get_logger()
    logger.info(f"=== H2 Experiment: {model_name} (generate_cot={generate_cot}) ===")
    
    # Load model
    model, tokenizer = load_model(model_name)
    
    # Load dataset
    data_dir = output_dir / "data"
    test_path = data_dir / "prontoqa_test.json"
    
    if not test_path.exists():
        # Use 500+500=1000 test samples to match cached activations
        generate_prontoqa_datasets(data_dir, n_test_true=500, n_test_false=500)
    
    test_dataset = Dataset.load(test_path)
    
    # For H2, we use a subset to make generation feasible (generation is slow)
    n_samples = min(config.get("h2_n_samples", 200), len(test_dataset))
    test_dataset = test_dataset[:n_samples]
    logger.info(f"Using {n_samples} samples for H2 (generation-based)")

    # Multi-layer testing: use shared "layers" key (set by CLI --layers), fall back to h2_layers, then default
    layers = config.get("layers", config.get("h2_layers", [19, 21, 23, 25, 27]))
    prompts = [s.prompt for s in test_dataset]
    
    # Add CoT prompt prefix for reasoning models
    if generate_cot:
        cot_prefix = "Think step by step and explain your reasoning:\n\n"
        prompts = [cot_prefix + p for p in prompts]
    
    import torch
    import numpy as np
    
    # Extract activations - use generation for reasoning models
    if generate_cot:
        logger.info("Generating reasoning traces with CoT prompting...")
        activation_result = extract_activations_with_generation(
            model=model,
            prompts=prompts,
            layers=layers,
            tokenizer=tokenizer,
            max_new_tokens=256,  # Reasonable trace length
            return_attention=True,
            temperature=0.0,  # Greedy for reproducibility
        )
        # Log thinking token statistics
        thinking_positions = activation_result.get("thinking_positions", [])
        avg_thinking = sum(len(p) for p in thinking_positions) / max(len(thinking_positions), 1)
        logger.info(f"Average thinking tokens per sample: {avg_thinking:.1f}")
    else:
        # Fallback to input-only extraction (won't have thinking tokens)
        logger.warning("Using input-only extraction - thinking tokens will be sparse!")
        activation_result = extract_activations(
            model=model,
            prompts=prompts,
            layers=layers,
            tokenizer=tokenizer,
            return_attention=True,
        )
    
    # Get target distances from reasoning depth
    depths = np.array([s.depth for s in test_dataset])
    target_distances = torch.tensor(
        np.abs(depths.reshape(-1, 1) - depths.reshape(1, -1)),
        dtype=torch.float32
    )
    
    # Run token ablation for each layer - use output_dim=5 (optimal from Phase 3a)
    all_results = []
    probe_cfg = config.get("probe", {})
    output_dim = probe_cfg.get("embedding_dim", 5)
    curvature = probe_cfg.get("curvature", 0.5)
    
    for layer in layers:
        logger.info(f"Testing token selection methods for layer {layer}")
        
        experiment = TokenAblationExperiment(
            input_dim=model.cfg.d_model,
            layers=[layer],
            probe_type="hyperbolic",
            output_dim=output_dim,
            curvature=curvature,
            seed=config.get("seed", 42),
        )
        
        hidden_states = activation_result["activations"][layer]
        tokens_list = activation_result["tokens"]
        attention_weights = activation_result.get("attention", {}).get(layer)
        
        results = experiment.run_layer(
            layer=layer,
            hidden_states=hidden_states,
            tokens_list=tokens_list,
            target_distances=target_distances,
            attention_weights=attention_weights,
        )
        
        all_results.extend(results)
    
    # Save results
    result_dicts = [r.to_dict() for r in all_results]
    
    h2_output = {
        "experiment": "token_ablation_h2",
        "generate_cot": generate_cot,
        "n_samples": n_samples,
        "layers": layers,
        "output_dim": output_dim,
        "results": result_dicts,
    }
    
    with open(output_dir / "h2_results.json", "w") as f:
        json.dump(make_json_serializable(h2_output), f, indent=2)
    
    # Generate plots
    fig = plot_token_ablation(result_dicts, output_dir / "h2_comparison.png")
    
    # Summary by method
    summary = {}
    for method in ["thinking_tokens", "attention_weighted", "all_pool", "last_token", "random"]:
        method_results = [r for r in result_dicts if r["selection_method"] == method]
        if method_results:
            avg_rho = sum(r["spearman_rho"] for r in method_results) / len(method_results)
            avg_tokens = sum(r["n_tokens_used"] for r in method_results) / len(method_results)
            summary[method] = {"avg_rho": avg_rho, "avg_tokens": avg_tokens}
    
    logger.info(f"H2 Results Summary: {summary}")
    
    return {"h2": result_dicts, "summary": summary, "generate_cot": generate_cot}


def main():
    parser = argparse.ArgumentParser(description="Run hyperbolic reasoning experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["h1", "h1.5", "h2", "all"],
        default="all",
        help="Which experiment to run (h1.5 = taxonomy tree target)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepseek_7b", "qwen_7b", "all"],
        default="deepseek_7b",
        help="Which model to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--cached-activations", type=Path, default=None, help="Path to cached activations file (optional)")
    parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings for visualization")
    
    # Additional CLI args for hyperparameter control (override config values)
    parser.add_argument("--output-dim", type=int, default=None, help="Probe embedding dimension (overrides config)")
    parser.add_argument("--curvature", type=float, default=None, help="Hyperbolic curvature (overrides config)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to probe (overrides config)")
    parser.add_argument("--probes", type=str, nargs="+", default=None, 
                       choices=["euclidean", "hyperbolic", "lorentz"],
                       help="Probe types to run (overrides config)")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=["prontoqa", "binarytree", "listops", "ailuminate"],
                       help="Dataset to use (overrides config)")
    parser.add_argument("--h2-n-samples", type=int, default=None,
                       help="Number of samples for H2 experiment (overrides config)")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir=output_dir)
    logger = get_logger()
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    config["seed"] = args.seed
    
    # Apply CLI overrides to config
    if args.output_dim is not None:
        if "probe" not in config:
            config["probe"] = {}
        config["probe"]["embedding_dim"] = args.output_dim
        logger.info(f"CLI override: output_dim = {args.output_dim}")
    
    if args.curvature is not None:
        if "probe" not in config:
            config["probe"] = {}
        config["probe"]["curvature"] = args.curvature
        logger.info(f"CLI override: curvature = {args.curvature}")
    
    if args.layers is not None:
        config["layers"] = args.layers
        logger.info(f"CLI override: layers = {args.layers}")
    
    if args.probes is not None:
        config["probe_types"] = args.probes
        logger.info(f"CLI override: probe_types = {args.probes}")
    
    if args.dataset is not None:
        config["dataset"] = args.dataset
        logger.info(f"CLI override: dataset = {args.dataset}")

    if hasattr(args, 'h2_n_samples') and args.h2_n_samples is not None:
        config["h2_n_samples"] = args.h2_n_samples
        logger.info(f"CLI override: h2_n_samples = {args.h2_n_samples}")

    
    # Determine models to run
    if args.model == "all":
        models = ["deepseek_7b", "qwen_7b"]
    else:
        models = [args.model]
    
    # Run experiments
    all_results = {}
    
    for model_name in models:
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiments for {model_name}")
        logger.info(f"{'='*60}\n")
        
        model_results = {}
        
        if args.experiment in ["h1", "all"]:
            model_results.update(run_h1_experiment(model_name, config, model_dir, args.cached_activations))

        if args.experiment in ["h1.5", "all"]:
            model_results.update(run_h15_experiment(model_name, config, model_dir, args.cached_activations))

        if args.experiment in ["h2", "all"]:
            model_results.update(run_h2_experiment(model_name, config, model_dir))
        
        all_results[model_name] = model_results
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"all_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("\n" + "="*60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
