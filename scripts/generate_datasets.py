#!/usr/bin/env python3
"""
Dataset generation script.

Generates and saves canonical datasets for experiments.
Usage:
    python scripts/generate_datasets.py --dataset prontoqa
    python scripts/generate_datasets.py --dataset all
    python scripts/generate_datasets.py --dataset prontoqa --n-train 1500 --n-test 600  # smaller for debug
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prontoqa import generate_prontoqa_datasets
from src.data.listops import generate_listops_datasets
from src.data.ailuminate import generate_ailuminate_datasets
from src.utils.logging import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Generate datasets for experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["prontoqa", "listops", "ailuminate", "all"],
        default="prontoqa",
        help="Which dataset to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/data",
        help="Output directory for datasets",
    )
    parser.add_argument("--n-train", type=int, default=2500, help="Training samples (500/depth × 5)")
    parser.add_argument("--n-test", type=int, default=1000, help="Test samples (100/depth × 5 × 2 labels)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--depth-min", type=int, default=1, help="Minimum depth")
    parser.add_argument("--depth-max", type=int, default=5, help="Maximum depth")

    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir=args.output_dir, file_prefix="data_generation")
    logger = get_logger()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    depth_range = (args.depth_min, args.depth_max)
    
    if args.dataset in ["prontoqa", "all"]:
        logger.info("Generating PrOntoQA dataset...")
        train, test = generate_prontoqa_datasets(
            output_dir=output_dir,
            n_train=args.n_train,
            n_test_true=args.n_test // 2,
            n_test_false=args.n_test // 2,
            depth_range=depth_range,
            seed=args.seed,
        )
        logger.info(f"PrOntoQA train: {len(train)} samples")
        logger.info(f"PrOntoQA test: {len(test)} samples")
        logger.info(f"Train stats: {train.get_statistics()}")
        logger.info(f"Test stats: {test.get_statistics()}")
    
    if args.dataset in ["listops", "all"]:
        logger.info("Generating ListOps dataset...")
        train, test = generate_listops_datasets(
            output_dir=output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            depth_range=depth_range,
            seed=args.seed,
        )
        logger.info(f"ListOps train: {len(train)} samples")
        logger.info(f"ListOps test: {len(test)} samples")
        logger.info(f"Train stats: {train.get_statistics()}")
        logger.info(f"Test stats: {test.get_statistics()}")

    if args.dataset in ["ailuminate", "all"]:
        logger.info("Generating AILuminate (harm-taxonomy) dataset...")
        train, test = generate_ailuminate_datasets(
            output_dir=output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
        )
        logger.info(f"AILuminate train: {len(train)} samples")
        logger.info(f"AILuminate test: {len(test)} samples")
        logger.info(f"Train stats: {train.get_statistics()}")
        logger.info(f"Test stats: {test.get_statistics()}")

    logger.info(f"Datasets saved to: {output_dir}")


if __name__ == "__main__":
    main()
