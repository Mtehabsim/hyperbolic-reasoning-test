"""CONTROL: synthetic activations that DO encode the AILuminate taxonomy.

Purpose: isolate "is H1.5's machinery capable of recovering a taxonomy when the
signal is present?" from "the placeholder prompts carried no signal". The DGX
mock extractor emits pure random noise (ignores prompt text), so it can never
test recovery. Here we instead PLANT the taxonomy structure directly into the
activations -- each sample's hidden state gets a branch component (from its family)
and a leaf component (from its hazard) plus noise -- then save in the fork's cached
format and run H1.5 on it.

Expected outcome if the pipeline is correct:
  H1.5 (taxonomy target) recovers the tree -> rho -> ~1 (esp. hyperbolic).
If it stays ~0 here, the bug is in the probe/target code, NOT the prompts.

Run:
  python scripts/make_taxonomy_control_activations.py \
    --out outputs/activations/control_taxonomy.pt --data-out outputs/data/ailuminate
  python run_experiments.py --experiment h1.5 --model deepseek_7b --dataset ailuminate \
    --cached-activations outputs/activations/control_taxonomy.pt \
    --output-dim 5 --curvature 0.5 --layers 8 12 16 19 21 23 25 27 \
    --probes euclidean hyperbolic --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.ailuminate import generate_ailuminate_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/activations/control_taxonomy.pt"))
    ap.add_argument("--data-out", type=Path, default=Path("outputs/data"))
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--d-model", type=int, default=3584)
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--branch-strength", type=float, default=3.0,
                    help="how strongly the FAMILY is encoded (bigger = easier tree)")
    ap.add_argument("--leaf-strength", type=float, default=1.0,
                    help="how strongly the HAZARD leaf is encoded")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1. Build the SAME ailuminate test set the experiment will load (so label_paths match).
    _, test = generate_ailuminate_datasets(args.data_out, n_test=args.n_test,
                                           n_train=args.n_test, seed=args.seed)
    samples = test.samples
    n = len(samples)
    label_paths = [list(s.metadata["label_path"]) for s in samples]     # [family_idx, hazard_idx]
    families = np.array([p[0] for p in label_paths])
    leaves = np.array([p[1] for p in label_paths])
    n_fam = int(families.max()) + 1
    n_leaf = int(leaves.max()) + 1
    d = args.d_model

    # 2. Fixed random direction per family and per leaf (shared across samples of that node).
    fam_dirs = rng.standard_normal((n_fam, d))
    leaf_dirs = rng.standard_normal((n_leaf, d))

    # 3. Per-sample hidden state = branch signal + leaf signal + noise, planted so the
    #    taxonomy tree is genuinely present. Same vector repeated across seq positions
    #    (mean-pool over seq recovers it). Done per layer (identical structure each layer).
    activations = {}
    for layer in range(args.n_layers):
        base = (args.branch_strength * fam_dirs[families]
                + args.leaf_strength * leaf_dirs[leaves]
                + args.noise * rng.standard_normal((n, d)))
        # (n, seq, d): broadcast the per-sample vector across seq, with small per-token jitter
        seq = np.repeat(base[:, None, :], args.seq_len, axis=1)
        seq = seq + 0.1 * rng.standard_normal((n, args.seq_len, d))
        activations[layer] = torch.tensor(seq, dtype=torch.float32)

    cached = {
        "activations": activations,
        "tokens": [[f"tok{i}" for i in range(args.seq_len)] for _ in range(n)],
        "attention": {},
        "metadata": {"synthetic_control": True, "planted_taxonomy": True,
                     "branch_strength": args.branch_strength,
                     "leaf_strength": args.leaf_strength, "noise": args.noise},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, args.out)
    print(f"wrote planted-taxonomy activations: {args.out}  "
          f"(n={n}, families={n_fam}, leaves={n_leaf}, layers={args.n_layers})")
    print(f"matching dataset at: {args.data_out}/ailuminate_test.json")
    print("Now run H1.5 on --cached-activations", args.out,
          "-> rho should approach ~1 if the pipeline is sound.")


if __name__ == "__main__":
    main()
