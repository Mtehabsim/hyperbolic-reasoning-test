# Hyperbolic Geometry of Reasoning: Probing LLM Hidden States

Code for the paper "Hyperbolic Geometry of Reasoning: Probing LLM Hidden States" (GRaM Workshop at ICLR 2026).

## Overview

This codebase implements hyperbolic and Euclidean structural probes to investigate the geometric structure of hierarchical reasoning in LLM hidden states. We probe reasoning-specialized (DeepSeek-R1-Distill-Qwen-7B) and standard instruction-tuned (Qwen2.5-7B-Instruct) models on PrOntoQA logical reasoning tasks, comparing probe geometries across layers.

**Key findings:**
- Hyperbolic probes maintain robust performance across all layers ($\rho \approx 0.97$)
- Euclidean probes exhibit late-layer degradation specific to reasoning-specialized models ($\rho = 0.49$ at L27)
- Thinking tokens concentrate hierarchical information at the compressed final layer ($\rho = 0.87$ vs $0.39$ for uniform pooling)

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+ and CUDA 12.1. A GPU with >=24GB VRAM is recommended for activation extraction; probing itself runs on CPU.

## Directory Structure

```
.
├── config/                  # Hydra configuration files
│   ├── config.yaml          # Main config
│   ├── dataset/             # Dataset configs (prontoqa, binary_tree, listops)
│   └── model/               # Model configs (deepseek_7b, qwen_7b, gpt2)
├── src/                     # Core source code
│   ├── data/                # Dataset generators (PrOntoQA, BinaryTree, ListOps)
│   ├── geometry/            # Euclidean and hyperbolic distance/metric implementations
│   ├── model/               # Model loading, activation hooks, token selection
│   ├── probes/              # Probe architectures and training
│   ├── experiments/         # Experiment runners (H1 layer-wise, H2 token ablation)
│   ├── analysis/            # Statistics and visualization
│   └── utils/               # Config, logging, reproducibility, W&B integration
├── scripts/                 # Standalone scripts for specific experiments
│   ├── extract_all_activations.py
│   ├── run_layer_ablation.py
│   ├── run_dimension_ablation.py
│   ├── compute_layer_statistics.py
│   └── generate_figures.py
├── tests/                   # Unit tests
├── run_experiments.py       # Main experiment runner
└── requirements.txt
```

## How to Run

### 1. Generate datasets

```bash
python scripts/generate_datasets.py --dataset prontoqa --num-hops 5 --num-samples 500
```

### 2. Extract activations (requires GPU)

```bash
python scripts/extract_all_activations.py \
    --model deepseek_7b \
    --dataset prontoqa \
    --output outputs/activations/deepseek_prontoqa.pt
```

### 3. Run layer-wise probing (H1)

```bash
python run_experiments.py \
    --experiment h1 \
    --model deepseek_7b \
    --dataset prontoqa \
    --cached-activations outputs/activations/deepseek_prontoqa.pt \
    --output-dim 5 \
    --curvature 0.5 \
    --layers 8 12 16 19 21 23 25 27 \
    --probes euclidean hyperbolic
```

### 4. Run token selection ablation (H2)

```bash
python run_experiments.py \
    --experiment h2 \
    --model deepseek_7b \
    --dataset prontoqa \
    --cached-activations outputs/activations/deepseek_prontoqa.pt \
    --layers 27
```

### 5. Compute layer statistics

```bash
python scripts/compute_layer_statistics.py \
    --cached-activations outputs/activations/deepseek_prontoqa.pt \
    --output outputs/results/layer_stats.json
```

### 6. Run dimension/curvature ablation

```bash
python scripts/run_dimension_ablation.py \
    --cached-activations outputs/activations/deepseek_prontoqa.pt \
    --layer 27 \
    --output outputs/results/dimension_ablation.json
```

## Reproducibility

All experiments use fixed seed 42. Hyperparameters: output dimension d=5, curvature c=0.5. See `config/config.yaml` for full training configuration.

## Tests

```bash
python -m pytest tests/ -v
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{raj2026hyperbolic,
  title={Hyperbolic Geometry of Reasoning: Probing {LLM} Hidden States},
  author={Arnav Raj},
  booktitle={ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling},
  year={2026},
  url={https://openreview.net/forum?id=JmWG0P9MDf}
}
```

## License

This project is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
