#!/bin/bash
# =============================================================================
# PRIORITY EXPERIMENTS RUNNER
# =============================================================================
# Run missing experiments with proper logging.
# Execute from the project root directory
# 
# Priority order:
# 1. Binary Tree H1 with correct config (c=0.5, d=5) - BLOCKS PAPER
# 2. Layer anisotropy analysis - Quick, strengthens claims
# 3. Qwen Binary Tree H1 - Cross-model validation
# =============================================================================

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/logs"
RESULTS_DIR="outputs/results_${TIMESTAMP}"

# Activate environment
source venv/bin/activate

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "=============================================="
echo "EXPERIMENT RUN: $TIMESTAMP"
echo "=============================================="

# =============================================================================
# PRIORITY 1: Binary Tree H1 with Correct Config (MUST RUN)
# =============================================================================
# Problem: Current results used curvature=1.0, output_dim=16
# Fix: Use optimal c=0.5, d=5 from PrOntoQA ablation
# Expected: Lorentz should improve from ρ≈0.045 to ρ≈0.7-0.8

echo ""
echo "[PRIORITY 1] Binary Tree H1 - DeepSeek (c=0.5, d=5)"
echo "=================================================="

python run_experiments.py \
    --experiment h1 \
    --model deepseek_7b \
    --dataset binarytree \
    --cached-activations outputs/activations/deepseek_binarytree.pt \
    --output-dim 5 \
    --curvature 0.5 \
    --layers 8 12 16 19 21 23 25 27 \
    --probes euclidean hyperbolic lorentz \
    --output-dir "$RESULTS_DIR/h1_binarytree_deepseek" \
    2>&1 | tee "$LOG_DIR/h1_binarytree_deepseek_${TIMESTAMP}.log"

echo "[PRIORITY 1] COMPLETE. Results in: $RESULTS_DIR/h1_binarytree_deepseek"

# =============================================================================
# PRIORITY 2: Layer Anisotropy Analysis (Quick - 1hr)
# =============================================================================
# Computes: norms, singular values, participation ratio, isotropy
# Explains WHY Euclidean fails at late layers

echo ""
echo "[PRIORITY 2] Layer Anisotropy Analysis"
echo "======================================="

# DeepSeek PrOntoQA
python scripts/compute_layer_statistics.py \
    --cached-activations outputs/activations/deepseek_prontoqa.pt \
    --output "$RESULTS_DIR/layer_stats_deepseek_prontoqa.json" \
    --max-samples 1000 \
    2>&1 | tee "$LOG_DIR/layer_stats_deepseek_prontoqa_${TIMESTAMP}.log"

# DeepSeek Binary Tree
python scripts/compute_layer_statistics.py \
    --cached-activations outputs/activations/deepseek_binarytree.pt \
    --output "$RESULTS_DIR/layer_stats_deepseek_binarytree.json" \
    --max-samples 1000 \
    2>&1 | tee "$LOG_DIR/layer_stats_deepseek_binarytree_${TIMESTAMP}.log"

# Qwen PrOntoQA (control)
python scripts/compute_layer_statistics.py \
    --cached-activations outputs/activations/qwen_prontoqa.pt \
    --output "$RESULTS_DIR/layer_stats_qwen_prontoqa.json" \
    --max-samples 1000 \
    2>&1 | tee "$LOG_DIR/layer_stats_qwen_prontoqa_${TIMESTAMP}.log"

echo "[PRIORITY 2] COMPLETE. Statistics saved to: $RESULTS_DIR/layer_stats_*.json"

# =============================================================================
# PRIORITY 3: Qwen Binary Tree H1 (4-5hr)
# =============================================================================
# Control condition: Qwen should show NO late-layer collapse

echo ""
echo "[PRIORITY 3] Qwen Binary Tree H1 (Control)"
echo "==========================================="

python run_experiments.py \
    --experiment h1 \
    --model qwen_7b \
    --dataset binarytree \
    --cached-activations outputs/activations/qwen_binarytree.pt \
    --output-dim 5 \
    --curvature 0.5 \
    --layers 8 12 16 19 21 23 25 27 \
    --probes euclidean hyperbolic lorentz \
    --output-dir "$RESULTS_DIR/h1_binarytree_qwen" \
    2>&1 | tee "$LOG_DIR/h1_binarytree_qwen_${TIMESTAMP}.log"

echo "[PRIORITY 3] COMPLETE. Results in: $RESULTS_DIR/h1_binarytree_qwen"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Results directory: $RESULTS_DIR"
echo "Log directory: $LOG_DIR"
echo ""
echo "Files created:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "No JSON files found"
echo ""
echo "Next steps:"
echo "1. Review results in $RESULTS_DIR"
echo "2. Analyze with scripts/generate_figures.py"
