#!/bin/bash
#SBATCH --job-name=missing_evals
#SBATCH --output=logs/final/ablations/missing_evals.out
#SBATCH --error=logs/final/ablations/missing_evals.err
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=250G


# ============================================================
# Comprehensive Ablation Study for First Iteration Training
# ============================================================
# This script orchestrates:
# 1. Initial pipeline run to generate base data
# 2. Data generation with different parameters
# 3. Training with hyperparameter sweeps using Axolotl
# 4. Evaluation of all trained models
# 5. Automatic selection of best models
# ============================================================

set -e  # Exit on error

# ============ Environment Setup ============
source $HOME/miniconda3/bin/activate env
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export RAY_TMPDIR="/data/user_data/riyaza/ray_tmp"
export HF_HOME="/data/user_data/riyaza/HF"

export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

set -e  # Exit on error

# Configuration
METRIC=length
MODELS_DIR=/data/user_data/riyaza/saved_models/ablations

echo "============================================================"
echo "Running Missing Evaluations"
echo "============================================================"
echo ""

# wSFT_lr2e-05 evaluation (the one that failed)
echo "1/4: Evaluating wSFT_lr2e-05..."
./improver run pipeline \
    --run_id wSFT_lr2e-05_test \
    --annotation --informal --examples 4 \
    --metric $METRIC --prompt_id final_test \
    --split test --model ${MODELS_DIR}/wSFT_lr2e-05 \
    --num_blocks 16 \
    --config experiments/final/test_eval.yaml

echo "✓ wSFT_lr2e-05 evaluation completed"
echo ""

# wSFT_vt0.5 evaluation
echo "2/4: Evaluating wSFT_vt0.5..."
./improver run pipeline \
    --run_id wSFT_vt0.5_test \
    --annotation --informal --examples 4 \
    --metric $METRIC --prompt_id final_test \
    --split test --model ${MODELS_DIR}/wSFT_vt0.5 \
    --num_blocks 16 \
    --config experiments/final/test_eval.yaml

echo "✓ wSFT_vt0.5 evaluation completed"
echo ""

# wSFT_vt0.8 evaluation
echo "3/4: Evaluating wSFT_vt0.8..."
./improver run pipeline \
    --run_id wSFT_vt0.8_test \
    --annotation --informal --examples 4 \
    --metric $METRIC --prompt_id final_test \
    --split test --model ${MODELS_DIR}/wSFT_vt0.8 \
    --num_blocks 16 \
    --config experiments/final/test_eval.yaml

echo "✓ wSFT_vt0.8 evaluation completed"
echo ""

# wSFT_vt1.0 evaluation
echo "4/4: Evaluating wSFT_vt1.0..."
./improver run pipeline \
    --run_id wSFT_vt1.0_test \
    --annotation --informal --examples 4 \
    --metric $METRIC --prompt_id final_test \
    --split test --model ${MODELS_DIR}/wSFT_vt1.0 \
    --num_blocks 16 \
    --config experiments/final/test_eval.yaml

echo "✓ wSFT_vt1.0 evaluation completed"
echo ""

echo "============================================================"
echo "✓ All missing evaluations completed successfully!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Review the evaluation results in evals/ directory"
echo "2. Run the updated first_iter_sweep.sh script to complete remaining phases (IRPO/DPO)"
