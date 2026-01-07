#!/bin/bash
#SBATCH --job-name=second_iter_sweep
#SBATCH --output=logs/final/ablations/second_iter_sweep.out
#SBATCH --error=logs/final/ablations/second_iter_sweep.err
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=250G

# ============================================================
# Second Iteration Ablation Study
# ============================================================
# This script orchestrates the second iteration ablation sweep:
# 1. Run inference on training set with best IRPO model from iter 1
# 2. Generate training datasets with different replay buffer configs
# 3. Train models with different base models and replay buffers
# 4. Evaluate all trained models
# 5. Select best model
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

# ============ Path Configuration ============
IMPROVER_BASE=/home/riyaza/eval_improver/improver
ABLATION_DIR=$IMPROVER_BASE/experiments/final/ablations/second_iter_sweep
CONFIG_DIR=$ABLATION_DIR/configs/base
SWEEP_DIR=$ABLATION_DIR/sweeps
DATA_DIR=$ABLATION_DIR/data
SCRIPTS_DIR=$ABLATION_DIR/scripts
MODELS_DIR=/data/user_data/riyaza/saved_models/ablations/second_iter
EVALS_DIR=$IMPROVER_BASE/evals

# Model and run configuration
ITER1_BEST_MODEL=/data/user_data/riyaza/saved_models/ablations/IRPO_w4_l4
METRIC="length"
SECOND_ITER_RUN_ID="second_iter_base_train"

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5

# ============================================================
# PHASE 1: Run Inference on Training Set with Best Model
# ============================================================
echo "============================================================"
echo "PHASE 1: Running Inference with Best IRPO Model from Iter 1"
echo "============================================================"

echo "Running inference with model: $ITER1_BEST_MODEL"
# ./improver run pipeline \
#     --run_id $SECOND_ITER_RUN_ID \
#     --annotation --informal --examples 4 \
#     --metric $METRIC --prompt_id final_train \
#     --split train --model $ITER1_BEST_MODEL \
#     --num_blocks 512 \
#     --config experiments/final/test_eval.yaml

echo "✓ Inference completed: run_id=$SECOND_ITER_RUN_ID"

# ============================================================
# PHASE 2: Generate Training Datasets
# ============================================================
echo "============================================================"
echo "PHASE 2: Generating Training Datasets"
echo "============================================================"

echo "Running data generation script..."
# bash $SCRIPTS_DIR/generate_data.sh

echo "✓ All training datasets generated"

# ============================================================
# PHASE 3: Train Models with Axolotl Sweep
# ============================================================
echo "============================================================"
echo "PHASE 3: Training Models (8 configurations)"
echo "============================================================"

echo "Running Axolotl sweep..."
echo "  - 2 base models (DeepSeek-7B, IRPO_w4_l4)"
echo "  - 4 replay configs (none, replace@0.2, replace@0.4, replace@0.6)"
echo "  - Total: 8 training runs"

# axolotl train $CONFIG_DIR/IRPO_deepseek_base.yaml \
#     --sweep $SWEEP_DIR/IRPO_sweep.yaml

echo "✓ All models trained successfully"

# ============================================================
# PHASE 4: Evaluate All Models
# ============================================================
echo "============================================================"
echo "PHASE 4: Evaluating All Trained Models"
echo "============================================================"

# Model names based on sweep configuration

base="deepseek"
replay="rep0.6"
MODEL_NAME="${base}_${replay}"
echo "Evaluating ${MODEL_NAME}..."

./improver run pipeline \
    --run_id ${MODEL_NAME}_test \
    --annotation --informal --examples 4 \
    --metric $METRIC --prompt_id final_test \
    --split test --model ${MODELS_DIR}/${MODEL_NAME} \
    --num_blocks 16 \
    --config experiments/final/test_eval.yaml

BASE_MODELS=("iter1")
REPLAY_CONFIGS=("norep" "rep0.2" "rep0.4" "rep0.6")

for base in "${BASE_MODELS[@]}"; do
    for replay in "${REPLAY_CONFIGS[@]}"; do
        MODEL_NAME="${base}_${replay}"
        echo "Evaluating ${MODEL_NAME}..."

        ./improver run pipeline \
            --run_id ${MODEL_NAME}_test \
            --annotation --informal --examples 4 \
            --metric $METRIC --prompt_id final_test \
            --split test --model ${MODELS_DIR}/${MODEL_NAME} \
            --num_blocks 16 \
            --config experiments/final/test_eval.yaml
    done
done

echo "✓ All model evaluations completed"

# ============================================================
# PHASE 5: Select Best Model
# ============================================================
echo "============================================================"
echo "PHASE 5: Selecting Best Model"
echo "============================================================"

# Collect all run IDs
RUN_IDS=""
for base in "${BASE_MODELS[@]}"; do
    for replay in "${REPLAY_CONFIGS[@]}"; do
        MODEL_NAME="${base}_${replay}"
        RUN_IDS="$RUN_IDS ${MODEL_NAME}_test"
    done
done

echo "Selecting best model from all runs..."
python $SCRIPTS_DIR/select_best_model.py \
    $RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_second_iter_model

BEST_MODEL=$(cat $ABLATION_DIR/.best_second_iter_model)
echo ""
echo "============================================================"
echo "SECOND ITERATION SWEEP COMPLETE!"
echo "============================================================"
echo "Best Model: $BEST_MODEL"
echo ""
echo "Results saved to:"
echo "  - Model path: $ABLATION_DIR/.best_second_iter_model"
echo "  - Evaluations: $EVALS_DIR/[run_id]/analysis/BoN/"
echo "  - Trained models: $MODELS_DIR/"
echo "============================================================"
