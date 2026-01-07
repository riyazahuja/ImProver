#!/bin/bash
#SBATCH --job-name=ablation_sweep_round3
#SBATCH --output=logs/final/ablations/small_fixes.out
#SBATCH --error=logs/final/ablations/small_fixes.err
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=250G
#SBATCH --exclude=babel-s5-24

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

# ============ Path Configuration ============
IMPROVER_BASE=/home/riyaza/eval_improver/improver
ABLATION_DIR=$IMPROVER_BASE/experiments/final/ablations/small_fixes
CONFIG_DIR=$ABLATION_DIR/configs
SWEEP_DIR=$ABLATION_DIR/sweeps
DATA_DIR=$ABLATION_DIR/data
SCRIPTS_DIR=$ABLATION_DIR/scripts
MODELS_DIR=/data/user_data/riyaza/saved_models/ablations
EVALS_DIR=$IMPROVER_BASE/evals

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
METRIC="length"
BASE_RUN_ID="base_${METRIC}_train"

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5


echo "============================================================"
echo "Phase 0: Generating Base Data with Annotations"
echo "============================================================"

# ./improver run pipeline \
#             --run_id SFT_length_train \
#             --annotation --informal --examples 4 \
#             --metric $METRIC --prompt_id final_train \
#             --split train --model /data/user_data/riyaza/saved_models/ablations/SFT_vt1.0_top \
#             --num_blocks 64 \
#             --config experiments/final/train_eval.yaml



./improver run training_data --run_id SFT_length_train --output_path /home/riyaza/eval_improver/improver/experiments/final/ablations/small_fixes/data/SFT_length_train.jsonl --type dpo --num_invalid 1 --max_champions 2 --min_gap 0 --prev_run_id base_length_train --replay_type replace --replay_buffer_split 0.2



echo "============================================================"
echo "PHASE 1: Training IRPO and DPO with alpha/beta sweep"
echo "============================================================"






axolotl train $CONFIG_DIR/IRPO_raw.yaml \
    --sweep $SWEEP_DIR/IRPO_and_DPO_alpha_beta.yaml



echo "============================================================"
echo "PHASE 2: Evaluating DPO Models"
echo "============================================================"



for beta in 0.02 0.05 0.1; do
    echo "Evaluating DPO_beta${beta}..."
    ./improver run pipeline \
        --run_id DPO_beta${beta}_SFT0.2_test \
        --annotation --informal --examples 4 \
        --metric $METRIC --prompt_id final_test \
        --split test --model ${MODELS_DIR}/DPO_beta${beta} \
        --num_blocks 16 \
        --config experiments/final/test_eval.yaml
done

echo "============================================================"
echo "PHASE 2: Evaluating IRPO Models"
echo "============================================================"

# Evaluate IRPO parameter sweep models
# Beta/Alpha combinations


for beta in 0.02 0.05 0.1; do
# for beta in 0.05 0.1; do
    for alpha in 0.2 0.5 1.0; do
        echo "Evaluating IRPO_beta${beta}_alpha${alpha}..."
        ./improver run pipeline \
            --run_id IRPO_beta${beta}_alpha${alpha}_SFT0.2_test \
            --annotation --informal --examples 4 \
            --metric $METRIC --prompt_id final_test \
            --split test --model ${MODELS_DIR}/IRPO_beta${beta}_alpha${alpha} \
            --num_blocks 16 \
            --config experiments/final/test_eval.yaml
    done
done

echo "============================================================"
echo "COMPLETE"
echo "============================================================"