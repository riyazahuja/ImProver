#!/bin/bash
#SBATCH --job-name=ablation_sweep_round_vt
#SBATCH --output=logs/final/ablations/small_fixes_round_vt.out
#SBATCH --error=logs/final/ablations/small_fixes_round_vt.err
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=250G
#SBATCH --exclude=babel-p9-32

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
echo "PHASE 1: Train SFT with vt sweep"
echo "============================================================"
# axolotl train $CONFIG_DIR/SFT_vt.yaml \
#     --sweep $SWEEP_DIR/SFT_vt.yaml




echo "============================================================"
echo "PHASE 2: Evaluate SFT vt models"
echo "============================================================"
# vt=1.0
# type="top"

# echo "Merging SFT_vt${vt}_${type} model..."

# python /home/riyaza/eval_improver/improver/experiments/final/merge.py --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --adapter ${MODELS_DIR}/SFT_vt${vt}_${type}_lora --output ${MODELS_DIR}/SFT_vt${vt}_${type}

# ./improver run pipeline \
#             --run_id SFT_vt${vt}_${type}_test \
#             --annotation --informal --examples 4 \
#             --metric $METRIC --prompt_id final_test \
#             --split test --model ${MODELS_DIR}/SFT_vt${vt}_${type} \
#             --num_blocks 16 \
#             --config experiments/final/test_eval.yaml

# rm -rf ${MODELS_DIR}/SFT_vt${vt}_${type}_lora

# for vt in 0.5 0.8; do
#     for type in top bottom random; do

#         echo "Merging SFT_vt${vt}_${type} model..."

#         python /home/riyaza/eval_improver/improver/experiments/final/merge.py --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --adapter ${MODELS_DIR}/SFT_vt${vt}_${type}_lora --output ${MODELS_DIR}/SFT_vt${vt}_${type}

#         echo "Evaluating SFT_vt${vt}_${type}..."
#         ./improver run pipeline \
#             --run_id SFT_vt${vt}_${type}_test \
#             --annotation --informal --examples 4 \
#             --metric $METRIC --prompt_id final_test \
#             --split test --model ${MODELS_DIR}/SFT_vt${vt}_${type} \
#             --num_blocks 16 \
#             --config experiments/final/test_eval.yaml


#         rm -rf ${MODELS_DIR}/SFT_vt${vt}_${type}_lora
#     done
# done

echo "============================================================"
echo "PHASE 3: Train IRPO vt models"
echo "============================================================"
axolotl train $CONFIG_DIR/IRPO_base.yaml \
    --sweep $SWEEP_DIR/IRPO_gap.yaml

echo "============================================================"
echo "PHASE 4: Evaluate IRPO vt models"
echo "============================================================"

vt=1.0
type="top"
./improver run pipeline \
            --run_id IRPO_vt${vt}_${type}_test \
            --annotation --informal --examples 4 \
            --metric $METRIC --prompt_id final_test \
            --split test --model ${MODELS_DIR}/IRPO_vt${vt}_${type} \
            --num_blocks 16 \
            --config experiments/final/test_eval.yaml

for vt in 0.5 0.8; do
    for type in top bottom random; do
        echo "Evaluating IRPO_vt${vt}_${type}..."
        ./improver run pipeline \
            --run_id IRPO_vt${vt}_${type}_test \
            --annotation --informal --examples 4 \
            --metric $METRIC --prompt_id final_test \
            --split test --model ${MODELS_DIR}/IRPO_vt${vt}_${type} \
            --num_blocks 16 \
            --config experiments/final/test_eval.yaml
    done
done

