#!/bin/bash
#SBATCH --job-name=ablation_sweep_round3
#SBATCH --output=logs/final/ablations/first_iter_sweep_round3.out
#SBATCH --error=logs/final/ablations/first_iter_sweep_round3.err
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
ABLATION_DIR=$IMPROVER_BASE/experiments/final/ablations/first_iter_sweep
CONFIG_DIR=$ABLATION_DIR/configs/base
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

# ============================================================
# PHASES 1-4: COMPLETED - Commented out
# ============================================================
# These phases have been completed successfully:
# - Phase 1: Base data generation
# - Phase 2: Training dataset generation
# - Phase 3A: SFT training (4 models) + merging
# - Phase 3B: wSFT LR sweep (4 models) + merging
# - Phase 3C: wSFT data sweep (3 models) + merging
# - Phase 4: All SFT/wSFT evaluations
#
# NOTE: Run run_missing_evals.sh if Phase 4 evaluations are incomplete
# ============================================================

# ============================================================
# PHASE 5: Select Best SFT and wSFT Models
# ============================================================
# echo "============================================================"
# echo "PHASE 5: Selecting Best Models"
# echo "============================================================"

# # Select best SFT model
# echo "Selecting best SFT model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     SFT_lr1e-06_test SFT_lr5e-06_test SFT_lr1e-05_test SFT_lr2e-05_test \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_sft_model

# BEST_SFT_MODEL=$(cat $ABLATION_DIR/.best_sft_model)
# echo "Best SFT Model: $BEST_SFT_MODEL"

# # Select best wSFT model (combining LR and data sweeps)
# echo "Selecting best wSFT model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     wSFT_lr1e-06_test wSFT_lr5e-06_test wSFT_lr1e-05_test wSFT_lr2e-05_test \
#     wSFT_vt0.5_test wSFT_vt0.8_test wSFT_vt1.0_test \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_wsft_model

# BEST_WSFT_MODEL=$(cat $ABLATION_DIR/.best_wsft_model)
# echo "Best wSFT Model: $BEST_WSFT_MODEL"

# ============================================================
# PHASE 6A: IRPO Training Sweeps (using best wSFT model)
# ============================================================
echo "============================================================"
echo "PHASE 6A: IRPO Training Sweeps"
echo "============================================================"

# Update IRPO base config to use best wSFT model
# Note: You may need to manually update the base config or use a template approach
# echo "Using best wSFT model for IRPO: $BEST_WSFT_MODEL"

# # Update the base config's base_model and ref_model
# sed -i "s|base_model:.*|base_model: $BEST_WSFT_MODEL|g" $CONFIG_DIR/IRPO_base.yaml
# sed -i "s|ref_model:.*|ref_model: $BEST_WSFT_MODEL|g" $CONFIG_DIR/IRPO_base.yaml

# # IRPO LR sweep
# echo "Running IRPO LR sweep..."
# axolotl train $CONFIG_DIR/IRPO_base.yaml \
#     --sweep $SWEEP_DIR/IRPO_lr.yaml

# IRPO parameter sweep (beta, alpha, W/L, gap)
echo "Running IRPO w/l sweep..."
axolotl train $CONFIG_DIR/IRPO_base.yaml \
    --sweep $SWEEP_DIR/IRPO_wl.yaml

# echo "✓ IRPO sweeps completed"

# # ============================================================
# # PHASE 6B: DPO Training Sweeps (using best wSFT model)
# # ============================================================
# echo "============================================================"
# echo "PHASE 6B: DPO Training Sweeps"
# echo "============================================================"

# # Update DPO base config to use best wSFT model
# sed -i "s|base_model:.*|base_model: $BEST_WSFT_MODEL|g" $CONFIG_DIR/DPO_base.yaml
# sed -i "s|ref_model:.*|ref_model: $BEST_WSFT_MODEL|g" $CONFIG_DIR/DPO_base.yaml

# echo "Running DPO parameter sweep..."
# axolotl train $CONFIG_DIR/DPO_base.yaml \
#     --sweep $SWEEP_DIR/DPO_params.yaml

# echo "✓ DPO sweeps completed"

# ============================================================
# PHASE 7: Evaluate IRPO and DPO Models
# ============================================================
echo "============================================================"
echo "PHASE 7: Evaluating IRPO and DPO Models"
echo "============================================================"

# Evaluate IRPO LR sweep
# for lr in 1e-06 5e-06 1e-05 2e-05; do
#     echo "Evaluating IRPO_lr${lr}..."
#     ./improver run pipeline \
#         --run_id IRPO_lr${lr}_test \
#         --annotation --informal --examples 4 \
#         --metric $METRIC --prompt_id final_test \
#         --split test --model ${MODELS_DIR}/IRPO_lr${lr} \
#         --num_blocks 16 \
#         --config experiments/final/test_eval.yaml
# done

# Evaluate IRPO parameter sweep models
# Beta/Alpha combinations
# beta=0.05
# alpha=1.0
# echo "Evaluating old broken one!"
# echo "Evaluating IRPO_beta${beta}_alpha${alpha}..."
#         ./improver run pipeline \
#             --run_id IRPO_beta${beta}_alpha${alpha}_test \
#             --annotation --informal --examples 4 \
#             --metric $METRIC --prompt_id final_test \
#             --split test --model ${MODELS_DIR}/IRPO_beta${beta}_alpha${alpha} \
#             --num_blocks 16 \
#             --config experiments/final/test_eval.yaml
# echo "Finished old broken one!"

# for beta in 0.02 0.05 0.1; do
# for beta in 0.1; do
#     for alpha in 0.2 0.5 1.0; do
#         echo "Evaluating IRPO_beta${beta}_alpha${alpha}..."
#         ./improver run pipeline \
#             --run_id IRPO_beta${beta}_alpha${alpha}_test \
#             --annotation --informal --examples 4 \
#             --metric $METRIC --prompt_id final_test \
#             --split test --model ${MODELS_DIR}/IRPO_beta${beta}_alpha${alpha} \
#             --num_blocks 16 \
#             --config experiments/final/test_eval.yaml
#     done
# done

# W/L pairs
# for w in 1 2 4; do
for w in 1 2 4; do
    for l in 1 2 4; do
        echo "Evaluating IRPO_w${w}_l${l}..."
        ./improver run pipeline \
            --run_id IRPO_w${w}_l${l}_test \
            --annotation --informal --examples 4 \
            --metric $METRIC --prompt_id final_test \
            --split test --model ${MODELS_DIR}/IRPO_w${w}_l${l} \
            --num_blocks 16 \
            --config experiments/final/test_eval.yaml
    done
done

# # Min gap
# for gap in 0 1 2; do
#     echo "Evaluating IRPO_gap${gap}..."
#     ./improver run pipeline \
#         --run_id IRPO_gap${gap}_test \
#         --annotation --informal --examples 4 \
#         --metric $METRIC --prompt_id final_test \
#         --split test --model ${MODELS_DIR}/IRPO_gap${gap} \
#         --num_blocks 16 \
#         --config experiments/final/test_eval.yaml
# done

# # Evaluate DPO models
# for beta in 0.02 0.05 0.1; do
#     echo "Evaluating DPO_beta${beta}..."
#     ./improver run pipeline \
#         --run_id DPO_beta${beta}_test \
#         --annotation --informal --examples 4 \
#         --metric $METRIC --prompt_id final_test \
#         --split test --model ${MODELS_DIR}/DPO_beta${beta} \
#         --num_blocks 16 \
#         --config experiments/final/test_eval.yaml
# done

echo "✓ All IRPO/DPO evaluations completed"

# ============================================================
# PHASE 8: Select Best IRPO/DPO Models
# ============================================================
echo "============================================================"
echo "PHASE 8: Final Model Selection"
echo "============================================================"

# Collect all IRPO run IDs
IRPO_RUN_IDS=""
# for lr in 1e-06 5e-06 1e-05 2e-05; do
#     IRPO_RUN_IDS="$IRPO_RUN_IDS IRPO_lr${lr}_test"
# done
# for beta in 0.02 0.05 0.1; do
#     for alpha in 0.2 0.5 1.0; do
#         IRPO_RUN_IDS="$IRPO_RUN_IDS IRPO_beta${beta}_alpha${alpha}_test"
#     done
# done
for w in 1 2 4; do
    for l in 1 2 4; do
        IRPO_RUN_IDS="$IRPO_RUN_IDS IRPO_w${w}_l${l}_test"
    done
done
# for gap in 0 1 2; do
#     IRPO_RUN_IDS="$IRPO_RUN_IDS IRPO_gap${gap}_test"
# done

echo "Selecting best IRPO model..."
python $SCRIPTS_DIR/select_best_model.py \
    $IRPO_RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_irpo_model

BEST_IRPO_MODEL=$(cat $ABLATION_DIR/.best_irpo_model)
echo "Best IRPO Model: $BEST_IRPO_MODEL"

# Select best DPO model
# DPO_RUN_IDS="DPO_beta0.02_test DPO_beta0.05_test DPO_beta0.1_test"
# echo "Selecting best DPO model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     $DPO_RUN_IDS \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_dpo_model

# BEST_DPO_MODEL=$(cat $ABLATION_DIR/.best_dpo_model)
# echo "Best DPO Model: $BEST_DPO_MODEL"

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "ABLATION STUDY COMPLETE"
echo "============================================================"
echo ""
echo "Best Models:"
echo "  SFT:  $BEST_SFT_MODEL"
echo "  wSFT: $BEST_WSFT_MODEL"
echo "  IRPO: $BEST_IRPO_MODEL"
echo "  DPO:  $BEST_DPO_MODEL"
echo ""
echo "Results saved to:"
echo "  - Configs: $CONFIG_DIR"
echo "  - Sweeps: $SWEEP_DIR"
echo "  - Data: $DATA_DIR"
echo "  - Models: $MODELS_DIR"
echo "  - Evaluations: $EVALS_DIR"
echo ""
echo "To use these models for the next iteration:"
echo "  1. Use the best wSFT model as the base for further IRPO/DPO training"
echo "  2. Use the best IRPO/DPO model for the next iteration's pipeline"
echo "============================================================"
