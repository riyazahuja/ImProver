#!/bin/bash
#SBATCH --job-name=second_iter_sweep_v2
#SBATCH --output=logs/final/ablations/second_iter_sweep_v2.out
#SBATCH --error=logs/final/ablations/second_iter_sweep_v2.err
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=250G

# ============================================================
# Second Iteration Ablation Study (with wSFT)
# ============================================================
# Pipeline: Base Model → wSFT → IRPO
#
# This script orchestrates the complete second iteration:
# 1. Run inference with best IRPO from iter 1
# 2. Generate wSFT training datasets (4 replay configs)
# 3. Train 8 wSFT models (2 base models × 4 replay configs)
# 4. Merge LoRA adapters for wSFT models
# 5. Evaluate wSFT models
# 6. Select best wSFT (informational)
# 7. Run inference + generate IRPO data for each wSFT
# 8. Train 8 IRPO models (one for each wSFT)
# 9. Evaluate IRPO models
# 10. Select best IRPO model
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

# Model combinations
BASE_MODELS=("deepseek" "iter1")
REPLAY_CONFIGS=("norep" "rep0.2" "rep0.4" "rep0.6")

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5

# ============================================================
# PHASE 1: Inference with Best Iter 1 Model
# ============================================================
echo "============================================================"
echo "PHASE 1: Running Inference with Best IRPO from Iter 1"
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
# PHASE 2: Generate wSFT Training Datasets
# ============================================================
echo "============================================================"
echo "PHASE 2: Generating wSFT Training Datasets"
echo "============================================================"

echo "Running wSFT data generation script..."
bash $SCRIPTS_DIR/generate_wsft_data.sh

echo "✓ All wSFT datasets generated and preprocessed"

echo "Running IRPO data generation script..."
bash $SCRIPTS_DIR/generate_data.sh

echo "✓ All IRPO datasets generated and preprocessed"

# ============================================================
# PHASE 3: Train wSFT Models
# ============================================================
echo "============================================================"
echo "PHASE 3: Training wSFT Models (8 configurations)"
echo "============================================================"

echo "Running Axolotl wSFT sweep..."
echo "  - 2 base models (DeepSeek-7B, IRPO_w4_l4)"
echo "  - 4 replay configs (none, 20%, 40%, 60%)"
echo "  - Total: 8 wSFT training runs"

axolotl train $CONFIG_DIR/wSFT_deepseek_base.yaml \
    --sweep $SWEEP_DIR/wSFT_sweep.yaml

echo "✓ All wSFT models trained successfully"

# ============================================================
# PHASE 4: Merge LoRA Adapters for wSFT Models
# ============================================================
echo "============================================================"
echo "PHASE 4: Merging LoRA Adapters for wSFT Models"
echo "============================================================"

for base in "${BASE_MODELS[@]}"; do
    for replay in "${REPLAY_CONFIGS[@]}"; do
        MODEL_NAME="wSFT_${base}_${replay}"
        echo "Merging ${MODEL_NAME}..."

        # Determine reference model
        if [ "$base" == "deepseek" ]; then
            REF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        else
            REF_MODEL=$ITER1_BEST_MODEL
        fi

        python $IMPROVER_BASE/experiments/final/merge.py \
            --ref $REF_MODEL \
            --adapter ${MODELS_DIR}/${MODEL_NAME}_lora \
            --output ${MODELS_DIR}/${MODEL_NAME}
        
        rm -rf ${MODELS_DIR}/${MODEL_NAME}_lora
    done
done

echo "✓ All wSFT LoRA adapters merged"

# ============================================================
# PHASE 5: Evaluate wSFT Models
# ============================================================
echo "============================================================"
echo "PHASE 5: Evaluating wSFT Models"
echo "============================================================"

for base in "${BASE_MODELS[@]}"; do
    for replay in "${REPLAY_CONFIGS[@]}"; do
        MODEL_NAME="wSFT_${base}_${replay}"
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

echo "✓ All wSFT model evaluations completed"

# ============================================================
# PHASE 6: Select Best wSFT Model (Informational)
# ============================================================
echo "============================================================"
echo "PHASE 6: Selecting Best wSFT Model (Informational)"
echo "============================================================"

WSFT_RUN_IDS=""
for base in "${BASE_MODELS[@]}"; do
    for replay in "${REPLAY_CONFIGS[@]}"; do
        WSFT_RUN_IDS="$WSFT_RUN_IDS wSFT_${base}_${replay}_test"
    done
done

python $SCRIPTS_DIR/select_best_model.py \
    $WSFT_RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_wsft_model

BEST_WSFT=$(cat $ABLATION_DIR/.best_wsft_model)
echo "Best wSFT Model: $BEST_WSFT"

# ============================================================
# PHASE 7: Generate IRPO Data for Each wSFT Model
# ============================================================
echo "============================================================"
echo "PHASE 7: Generating IRPO Training Data (8 runs)"
echo "============================================================"

echo "SKIPPING"
# for base in "${BASE_MODELS[@]}"; do
#     for replay in "${REPLAY_CONFIGS[@]}"; do
#         MODEL_NAME="wSFT_${base}_${replay}"
#         # RUN_ID="${MODEL_NAME}_train"

#         # echo "Running inference for ${MODEL_NAME}..."
#         # ./improver run pipeline \
#         #     --run_id $RUN_ID \
#         #     --annotation --informal --examples 4 \
#         #     --metric $METRIC --prompt_id final_train \
#         #     --split train --model ${MODELS_DIR}/${MODEL_NAME} \
#         #     --num_blocks 512 \
#         #     --config experiments/final/test_eval.yaml

#         echo "Generating IRPO data for ${MODEL_NAME}..."
#         ./improver run training_data \
#             --run_id $SECOND_ITER_RUN_ID \
#             --output_path ${DATA_DIR}/IRPO_${base}_${replay}.jsonl \
#             --type dpo \
#             --max_champions 4 \
#             --num_invalid 4 \
#             --min_gap 1 \
#             --filter_threshold 0.8 \
#             --prev_run_id base_length_train \
#             --replay_type replace \
#             --replay_buffer_split 0.2
#     done
# done

echo "✓ All IRPO datasets generated"

# ============================================================
# PHASE 8: Train IRPO Models
# ============================================================
echo "============================================================"
echo "PHASE 8: Training IRPO Models (8 configurations)"
echo "============================================================"

echo "Running Axolotl IRPO sweep..."
echo "  - 8 IRPO models (one for each wSFT model)"

axolotl train $CONFIG_DIR/IRPO_base.yaml \
    --sweep $SWEEP_DIR/IRPO_sweep.yaml

echo "✓ All IRPO models trained successfully"

# ============================================================
# PHASE 9: Evaluate IRPO Models
# ============================================================
echo "============================================================"
echo "PHASE 9: Evaluating IRPO Models"
echo "============================================================"

for base in "${BASE_MODELS[@]}"; do
    for replay in "${REPLAY_CONFIGS[@]}"; do
        MODEL_NAME="IRPO_${base}_${replay}"
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

echo "✓ All IRPO model evaluations completed"

# ============================================================
# PHASE 10: Select Best IRPO Model
# ============================================================
echo "============================================================"
echo "PHASE 10: Selecting Best IRPO Model"
echo "============================================================"

IRPO_RUN_IDS=""
for base in "${BASE_MODELS[@]}"; do
    for replay in "${REPLAY_CONFIGS[@]}"; do
        IRPO_RUN_IDS="$IRPO_RUN_IDS IRPO_${base}_${replay}_test"
    done
done

python $SCRIPTS_DIR/select_best_model.py \
    $IRPO_RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_second_iter_model

BEST_IRPO=$(cat $ABLATION_DIR/.best_second_iter_model)

echo ""
echo "============================================================"
echo "SECOND ITERATION SWEEP COMPLETE!"
echo "============================================================"
echo "Best wSFT Model:  $BEST_WSFT"
echo "Best IRPO Model:  $BEST_IRPO"
echo ""
echo "Results saved to:"
echo "  - Best wSFT:  $ABLATION_DIR/.best_wsft_model"
echo "  - Best IRPO:  $ABLATION_DIR/.best_second_iter_model"
echo "  - Evaluations: $EVALS_DIR/[run_id]/analysis/BoN/"
echo "  - Trained models: $MODELS_DIR/"
echo ""
echo "Total models trained: 16 (8 wSFT + 8 IRPO)"
echo "============================================================"
