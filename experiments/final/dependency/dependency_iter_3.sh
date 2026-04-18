#!/bin/bash
#SBATCH --job-name=dependency_iter_3
#SBATCH --output=logs/final/dependency/dependency_iter_3.out
#SBATCH --error=logs/final/dependency/dependency_iter_3.err
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100G

set -euo pipefail

source "$HOME/miniconda3/bin/activate" env

export HF_HOME="/data/user_data/$USER/HF"
export RAY_TMPDIR="/data/user_data/$USER/ray_tmp"
mkdir -p "$RAY_TMPDIR" logs/final/dependency

export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

IMPROVER_BASE="/home/$USER/eval_improver/improver"
cd "$IMPROVER_BASE"

METRIC="dependency"
BASE_MODEL="/data/user_data/riyaza/saved_models/IRPO_dependency_iter_2"
MODEL_NAME="IRPO_${METRIC}_iter_3"
OUTPUT_DIR="/data/user_data/riyaza/saved_models/${MODEL_NAME}_NEW"

RUN_ID="IRPO_${METRIC}_iter_2_train"
DATA_DIR="$IMPROVER_BASE/experiments/final/dependency/data"
CONFIG_DIR="$IMPROVER_BASE/experiments/final/dependency/iter_3/configs"
DATASET_PATH="$DATA_DIR/IRPO_${METRIC}_iter_3.jsonl"
CONFIG_PATH="$CONFIG_DIR/$MODEL_NAME.yaml"
EVAL_CONFIG="$IMPROVER_BASE/experiments/final/test_eval.yaml"

mkdir -p "$DATA_DIR" "$CONFIG_DIR"

PREV_RUN_IDS=""
for prev_run_id in base_dependency_train IRPO_dependency_iter_1_train; do
    if [ -f "evals/$prev_run_id/analysis/BoN/training_data.json" ]; then
        if [ -z "$PREV_RUN_IDS" ]; then
            PREV_RUN_IDS="$prev_run_id"
        else
            PREV_RUN_IDS="$PREV_RUN_IDS,$prev_run_id"
        fi
    else
        echo "Skipping missing replay run: $prev_run_id"
    fi
done

REPLAY_ARGS=()
if [ -n "$PREV_RUN_IDS" ]; then
    REPLAY_ARGS=(
        --replay_buffer_split 0.4
        --replay_type replace
        --prev_run_id "$PREV_RUN_IDS"
    )
fi

echo "Building ImProver..."
lake build eval_improver
sleep 5

# echo "Evaluating iter 2 dependency model on train split..."
# ./improver run pipeline \
#     --run_id "$RUN_ID" \
#     --annotation \
#     --context 10 \
#     --informal \
#     --examples 4 \
#     --metric "$METRIC" \
#     --prompt_id "$IMPROVER_BASE/prompts/final_train_v2" \
#     --split train \
#     --model "$BASE_MODEL" \
#     --num_blocks 512 \
#     --config "$EVAL_CONFIG"
echo "Evaluating iter 2 dependency model on train split..."
./improver run eval \
    --run_id "$RUN_ID" \
    --cpus 50 \
    --config "$EVAL_CONFIG"

./improver run analysis \
    --run_id "$RUN_ID" \
    --config "$EVAL_CONFIG"

echo "Generating DPO data for iter 3..."
./improver run training_data \
    --run_id "$RUN_ID" \
    --output_path "$DATASET_PATH" \
    --type dpo \
    --num_invalid -1 \
    --max_champions -1 \
    --filter_threshold 1.1 \
    "${REPLAY_ARGS[@]}"

wc -l "$DATASET_PATH"

echo "Creating Axolotl config for $MODEL_NAME..."
python experiments/final/dependency/make_train_config.py \
    --base-model "$BASE_MODEL" \
    --learning-rate 5e-6 \
    --alpha 1.0 \
    --beta 0.05 \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --project-name "$MODEL_NAME" \
    --config-path "$CONFIG_PATH"

echo "Training $MODEL_NAME..."
accelerate launch -m axolotl.cli.train "$CONFIG_PATH"

echo "Evaluating $MODEL_NAME on test split..."
./improver run pipeline \
    --run_id "${MODEL_NAME}_test_NEW" \
    --annotation \
    --context 10 \
    --informal \
    --examples 4 \
    --metric "$METRIC" \
    --prompt_id "$IMPROVER_BASE/prompts/final_test" \
    --split test \
    --model "$OUTPUT_DIR" \
    --num_blocks 64 \
    --config "$EVAL_CONFIG"
