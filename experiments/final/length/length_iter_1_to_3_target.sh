#!/bin/bash
#SBATCH --job-name=length_1_to_3_target2
#SBATCH --output=logs/final/length/length_iter_1_to_3_target3.out
#SBATCH --error=logs/final/length/length_iter_1_to_3_target3.err
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:6
#SBATCH --time=5:00:00
#SBATCH --mem=200G

set -euo pipefail

source "$HOME/miniconda3/bin/activate" env
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export RAY_TMPDIR=/data/user_data/riyaza/ray_tmp
export HF_HOME=/data/user_data/riyaza/HF

export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

export TMPDIR=/data/user_data/riyaza/tmp
export TEMP=/data/user_data/riyaza/tmp
export TMP=/data/user_data/riyaza/tmp
export TRITON_CACHE_DIR=/data/user_data/riyaza/triton_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

IMPROVER_BASE=/home/riyaza/eval_improver/improver
METRIC=length
DATA_DIR="$IMPROVER_BASE/experiments/final/$METRIC/data"
CONFIG_DIR="$IMPROVER_BASE/experiments/final/$METRIC/configs/target"
TEST_CONFIG=experiments/final/test_eval.yaml

BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
TARGET_N=${TARGET_N:-16}
TARGET_SCORE=${TARGET_SCORE:-0.417}
TRAIN_BLOCKS=${TRAIN_BLOCKS:-512}
TEST_BLOCKS=${TEST_BLOCKS:-64}

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$IMPROVER_BASE/logs/final/length"

cd "$IMPROVER_BASE"

echo "Building ImProver..."
lake build eval_improver
sleep 5

run_train_eval() {
    local run_id="$1"
    local model="$2"

    ./improver run pipeline \
        --run_id "$run_id" \
        --annotation --informal --examples 4 --context 0 \
        --metric "$METRIC" --prompt_id final_train \
        --split train --model "$model" \
        --num_blocks "$TRAIN_BLOCKS" \
        --config "$TEST_CONFIG"
}

run_test_eval() {
    local run_id="$1"
    local model="$2"

    ./improver run pipeline \
        --run_id "$run_id" \
        --annotation --informal --examples 4 --context 0 \
        --metric "$METRIC" --prompt_id final_test \
        --split test --model "$model" \
        --num_blocks "$TEST_BLOCKS" \
        --config "$TEST_CONFIG"
}

make_config() {
    local base_model="$1"
    local lr="$2"
    local alpha="$3"
    local beta="$4"
    local dataset_path="$5"
    local output_dir="$6"
    local project_name="$7"
    local config_path="$8"
    local epochs="$9"

    python experiments/final/length/make_train_config.py \
        --base-model "$base_model" \
        --learning-rate "$lr" \
        --alpha "$alpha" \
        --beta "$beta" \
        --dataset-path "$dataset_path" \
        --output-dir "$output_dir" \
        --project-name "$project_name" \
        --config-path "$config_path" \
        --epochs "$epochs"
}

train_model() {
    local config_path="$1"

    accelerate launch -m axolotl.cli.train "$config_path"
}

check_target_score() {
    local csv_path="evals/IRPO_length_iter_3_test/analysis/BoN/data.csv"

    python - "$csv_path" "$TARGET_N" "$TARGET_SCORE" <<'PY'
import csv
import sys

csv_path, target_n, target_score = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])

with open(csv_path, newline="") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    if int(float(row["n_value"])) == target_n:
        improvement = float(row["improvement"])
        score = abs(improvement)
        print(
            f"Final length iter 3 n={target_n} improvement={improvement:.12f}; "
            f"abs={score:.12f}; target={target_score:.12f}"
        )
        if score < target_score:
            raise SystemExit(
                f"FAILED target: abs improvement {score:.12f} < {target_score:.12f}"
            )
        raise SystemExit(0)

raise SystemExit(f"FAILED target: n={target_n} not found in {csv_path}")
PY
}

# echo "Step 0: evaluate base model on train for length data."
# run_train_eval base_length_train "$BASE_MODEL"

# echo "Step 1: train IRPO_length_iter_1 from base_length_train."
# ITER1_DATA="$DATA_DIR/IRPO_length_iter_1.jsonl"
# ITER1_MODEL=/data/user_data/riyaza/saved_models/IRPO_length_iter_1
# ITER1_CONFIG="$CONFIG_DIR/IRPO_length_iter_1.yaml"

# ./improver run training_data \
#     --run_id base_length_train \
#     --output_path "$ITER1_DATA" \
#     --type dpo \
#     --num_invalid 4 \
#     --max_champions 4 \
#     --filter_threshold 0.5 \
#     --min_gap 0.0

# make_config "$BASE_MODEL" 5e-6 0.5 0.05 "$ITER1_DATA" "$ITER1_MODEL" IRPO_length_iter_1 "$ITER1_CONFIG" 1
# train_model "$ITER1_CONFIG"
# run_test_eval IRPO_length_iter_1_test "$ITER1_MODEL"
# run_train_eval IRPO_length_iter_1_train "$ITER1_MODEL"

echo "Step 2: train IRPO_length_iter_2 from IRPO_length_iter_1_train."
ITER2_DATA="$DATA_DIR/IRPO_length_iter_2.jsonl"
ITER2_MODEL=/data/user_data/riyaza/saved_models/IRPO_length_iter_2
ITER2_CONFIG="$CONFIG_DIR/IRPO_length_iter_2.yaml"

./improver run training_data \
    --run_id IRPO_length_iter_1_train \
    --output_path "$ITER2_DATA" \
    --type dpo \
    --num_invalid 4 \
    --max_champions 1 \
    --filter_threshold 0.8 \
    --min_gap 0.0 \
    --replay_buffer_split 0.2 \
    --replay_type join \
    --prev_run_id base_length_train \
    --hardness_weight 2.0

make_config "$ITER1_MODEL" 1e-6 1.0 0.02 "$ITER2_DATA" "$ITER2_MODEL" IRPO_length_iter_2 "$ITER2_CONFIG" 1
train_model "$ITER2_CONFIG"
run_test_eval IRPO_length_iter_2_test "$ITER2_MODEL"
run_train_eval IRPO_length_iter_2_train "$ITER2_MODEL"

echo "Step 3: train IRPO_length_iter_3 from IRPO_length_iter_2_train."
ITER3_DATA="$DATA_DIR/IRPO_length_iter_3.jsonl"
ITER3_MODEL=/data/user_data/riyaza/saved_models/IRPO_length_iter_3
ITER3_CONFIG="$CONFIG_DIR/IRPO_length_iter_3.yaml"

./improver run training_data \
    --run_id IRPO_length_iter_2_train \
    --output_path "$ITER3_DATA" \
    --type dpo \
    --num_invalid 2 \
    --max_champions 2 \
    --filter_threshold 1.0 \
    --min_gap 0.25 \
    --replay_buffer_split 0.0 \
    --replay_type replace \
    --prev_run_id base_length_train,IRPO_length_iter_1_train

make_config "$ITER2_MODEL" 5e-6 0.2 0.05 "$ITER3_DATA" "$ITER3_MODEL" IRPO_length_iter_3 "$ITER3_CONFIG" 1
train_model "$ITER3_CONFIG"
run_test_eval IRPO_length_iter_3_test "$ITER3_MODEL"
run_train_eval IRPO_length_iter_3_train "$ITER3_MODEL"

check_target_score
