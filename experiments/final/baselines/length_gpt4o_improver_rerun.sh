#!/bin/bash
#SBATCH --job-name=length_gpt4o_improver
#SBATCH --output=logs/final/baselines/length_gpt4o_improver_rerun.out
#SBATCH --error=logs/final/baselines/length_gpt4o_improver_rerun.err
#SBATCH --cpus-per-task=64
#SBATCH --partition=cpu
#SBATCH --time=4:00:00
#SBATCH --mem=100G

set -euo pipefail

source "$HOME/miniconda3/bin/activate" env
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export RAY_TMPDIR=/data/user_data/riyaza/ray_tmp
export HF_HOME="/data/user_data/riyaza/HF"

export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

IMPROVER_BASE=/home/riyaza/eval_improver/improver
TEST_CONFIG=experiments/final/test_eval.yaml
RERUN_TAG=${RERUN_TAG:-rerun_20260417}

mkdir -p "$IMPROVER_BASE/logs/final/baselines"
cd "$IMPROVER_BASE"

echo "Building ImProver..."
lake build eval_improver
sleep 5

# : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before submitting this job}"
export OPENROUTER_API_KEY="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"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OPENROUTER_X_TITLE="${OPENROUTER_X_TITLE:-ImProver length baseline rerun}"
# `--azure true` selects the async server/API inference path; the provider is
# OpenRouter because OPENROUTER_API_KEY and OPENAI_BASE_URL are set above.

metric=length
model="openai/gpt-4o"
model_label="gpt-4o"

RUN_ID="${model_label}_${metric}_test_${RERUN_TAG}"
echo "Running GPT-4o length baseline: ${RUN_ID}"
./improver run pipeline \
  --run_id "$RUN_ID" \
  --metric "$metric" \
  --prompt_id final_test \
  --split test \
  --azure true \
  --server_concurrency 1000 \
  --server_rate_limit 4096 \
  --model "$model" \
  --max_tokens 480 \
  --num_blocks 64 \
  --config "$TEST_CONFIG"
  # --examples 4 \

# RUN_ID="${model_label}_${metric}_neuro_test_${RERUN_TAG}"
# echo "Running GPT-4o length neuro baseline: ${RUN_ID}"
# ./improver run pipeline \
#   --run_id "$RUN_ID" \
#   --annotation \
#   --informal \
#   --examples 4 \
#   --context 5 \
#   --metric "$metric" \
#   --prompt_id final_test \
#   --split test \
#   --azure true \
#   --server_concurrency 1000 \
#   --server_rate_limit 4096 \
#   --model "$model" \
#   --max_tokens 1024 \
#   --num_blocks 64 \
#   --config "$TEST_CONFIG"

RUN_ID="improver_${metric}_test_${RERUN_TAG}"
echo "Running ImProver length baseline: ${RUN_ID}"
./improver run pipeline \
  --run_id "$RUN_ID" \
  --annotation \
  --metric "$metric" \
  --prompt_id final_test \
  --split test \
  --azure true \
  --server_concurrency 1000 \
  --server_rate_limit 4096 \
  --model "$model" \
  --max_tokens 480 \
  --num_blocks 64 \
  --config "$TEST_CONFIG"
