#!/bin/bash
#SBATCH --job-name=distill3
#SBATCH --output=logs/final/baselines/distill3.out
#SBATCH --error=logs/final/baselines/distill3.err
#SBATCH --cpus-per-task=64
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=250G


source $HOME/miniconda3/bin/activate env
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

# ============ Path Configuration ============
IMPROVER_BASE=/home/riyaza/eval_improver/improver
ABLATION_DIR=$IMPROVER_BASE/experiments/final/baselines/improver_baseline
CONFIG_DIR=$ABLATION_DIR/configs
SWEEP_DIR=$ABLATION_DIR/configs/gridsearch
DATA_DIR=$ABLATION_DIR/data
SWEEP_DATASET_DIR=$ABLATION_DIR/data/gridsearch
SCRIPTS_DIR=$ABLATION_DIR/scripts
MODELS_DIR=/data/user_data/riyaza/saved_models/baselines/improver_baseline/gridsearch
EVALS_DIR=$IMPROVER_BASE/evals


# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5

GRID_CONFIG=experiments/final/grid_eval.yaml
TEST_CONFIG=experiments/final/test_eval.yaml
DISTILL_CONFIG=experiments/final/distill_eval.yaml


export AZURE_OPENAI_API_KEY="7Ct6awSMrJg65ywG8SJEVeiqlootUzaITAjbrLiWHoK96wsSvQsDJQQJ99BIACHYHv6XJ3w3AAAAACOGzuBg"




export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

model="gpt-oss-120B"


# metric=length
# RUN_ID="${model}_${metric}_neuro_train_distillation"

# ./improver run pipeline --run_id ${RUN_ID}   --annotation  --informal --examples 4     --metric $metric --prompt_id final_train     --split train --azure true --server_concurrency 1000 --server_rate_limit 4096 --model $model --max_tokens 2048  --num_blocks 64     --config $DISTILL_CONFIG


metric=dependency
RUN_ID="${model}_${metric}_neuro_train_distillation"

./improver run pipeline --run_id ${RUN_ID}   --annotation  --informal --examples 4     --metric $metric --context 5 --prompt_id final_train     --split train --azure true --server_concurrency 1000 --server_rate_limit 4096 --model $model --max_tokens 2048  --num_blocks 64     --config $DISTILL_CONFIG


export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview"

model="gpt-5-mini"


metric=length
RUN_ID="${model}_${metric}_neuro_train_distillation"

./improver run pipeline --run_id ${RUN_ID}   --annotation  --informal --examples 4     --metric $metric --prompt_id final_train     --split train --azure true --server_concurrency 1000 --server_rate_limit 4096 --model $model --max_tokens 2048  --num_blocks 64     --config $DISTILL_CONFIG


metric=dependency
RUN_ID="${model}_${metric}_neuro_train_distillation"

./improver run pipeline --run_id ${RUN_ID}   --annotation  --informal --examples 4     --metric $metric --context 5 --prompt_id final_train     --split train --azure true --server_concurrency 1000 --server_rate_limit 4096 --model $model --max_tokens 2048  --num_blocks 64     --config $DISTILL_CONFIG

