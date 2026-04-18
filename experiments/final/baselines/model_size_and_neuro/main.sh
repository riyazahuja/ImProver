#!/bin/bash
#SBATCH --job-name=i3_again
#SBATCH --output=logs/final/baselines/i3_again.out
#SBATCH --error=logs/final/baselines/i3_again.err
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40S:7
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
ABLATION_DIR=$IMPROVER_BASE/experiments/final/baselines/model_size_and_neuro
CONFIG_DIR=$ABLATION_DIR/configs
SWEEP_DIR=$ABLATION_DIR/configs/gridsearch
DATA_DIR=$ABLATION_DIR/data
SWEEP_DATASET_DIR=$ABLATION_DIR/data/gridsearch
SCRIPTS_DIR=$ABLATION_DIR/scripts
MODELS_DIR=/data/user_data/riyaza/saved_models/baselines/model_size_and_neuro/gridsearch
EVALS_DIR=$IMPROVER_BASE/evals



# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5

GRID_CONFIG=experiments/final/grid_eval.yaml
TEST_CONFIG=experiments/final/test_eval.yaml



# # ============ Run base models ===========
# for metric in length dependency; do
#     model=14B
#     MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-${model}"
#     RUN_ID="base_${metric}_${model}_test"

#     ./improver run pipeline --run_id ${RUN_ID}_2 --examples 4  --metric $metric --prompt_id final_test     --split test --model $MODEL_PATH  --num_blocks 64  --max_tokens 1600   --config $TEST_CONFIG

#     RUN_ID="base_${metric}_${model}_neuro_test"

#     ./improver run pipeline --run_id ${RUN_ID}_anno --examples 4 --annotation  --metric $metric --prompt_id final_test     --split test --model $MODEL_PATH  --num_blocks 64  --max_tokens 1600   --config $TEST_CONFIG
    
#     ./improver run pipeline --run_id ${RUN_ID}_anno_inf --examples 4 --annotation  --informal  --metric $metric --prompt_id final_test     --split test --model $MODEL_PATH  --num_blocks 64  --max_tokens 1600   --config $TEST_CONFIG

#     ./improver run pipeline --run_id ${RUN_ID}_full --examples 4 --annotation  --informal  --context 5 --metric $metric --prompt_id final_test     --split test --model $MODEL_PATH  --num_blocks 64  --max_tokens 1600   --config $TEST_CONFIG
# done


./improver run pipeline --run_id length_iter_3_again --examples 0 --annotation --informal  --metric length --prompt_id final_test     --split test --model  /data/user_data/riyaza/saved_models/IRPO_length_iter_1 --num_blocks 64  --max_tokens 2000   --config $TEST_CONFIG


# model=32B
# for metric in length dependency; do
#     MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-${model}"

#     RUN_ID="base_${metric}_${model}_test"
#     ./improver run pipeline --run_id $RUN_ID  --examples 4     --metric $metric --prompt_id final_test     --split test --model $MODEL_PATH  --num_blocks 64  --max_tokens 1500   --config $TEST_CONFIG
    
#     RUN_ID="base_${metric}_${model}_neuro_test"

#     ./improver run pipeline --run_id ${RUN_ID}_no_ctx   --annotation  --informal --examples 4     --metric $metric --prompt_id final_test     --split test --model $MODEL_PATH  --num_blocks 64  --max_tokens 1500   --config $TEST_CONFIG

#     ./improver run pipeline --run_id $RUN_ID   --annotation  --informal --examples 4 --context 5     --metric $metric --prompt_id final_test     --split test --model $MODEL_PATH  --num_blocks 64  --max_tokens 1500   --config $TEST_CONFIG
# done
