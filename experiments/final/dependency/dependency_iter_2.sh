#!/bin/bash
#SBATCH --job-name=iter2_dependency
#SBATCH --output=logs/final/dependency/iter_2.out
#SBATCH --error=logs/final/dependency/iter_2.err
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:A6000:8
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
ITER2_DIR=$IMPROVER_BASE/experiments/final/dependency/iter_2
CONFIG_DIR=$ITER2_DIR/configs
DATA_DIR=$ITER2_DIR/data
EVALS_DIR=$IMPROVER_BASE/evals


# Best model from iter_2: gap_0.0
# BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BASE_MODEL="/data/user_data/riyaza/saved_models/IRPO_dependency_iter_1"
METRIC="dependency"

# Use the existing train eval that user already ran
BASE_RUN_ID="IRPO_dependency_iter_1_train"
BEST_REPLAY_VALUE=0.2 
BEST_REPLAY_TYPE="join"
PREV_RUN_IDS="base_dependency_train"

# ============ Best Hyperparameters from Iter 2 Grid Search ============
# From gap_0.0 which was trained with these params:
BEST_W=1
BEST_L=4
BEST_ALPHA=1.0
BEST_BETA=0.02
BEST_LR=1e-6
BEST_THRESHOLD=0.8
BEST_GAP=0.0

GRID_CONFIG=experiments/final/grid_eval.yaml
TEST_CONFIG=experiments/final/test_eval.yaml

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5




# ============ Step 0: Get training data ============

# ./improver run pipeline --run_id base_dependency_test     --annotation --context 5  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_test     --split test --model $BASE_MODEL     --num_blocks 64     --config $TEST_CONFIG

./improver run pipeline --run_id $BASE_RUN_ID     --annotation --context 5  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_train     --split train --model $BASE_MODEL     --num_blocks 256     --config $TEST_CONFIG


# ============ Step 1: Generate Training Data ============

./improver run training_data --run_id $BASE_RUN_ID \
    --output_path $DATA_DIR/iter_2.jsonl \
    --type dpo \
    --num_invalid $BEST_L \
    --max_champions $BEST_W \
    --filter_threshold $BEST_THRESHOLD \
    --min_gap $BEST_GAP \
    --replay_buffer_split $BEST_REPLAY_VALUE \
    --replay_type $BEST_REPLAY_TYPE \
    --prev_run_id $PREV_RUN_IDS
    
echo "Training data generated at $DATA_DIR/iter_2.jsonl"
wc -l $DATA_DIR/iter_2.jsonl

# ============ Step 2: Train Iter 2 Model ============
echo "Training iter_2 model..."

MODEL_NAME="IRPO_${METRIC}_iter_2"
OUTPUT_DIR=/data/user_data/riyaza/saved_models/$MODEL_NAME
CONFIG_PATH=$CONFIG_DIR/$MODEL_NAME.yaml

python experiments/final/dependency/make_train_config.py --base-model $BASE_MODEL \
    --learning-rate $BEST_LR \
    --alpha $BEST_ALPHA \
    --beta $BEST_BETA \
    --dataset-path $DATA_DIR/iter_2.jsonl \
    --output-dir $OUTPUT_DIR \
    --project-name $MODEL_NAME \
    --config-path $CONFIG_PATH

accelerate launch -m axolotl.cli.train $CONFIG_PATH

# ============ Step 3: Evaluate on Test Set ============
echo "Evaluating iter_2 model on test set..."

./improver run pipeline --run_id IRPO_${METRIC}_iter_2_test_from_base \
    --annotation --informal --examples 4 --context 5 \
    --metric $METRIC --prompt_id final_test \
    --split test --model $OUTPUT_DIR \
    --num_blocks 64 \
    --config $TEST_CONFIG
