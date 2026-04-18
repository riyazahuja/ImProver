#!/bin/bash
#SBATCH --job-name=iter4_length
#SBATCH --output=logs/final/length/iter_4.out
#SBATCH --error=logs/final/length/iter_4.err
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40S:8
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
ITER4_DIR=$IMPROVER_BASE/experiments/final/length/iter_4
CONFIG_DIR=$ITER4_DIR/configs
DATA_DIR=$ITER4_DIR/data
EVALS_DIR=$IMPROVER_BASE/evals

# Best model from iter_2: gap_0.0
BASE_MODEL="/data/user_data/riyaza/saved_models/IRPO_length_iter_3"
METRIC="length"

# Use the existing train eval that user already ran
BASE_RUN_ID="IRPO_length_iter_3_train"
BEST_REPLAY_VALUE=0.2
BEST_REPLAY_TYPE="replace"
PREV_RUN_IDS="IRPO_length_iter_2_train,IRPO_length_iter_1_train,base_length_train"

# ============ Best Hyperparameters from Iter 3 Grid Search ============
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

# ============ Step 1: Generate Training Data ============
# User already ran inference on train set, so we skip to training data generation
echo "Generating training data for iter_4..."

./improver run training_data --run_id $BASE_RUN_ID \
    --output_path $DATA_DIR/iter_4.jsonl \
    --type dpo \
    --num_invalid $BEST_L \
    --max_champions $BEST_W \
    --filter_threshold $BEST_THRESHOLD \
    --min_gap $BEST_GAP \
    --replay_buffer_split $BEST_REPLAY_VALUE \
    --replay_type $BEST_REPLAY_TYPE \
    --prev_run_id $PREV_RUN_IDS

echo "Training data generated at $DATA_DIR/iter_4.jsonl"
wc -l $DATA_DIR/iter_4.jsonl

# ============ Step 2: Train Iter 4 Model ============
echo "Training iter_4 model..."
MODEL_NAME="IRPO_${METRIC}_iter_4_2_CONFIG"
OUTPUT_DIR=/data/user_data/riyaza/saved_models/$MODEL_NAME
CONFIG_PATH=$CONFIG_DIR/$MODEL_NAME.yaml

python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
    --learning-rate $BEST_LR \
    --alpha $BEST_ALPHA \
    --beta $BEST_BETA \
    --dataset-path $DATA_DIR/iter_4.jsonl \
    --output-dir $OUTPUT_DIR \
    --project-name $MODEL_NAME \
    --config-path $CONFIG_PATH

accelerate launch -m axolotl.cli.train $CONFIG_PATH

# ============ Step 4: Evaluate on Test Set ============
echo "Evaluating iter_4 model on test set..."

./improver run pipeline --run_id IRPO_${METRIC}_iter_4_test_2_CONFIG \
    --annotation --informal --examples 4 \
    --metric $METRIC --prompt_id final_test \
    --split test --model $OUTPUT_DIR \
    --num_blocks 64 \
    --config $TEST_CONFIG





# =========== OG ============


# Use the existing train eval that user already ran
BASE_RUN_ID="IRPO_length_iter_3_train"
BEST_REPLAY_VALUE=0.0
BEST_REPLAY_TYPE="replace"
PREV_RUN_IDS="IRPO_length_iter_2_train,IRPO_length_iter_1_train,base_length_train"

# ============ Best Hyperparameters from Iter 3 Grid Search ============
# From gap_0.0 which was trained with these params:
BEST_W=2
BEST_L=2
BEST_ALPHA=0.2
BEST_BETA=0.05
BEST_LR=5e-6
BEST_THRESHOLD=1.0
BEST_GAP=0.25

GRID_CONFIG=experiments/final/grid_eval.yaml
TEST_CONFIG=experiments/final/test_eval.yaml

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5

# ============ Step 1: Generate Training Data ============
# User already ran inference on train set, so we skip to training data generation
echo "Generating training data for iter_4..."

./improver run training_data --run_id $BASE_RUN_ID \
    --output_path $DATA_DIR/iter_4.jsonl \
    --type dpo \
    --num_invalid $BEST_L \
    --max_champions $BEST_W \
    --filter_threshold $BEST_THRESHOLD \
    --min_gap $BEST_GAP \
    --replay_buffer_split $BEST_REPLAY_VALUE \
    --replay_type $BEST_REPLAY_TYPE \
    --prev_run_id $PREV_RUN_IDS

echo "Training data generated at $DATA_DIR/iter_4.jsonl"
wc -l $DATA_DIR/iter_4.jsonl

# ============ Step 2: Train Iter 4 Model ============
echo "Training iter_4 model..."
MODEL_NAME="IRPO_${METRIC}_iter_4_3_CONFIG"
OUTPUT_DIR=/data/user_data/riyaza/saved_models/$MODEL_NAME
CONFIG_PATH=$CONFIG_DIR/$MODEL_NAME.yaml

python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
    --learning-rate $BEST_LR \
    --alpha $BEST_ALPHA \
    --beta $BEST_BETA \
    --dataset-path $DATA_DIR/iter_4.jsonl \
    --output-dir $OUTPUT_DIR \
    --project-name $MODEL_NAME \
    --config-path $CONFIG_PATH

accelerate launch -m axolotl.cli.train $CONFIG_PATH

# ============ Step 4: Evaluate on Test Set ============
echo "Evaluating iter_4 model on test set..."

./improver run pipeline --run_id IRPO_${METRIC}_iter_4_test_3_CONFIG \
    --annotation --informal --examples 4 \
    --metric $METRIC --prompt_id final_test \
    --split test --model $OUTPUT_DIR \
    --num_blocks 64 \
    --config $TEST_CONFIG