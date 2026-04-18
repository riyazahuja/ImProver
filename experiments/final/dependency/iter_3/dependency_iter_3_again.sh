#!/bin/bash
#SBATCH --job-name=dependency_iter_3
#SBATCH --output=logs/final/dependency/real_iter_3_again.out
#SBATCH --error=logs/final/dependency/real_iter_3_again.err
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
ABLATION_DIR=$IMPROVER_BASE/experiments/final/dependency/iter_3
CONFIG_DIR=$ABLATION_DIR/configs
SWEEP_DIR=$ABLATION_DIR/configs/gridsearch
DATA_DIR=$ABLATION_DIR/data
SWEEP_DATASET_DIR=$ABLATION_DIR/data/gridsearch
SCRIPTS_DIR=$ABLATION_DIR/scripts
MODELS_DIR=/data/user_data/riyaza/saved_models/dependency/iter_3/gridsearch
EVALS_DIR=$IMPROVER_BASE/evals

BASE_MODEL="/data/user_data/riyaza/saved_models/IRPO_dependency_iter_2"
METRIC="dependency"
BASE_RUN_ID="IRPO_${METRIC}_iter_2_train"
PREV_RUN_IDS="base_dependency_train,IRPO_${METRIC}_iter_1_train"

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5

GRID_CONFIG=experiments/final/grid_eval.yaml
TEST_CONFIG=experiments/final/test_eval.yaml
TRAIN_CONFIG=experiments/final/train_eval.yaml


# ============ Default Hyperparameters ============

DEFAULT_ALPHA=0.5
DEFAULT_BETA=0.05
DEFAULT_LR=5e-6
DEFAULT_W=2
DEFAULT_L=2
DEFAULT_THRESHOLD=1.0
DEFAULT_GAP=0.0
DEFAULT_REPLAY_TYPE="replace"
DEFAULT_REPLAY_SPLIT=0.0



# ============ Run on Train Set ===========

./improver run pipeline --run_id $BASE_RUN_ID     --annotation  --informal --examples 4 --context 5     --metric dependency --prompt_id final_train     --split train --model $BASE_MODEL  --num_blocks 96     --config $TRAIN_CONFIG


# === GRID SEARCH === 


RUN_IDENTIFIER="${METRIC}_iter_3"



BEST_REPLAY_MODEL=$(cat $ABLATION_DIR/.best_replay)
echo "Best Replay Model: $BEST_REPLAY_MODEL"
BEST_REPLAY_TYPE=$(echo $BEST_REPLAY_MODEL | cut -d'_' -f9)
BEST_REPLAY_VALUE=$(echo $BEST_REPLAY_MODEL | cut -d'_' -f10)
echo "Best Replay Type: $BEST_REPLAY_TYPE"
echo "Best Replay Value: $BEST_REPLAY_VALUE"


BEST_ALPHA_BETA_MODEL=$(cat $ABLATION_DIR/.best_alpha_beta)
echo "Best Alpha Beta Model: $BEST_ALPHA_BETA_MODEL"
BEST_ALPHA_VALUE=$(echo $BEST_ALPHA_BETA_MODEL | cut -d'_' -f9)
BEST_BETA_VALUE=$(echo $BEST_ALPHA_BETA_MODEL | cut -d'_' -f10)
echo "Best Alpha: $BEST_ALPHA_VALUE"
echo "Best Beta: $BEST_BETA_VALUE"


BEST_W_L_MODEL=$(cat $ABLATION_DIR/.best_W_L)
echo "Best W L Model: $BEST_W_L_MODEL"
BEST_W_VALUE=$(echo $BEST_W_L_MODEL | cut -d'_' -f9)
BEST_L_VALUE=$(echo $BEST_W_L_MODEL | cut -d'_' -f10)
echo "Best W: $BEST_W_VALUE"
echo "Best L: $BEST_L_VALUE"


BEST_LR_MODEL=$(cat $ABLATION_DIR/.best_LR)
echo "Best LR Model: $BEST_LR_MODEL"
BEST_LR_VALUE=$(echo $BEST_LR_MODEL | cut -d'_' -f9)
echo "Best LR: $BEST_LR_VALUE"
BEST_THRESHOLD_MODEL=$(cat $ABLATION_DIR/.best_threshold)
echo "Best Threshold Model: $BEST_THRESHOLD_MODEL"
BEST_THRESHOLD_VALUE=$(echo $BEST_THRESHOLD_MODEL | cut -d'_' -f9)
echo "Best Threshold: $BEST_THRESHOLD_VALUE"



BEST_GAP_MODEL=$(cat $ABLATION_DIR/.best_gap)
echo "Best Gap Model: $BEST_GAP_MODEL"
BEST_GAP_VALUE=$(echo $BEST_GAP_MODEL | cut -d'_' -f9)
echo "Best Gap: $BEST_GAP_VALUE"


# ============ Final Training with Best Hyperparameters ===========

./improver run training_data --run_id $BASE_RUN_ID --output_path $DATA_DIR/final.jsonl --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $BEST_THRESHOLD_VALUE  --min_gap $BEST_GAP_VALUE  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS

MODEL_NAME="IRPO_${METRIC}_iter_3_longer"
OUTPUT_DIR=/data/user_data/riyaza/saved_models/$MODEL_NAME
CONFIG_PATH=$CONFIG_DIR/$MODEL_NAME.yaml

python experiments/final/dependency/make_train_config.py --base-model $BASE_MODEL \
    --learning-rate $BEST_LR_VALUE \
    --alpha $BEST_ALPHA_VALUE \
    --beta $BEST_BETA_VALUE \
    --dataset-path $DATA_DIR/final.jsonl \
    --output-dir $OUTPUT_DIR \
    --project-name $MODEL_NAME \
    --config-path $CONFIG_PATH

accelerate launch -m  axolotl.cli.train $CONFIG_PATH

./improver run pipeline --run_id ${MODEL_NAME}_test     --annotation  --informal --examples 4 --context 5     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config $TEST_CONFIG

./improver run pipeline --run_id ${MODEL_NAME}_test_2     --annotation  --informal --examples 4 --context 5     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config $TEST_CONFIG

# ./improver run pipeline --run_id ${MODEL_NAME}_test_3     --annotation  --informal --examples 4 --context 5     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config $TEST_CONFIG


# ./improver run pipeline --run_id ${MODEL_NAME}_train     --annotation  --informal --examples 4 --context 5     --metric $METRIC --prompt_id final_train     --split train --model $OUTPUT_DIR     --num_blocks 256     --config $TEST_CONFIG


