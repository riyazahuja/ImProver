#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/final/length/final_real_iter_1.out
#SBATCH --error=logs/final/length/final_real_iter_1.err
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:7
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
ABLATION_DIR=$IMPROVER_BASE/experiments/final/length/iter_1
CONFIG_DIR=$ABLATION_DIR/configs
SWEEP_DIR=$ABLATION_DIR/configs/gridsearch
DATA_DIR=$ABLATION_DIR/data
SCRIPTS_DIR=$ABLATION_DIR/scripts
MODELS_DIR=/data/user_data/riyaza/saved_models/length/iter_1/gridsearch
EVALS_DIR=$IMPROVER_BASE/evals

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
METRIC="length"
BASE_RUN_ID="base_${METRIC}_train"

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5


# ============ Hyperparameter Settings ===========
RUN_IDS=""
for alpha in 0.2 0.5 1.0; do
    for beta in 0.02 0.05 0.1; do
        CURR_RUN_ID="length_iter_1_gridsearch_ab_${alpha}_${beta}"
        RUN_IDS="$RUN_IDS $CURR_RUN_ID"
    done
done

# select best

echo "Selecting best model..."
python $SCRIPTS_DIR/select_best_model.py \
    $RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_alpha_beta

BEST_ALPHA_BETA_MODEL=$(cat $ABLATION_DIR/.best_alpha_beta)
echo "Best Alpha Beta Model: $BEST_ALPHA_BETA_MODEL"
BEST_ALPHA_VALUE=$(echo $BEST_ALPHA_BETA_MODEL | cut -d'_' -f9)
BEST_BETA_VALUE=$(echo $BEST_ALPHA_BETA_MODEL | cut -d'_' -f10)
echo "Best Alpha: $BEST_ALPHA_VALUE"
echo "Best Beta: $BEST_BETA_VALUE"



RUN_IDS=""
for W in 1 2 4; do
    for L in 1 2 4; do       
        CURR_RUN_ID="length_iter_1_gridsearch_WL_${W}_${L}"
        RUN_IDS="$RUN_IDS $CURR_RUN_ID"
    done
done

# select best
echo "Selecting best model..."
python $SCRIPTS_DIR/select_best_model.py \
    $RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_W_L

BEST_W_L_MODEL=$(cat $ABLATION_DIR/.best_W_L)
echo "Best W L Model: $BEST_W_L_MODEL"
BEST_W_VALUE=$(echo $BEST_W_L_MODEL | cut -d'_' -f9)
BEST_L_VALUE=$(echo $BEST_W_L_MODEL | cut -d'_' -f10)
echo "Best W: $BEST_W_VALUE"
echo "Best L: $BEST_L_VALUE"


RUN_IDS=""
for LR in 1e-6 2e-6 5e-6 1e-5; do        
    CURR_RUN_ID="length_iter_1_gridsearch_LR_${LR}"
    RUN_IDS="$RUN_IDS $CURR_RUN_ID"
done

# select best
echo "Selecting best model..."
python $SCRIPTS_DIR/select_best_model.py \
    $RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_LR

BEST_LR_MODEL=$(cat $ABLATION_DIR/.best_LR)
echo "Best LR Model: $BEST_LR_MODEL"
BEST_LR_VALUE=$(echo $BEST_LR_MODEL | cut -d'_' -f9)
echo "Best LR: $BEST_LR_VALUE"

RUN_IDS=""
for threshold in 0.5 0.8 1.0; do
    CURR_RUN_ID="length_iter_1_gridsearch_threshold_${threshold}"
    RUN_IDS="$RUN_IDS $CURR_RUN_ID"
done

# select best
echo "Selecting best model..."
python $SCRIPTS_DIR/select_best_model.py \
    $RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_threshold

BEST_THRESHOLD_MODEL=$(cat $ABLATION_DIR/.best_threshold)
echo "Best Threshold Model: $BEST_THRESHOLD_MODEL"
BEST_THRESHOLD_VALUE=$(echo $BEST_THRESHOLD_MODEL | cut -d'_' -f9)
echo "Best Threshold: $BEST_THRESHOLD_VALUE"

RUN_IDS=""
for gap in 0.0 0.25 0.5; do
    CURR_RUN_ID="length_iter_1_gridsearch_gap_${gap}"
    RUN_IDS="$RUN_IDS $CURR_RUN_ID"
done

# select best
echo "Selecting best model..."
python $SCRIPTS_DIR/select_best_model.py \
    $RUN_IDS \
    --evals-path $EVALS_DIR \
    --models-path $MODELS_DIR \
    --output-file $ABLATION_DIR/.best_gap

BEST_GAP_MODEL=$(cat $ABLATION_DIR/.best_gap)
echo "Best Gap Model: $BEST_GAP_MODEL"
BEST_GAP_VALUE=$(echo $BEST_GAP_MODEL | cut -d'_' -f9)
echo "Best Gap: $BEST_GAP_VALUE"


# ============ Final Training with Best Hyperparameters ===========

./improver run training_data --run_id base_length_train --output_path experiments/final/length/iter_1/data/final.jsonl --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $BEST_THRESHOLD_VALUE  --min_gap $BEST_GAP_VALUE  


MODEL_NAME="FINAL_IRPO_length_iter_1"
OUTPUT_DIR=/data/user_data/riyaza/saved_models/FINAL_IRPO_length_iter_1
CONFIG_PATH=experiments/final/length/iter_1/configs/FINAL_IRPO_length_iter_1.yaml

python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
    --learning-rate $BEST_LR_VALUE \
    --alpha $BEST_ALPHA_VALUE \
    --beta $BEST_BETA_VALUE \
    --dataset-path experiments/final/length/iter_1/data/final.jsonl \
    --output-dir $OUTPUT_DIR \
    --project-name $MODEL_NAME \
    --config-path $CONFIG_PATH \
    --epochs 3

accelerate launch -m  axolotl.cli.train $CONFIG_PATH

./improver run pipeline --run_id IRPO_length_iter_1_test     --annotation  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1     --num_blocks 64     --config experiments/final/test_eval.yaml
