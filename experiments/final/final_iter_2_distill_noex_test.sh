#!/bin/bash
#SBATCH --job-name=distill_test3
#SBATCH --output=logs/final/all/length_distill_test3.out
#SBATCH --error=logs/final/all/length_distill_test3.err
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

# ============ Temp Directory Configuration (prevent /tmp exhaustion) ============
export TMPDIR=/data/user_data/riyaza/tmp
export TEMP=/data/user_data/riyaza/tmp
export TMP=/data/user_data/riyaza/tmp
export TRITON_CACHE_DIR=/data/user_data/riyaza/triton_cache
mkdir -p $TMPDIR $TRITON_CACHE_DIR

# ============ Path Configuration ============


METRIC=length
IMPROVER_BASE=/home/riyaza/eval_improver/improver
ABLATION_DIR=$IMPROVER_BASE/experiments/final/${METRIC}/iter_2
CONFIG_DIR=$ABLATION_DIR/configs
SWEEP_DIR=$ABLATION_DIR/configs/gridsearch
DATA_DIR=$ABLATION_DIR/data
SCRIPTS_DIR=$ABLATION_DIR/scripts
MODELS_DIR=/data/user_data/riyaza/saved_models/${METRIC}/iter_2/gridsearch
EVALS_DIR=$IMPROVER_BASE/evals

# BASE_MODEL="/data/user_data/riyaza/saved_models/IRPO_${METRIC}_iter_1"
BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BASE_RUN_ID="IRPO_${METRIC}_iter_1_train"
PREV_RUN_IDS="base_${METRIC}_train"

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5


# ============ Hyperparameter Settings ===========

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

BASE_DATASET=experiments/final/${METRIC}/iter_2/data/final_raw.jsonl
GPT_DISTILL_ID=gpt-5-mini_length_neuro_train_distillation
OSS_DISTILL_ID=gpt-oss-120B_length_neuro_train_distillation
GPT_DISTILL_DATASET=experiments/final/${METRIC}/iter_2/data/${GPT_DISTILL_ID}.jsonl
OSS_DISTILL_DATASET=experiments/final/${METRIC}/iter_2/data/${OSS_DISTILL_ID}.jsonl

GPT_MERGED_DATASET=experiments/final/${METRIC}/iter_2/data/final_gpt_merged.jsonl
FULL_MERGED_DATASET=experiments/final/${METRIC}/iter_2/data/final.jsonl

BEST_W_VALUE=4
BEST_L_VALUE=4
BEST_THRESHOLD_VALUE=0.5
BEST_REPLAY_VALUE=0.4
BEST_REPLAY_TYPE="join"

./improver run training_data --run_id $BASE_RUN_ID --output_path $BASE_DATASET  --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $BEST_THRESHOLD_VALUE  --min_gap $BEST_GAP_VALUE  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS

./improver run training_data --run_id $GPT_DISTILL_ID --output_path $GPT_DISTILL_DATASET --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $BEST_THRESHOLD_VALUE  --min_gap $BEST_GAP_VALUE  

./improver run training_data --run_id $OSS_DISTILL_ID --output_path $OSS_DISTILL_DATASET --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $BEST_THRESHOLD_VALUE  --min_gap $BEST_GAP_VALUE  

python experiments/final/intermix_distill_data.py --raw $BASE_DATASET --distill $GPT_DISTILL_DATASET --output $GPT_MERGED_DATASET --distill_frac 1.0

python experiments/final/intermix_distill_data.py --raw $GPT_MERGED_DATASET --distill $OSS_DISTILL_DATASET --output $FULL_MERGED_DATASET --distill_frac 1.0


# =========== No distillation ===========
RUN_ID=FINAL_IRPO_${METRIC}_iter_2_wahoo2_no_distill
CONFIG_PATH=${CONFIG_DIR}/${RUN_ID}.yaml
OUTPUT_DIR=/data/user_data/riyaza/saved_models/${RUN_ID}


python experiments/final/${METRIC}/make_train_config.py --base-model $BASE_MODEL \
    --learning-rate 5e-6 \
    --alpha 0.5 \
    --beta 0.05 \
    --dataset-path $BASE_DATASET \
    --output-dir $OUTPUT_DIR \
    --project-name $RUN_ID \
    --config-path $CONFIG_PATH

accelerate launch -m  axolotl.cli.train $CONFIG_PATH

./improver run pipeline --run_id $RUN_ID     --annotation  --informal --examples 0     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config experiments/final/test_eval.yaml


./improver run pipeline --run_id ${RUN_ID}_2     --annotation  --informal --examples 0     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config experiments/final/test_eval.yaml




# # =========== gpt distillation ===========

# RUN_ID=FINAL_IRPO_${METRIC}_iter_2_wahoo2_gpt_distill
# CONFIG_PATH=${CONFIG_DIR}/${RUN_ID}.yaml
# OUTPUT_DIR=/data/user_data/riyaza/saved_models/${RUN_ID}


# python experiments/final/${METRIC}/make_train_config.py --base-model $BASE_MODEL \
#     --learning-rate 5e-6 \
#     --alpha 0.5 \
#     --beta 0.05 \
#     --dataset-path $BASE_DATASET \
#     --output-dir $OUTPUT_DIR \
#     --project-name $RUN_ID \
#     --config-path $CONFIG_PATH

# accelerate launch -m  axolotl.cli.train $CONFIG_PATH

# ./improver run pipeline --run_id $RUN_ID     --annotation  --informal --examples 0     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config experiments/final/test_eval.yaml


# ./improver run pipeline --run_id ${RUN_ID}_2     --annotation  --informal --examples 0     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config experiments/final/test_eval.yaml




# =========== all distillation ===========

RUN_ID=FINAL_IRPO_${METRIC}_iter_2_wahoo2_all_distill
CONFIG_PATH=${CONFIG_DIR}/${RUN_ID}.yaml
OUTPUT_DIR=/data/user_data/riyaza/saved_models/${RUN_ID}


python experiments/final/${METRIC}/make_train_config.py --base-model $BASE_MODEL \
    --learning-rate 5e-6 \
    --alpha 0.5 \
    --beta 0.05 \
    --dataset-path $BASE_DATASET \
    --output-dir $OUTPUT_DIR \
    --project-name $RUN_ID \
    --config-path $CONFIG_PATH

accelerate launch -m  axolotl.cli.train $CONFIG_PATH

./improver run pipeline --run_id $RUN_ID     --annotation  --informal --examples 0     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config experiments/final/test_eval.yaml


./improver run pipeline --run_id ${RUN_ID}_2     --annotation  --informal --examples 0     --metric $METRIC --prompt_id final_test     --split test --model $OUTPUT_DIR     --num_blocks 64     --config experiments/final/test_eval.yaml

