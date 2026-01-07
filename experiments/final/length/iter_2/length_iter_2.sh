#!/bin/bash
#SBATCH --job-name=baselines2
#SBATCH --output=logs/final/length/real_iter_2_4.out
#SBATCH --error=logs/final/length/real_iter_2_4.err
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
ABLATION_DIR=$IMPROVER_BASE/experiments/final/length/iter_2
CONFIG_DIR=$ABLATION_DIR/configs
SWEEP_DIR=$ABLATION_DIR/configs/gridsearch
DATA_DIR=$ABLATION_DIR/data
SWEEP_DATASET_DIR=$ABLATION_DIR/data/gridsearch
SCRIPTS_DIR=$ABLATION_DIR/scripts
MODELS_DIR=/data/user_data/riyaza/saved_models/length/iter_2/gridsearch
EVALS_DIR=$IMPROVER_BASE/evals

BASE_MODEL="/data/user_data/riyaza/saved_models/IRPO_length_iter_1"
METRIC="length"
BASE_RUN_ID="IRPO_${METRIC}_iter_1_train"
PREV_RUN_IDS="base_length_train"

# ============ Build ImProver ============
echo "Building ImProver..."
cd $IMPROVER_BASE
lake build eval_improver
sleep 5

GRID_CONFIG=experiments/final/grid_eval.yaml
TEST_CONFIG=experiments/final/test_eval.yaml


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

# ./improver run pipeline --run_id $BASE_RUN_ID     --annotation  --informal --examples 4     --metric length --prompt_id final_train     --split train --model $BASE_MODEL  --num_blocks 96     --config $TEST_CONFIG


# === GRID SEARCH === 


RUN_IDENTIFIER="${METRIC}_iter_2"


# # --- replay ---
# for replay_type in replace join; do
#     for replay_split in 0.0 0.2 0.4; do
#             ./improver run training_data --run_id $BASE_RUN_ID --output_path $SWEEP_DATASET_DIR/replay.jsonl --type dpo --num_invalid $DEFAULT_L --max_champions $DEFAULT_W --filter_threshold $DEFAULT_THRESHOLD  --min_gap $DEFAULT_GAP  --replay_buffer_split $replay_split --replay_type $replay_type --prev_run_id $PREV_RUN_IDS


#             MODEL_NAME="replay_${replay_type}_${replay_split}"

#             OUTPUT_DIR=$MODELS_DIR/$MODEL_NAME

#             CONFIG_PATH=$SWEEP_DIR/$MODEL_NAME.yaml
            
#             python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
#                 --learning-rate $DEFAULT_LR \
#                 --alpha $DEFAULT_ALPHA \
#                 --beta $DEFAULT_BETA \
#                 --dataset-path $SWEEP_DATASET_DIR/replay.jsonl \
#                 --output-dir $OUTPUT_DIR \
#                 --project-name $MODEL_NAME \
#                 --config-path $CONFIG_PATH

#             accelerate launch -m  axolotl.cli.train $CONFIG_PATH
#     done
# done

# RUN_IDS=""
# for replay_type in replace join; do
#     for replay_split in 0.0 0.2 0.4; do
#         CURR_RUN_ID="${RUN_IDENTIFIER}_gridsearch_replay_${replay_type}_${replay_split}"

#         MODEL_NAME="replay_${replay_type}_${replay_split}"

#         RUN_IDS="$RUN_IDS $CURR_RUN_ID"
#         ./improver run pipeline --run_id $CURR_RUN_ID  --annotation  --informal --examples 4 --metric length --prompt_id final_test   --split test --model $MODELS_DIR/$MODEL_NAME     --num_blocks 64     --config $GRID_CONFIG
#     done
# done

# # select best
# echo "Selecting best model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     $RUN_IDS \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_replay

BEST_REPLAY_MODEL=$(cat $ABLATION_DIR/.best_replay)
echo "Best Replay Model: $BEST_REPLAY_MODEL"
BEST_REPLAY_TYPE=$(echo $BEST_REPLAY_MODEL | cut -d'_' -f9)
BEST_REPLAY_VALUE=$(echo $BEST_REPLAY_MODEL | cut -d'_' -f10)
echo "Best Replay Type: $BEST_REPLAY_TYPE"
echo "Best Replay Value: $BEST_REPLAY_VALUE"


# # First iteration, so run a/b + W/L

# # --- alpha/beta ---
# ./improver run training_data --run_id $BASE_RUN_ID --output_path $SWEEP_DATASET_DIR/alphabeta.jsonl --type dpo --num_invalid $DEFAULT_L --max_champions $DEFAULT_W --filter_threshold $DEFAULT_THRESHOLD  --min_gap $DEFAULT_GAP  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS


# for alpha in 0.2 0.5 1.0; do
#     for beta in 0.02 0.05 0.1; do
#         MODEL_NAME="ab_${alpha}_${beta}"

#         OUTPUT_DIR=$MODELS_DIR/$MODEL_NAME

#         CONFIG_PATH=$SWEEP_DIR/$MODEL_NAME.yaml
        
#         python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
#             --learning-rate $DEFAULT_LR \
#             --alpha $alpha \
#             --beta $beta \
#             --dataset-path $SWEEP_DATASET_DIR/alphabeta.jsonl \
#             --output-dir $OUTPUT_DIR \
#             --project-name $MODEL_NAME \
#             --config-path $CONFIG_PATH

#         accelerate launch -m  axolotl.cli.train $CONFIG_PATH
#     done
# done
# RUN_IDS=""
# for alpha in 0.2 0.5 1.0; do
#     for beta in 0.02 0.05 0.1; do
#         CURR_RUN_ID="${RUN_IDENTIFIER}_gridsearch_ab_${alpha}_${beta}"
#         MODEL_NAME="ab_${alpha}_${beta}"
#         RUN_IDS="$RUN_IDS $CURR_RUN_ID"
#         ./improver run pipeline --run_id $CURR_RUN_ID  --annotation  --informal --examples 4 --metric length --prompt_id final_test   --split test --model $MODELS_DIR/$MODEL_NAME     --num_blocks 64     --config $GRID_CONFIG
#     done
# done

# # select best

# echo "Selecting best model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     $RUN_IDS \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_alpha_beta

BEST_ALPHA_BETA_MODEL=$(cat $ABLATION_DIR/.best_alpha_beta)
echo "Best Alpha Beta Model: $BEST_ALPHA_BETA_MODEL"
BEST_ALPHA_VALUE=$(echo $BEST_ALPHA_BETA_MODEL | cut -d'_' -f9)
BEST_BETA_VALUE=$(echo $BEST_ALPHA_BETA_MODEL | cut -d'_' -f10)
echo "Best Alpha: $BEST_ALPHA_VALUE"
echo "Best Beta: $BEST_BETA_VALUE"

# # --- W/L ---
# for W in 1 2 4; do
#     for L in 1 2 4; do        
#         ./improver run training_data --run_id $BASE_RUN_ID --output_path $SWEEP_DATASET_DIR/WL_${W}_${L}.jsonl --type dpo --num_invalid $L --max_champions $W --filter_threshold $DEFAULT_THRESHOLD  --min_gap $DEFAULT_GAP  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS

#         MODEL_NAME="WL_${W}_${L}"
#         OUTPUT_DIR=$MODELS_DIR/$MODEL_NAME
        
#         CONFIG_PATH=$SWEEP_DIR/$MODEL_NAME.yaml

#         python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
#             --learning-rate $DEFAULT_LR \
#             --alpha $BEST_ALPHA_VALUE \
#             --beta $BEST_BETA_VALUE \
#             --dataset-path $SWEEP_DATASET_DIR/WL_${W}_${L}.jsonl \
#             --output-dir $OUTPUT_DIR \
#             --project-name $MODEL_NAME \
#             --config-path $CONFIG_PATH
        
#         accelerate launch -m  axolotl.cli.train $CONFIG_PATH
#     done
# done

# RUN_IDS=""
# for W in 1 2 4; do
#     for L in 1 2 4; do       
#         CURR_RUN_ID="${RUN_IDENTIFIER}_gridsearch_WL_${W}_${L}"
#         RUN_IDS="$RUN_IDS $CURR_RUN_ID"
#         ./improver run pipeline --run_id $CURR_RUN_ID  --annotation  --informal --examples 4 --metric length --prompt_id final_test   --split test --model $MODELS_DIR/WL_${W}_${L}     --num_blocks 64     --config $GRID_CONFIG
#     done
# done

# # select best
# echo "Selecting best model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     $RUN_IDS \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_W_L

BEST_W_L_MODEL=$(cat $ABLATION_DIR/.best_W_L)
echo "Best W L Model: $BEST_W_L_MODEL"
BEST_W_VALUE=$(echo $BEST_W_L_MODEL | cut -d'_' -f9)
BEST_L_VALUE=$(echo $BEST_W_L_MODEL | cut -d'_' -f10)
echo "Best W: $BEST_W_VALUE"
echo "Best L: $BEST_L_VALUE"


# --- LR ---
# ./improver run training_data --run_id $BASE_RUN_ID --output_path $SWEEP_DATASET_DIR/lr.jsonl --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $DEFAULT_THRESHOLD  --min_gap $DEFAULT_GAP  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS

# for LR in 1e-6 2e-6 5e-6 1e-5; do        
#         MODEL_NAME="LR_${LR}"
#         OUTPUT_DIR=$MODELS_DIR/$MODEL_NAME
#         CONFIG_PATH=$SWEEP_DIR/$MODEL_NAME.yaml

#         python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
#             --learning-rate $LR \
#             --alpha $BEST_ALPHA_VALUE \
#             --beta $BEST_BETA_VALUE \
#             --dataset-path $SWEEP_DATASET_DIR/lr.jsonl \
#             --output-dir $OUTPUT_DIR \
#             --project-name $MODEL_NAME \
#             --config-path $CONFIG_PATH

#         accelerate launch -m  axolotl.cli.train $CONFIG_PATH
# done

# RUN_IDS=""
# for LR in 1e-6 2e-6 5e-6 1e-5; do        
#     CURR_RUN_ID="${RUN_IDENTIFIER}_gridsearch_LR_${LR}"
#     RUN_IDS="$RUN_IDS $CURR_RUN_ID"
#     ./improver run pipeline --run_id $CURR_RUN_ID  --annotation  --informal --examples 4 --metric length --prompt_id final_test   --split test --model $MODELS_DIR/LR_${LR}     --num_blocks 64     --config $GRID_CONFIG
# done

# # select best
# echo "Selecting best model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     $RUN_IDS \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_LR

BEST_LR_MODEL=$(cat $ABLATION_DIR/.best_LR)
echo "Best LR Model: $BEST_LR_MODEL"
BEST_LR_VALUE=$(echo $BEST_LR_MODEL | cut -d'_' -f9)
echo "Best LR: $BEST_LR_VALUE"

# --- filter threshold ---

# for threshold in 0.5 0.8 1.0; do
#         ./improver run training_data --run_id $BASE_RUN_ID --output_path $SWEEP_DATASET_DIR/threshold_${threshold}.jsonl --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $threshold  --min_gap $DEFAULT_GAP  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS

#         MODEL_NAME="threshold_${threshold}"
#         OUTPUT_DIR=$MODELS_DIR/$MODEL_NAME
#         CONFIG_PATH=$SWEEP_DIR/$MODEL_NAME.yaml

#         python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
#             --learning-rate $BEST_LR_VALUE \
#             --alpha $BEST_ALPHA_VALUE \
#             --beta $BEST_BETA_VALUE \
#             --dataset-path $SWEEP_DATASET_DIR/threshold_${threshold}.jsonl \
#             --output-dir $OUTPUT_DIR \
#             --project-name $MODEL_NAME \
#             --config-path $CONFIG_PATH

#         accelerate launch -m  axolotl.cli.train $CONFIG_PATH
# done

# RUN_IDS=""
# for threshold in 0.5 0.8 1.0; do
#     CURR_RUN_ID="${RUN_IDENTIFIER}_gridsearch_threshold_${threshold}"
#     RUN_IDS="$RUN_IDS $CURR_RUN_ID"
#     ./improver run pipeline --run_id $CURR_RUN_ID  --annotation  --informal --examples 4 --metric length --prompt_id final_test   --split test --model $MODELS_DIR/threshold_${threshold}     --num_blocks 64     --config $GRID_CONFIG
# done

# # select best
# echo "Selecting best model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     $RUN_IDS \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_threshold

BEST_THRESHOLD_MODEL=$(cat $ABLATION_DIR/.best_threshold)
echo "Best Threshold Model: $BEST_THRESHOLD_MODEL"
BEST_THRESHOLD_VALUE=$(echo $BEST_THRESHOLD_MODEL | cut -d'_' -f9)
echo "Best Threshold: $BEST_THRESHOLD_VALUE"

# --- gap ---

# for gap in 0.0 0.25 0.5; do
#         ./improver run training_data --run_id $BASE_RUN_ID --output_path $SWEEP_DATASET_DIR/gap_${gap}.jsonl --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $BEST_THRESHOLD_VALUE  --min_gap $gap  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS

#         MODEL_NAME="gap_${gap}"
#         OUTPUT_DIR=$MODELS_DIR/$MODEL_NAME
#         CONFIG_PATH=$SWEEP_DIR/$MODEL_NAME.yaml

#         python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
#             --learning-rate $BEST_LR_VALUE \
#             --alpha $BEST_ALPHA_VALUE \
#             --beta $BEST_BETA_VALUE \
#             --dataset-path $SWEEP_DATASET_DIR/gap_${gap}.jsonl \
#             --output-dir $OUTPUT_DIR \
#             --project-name $MODEL_NAME \
#             --config-path $CONFIG_PATH

#         accelerate launch -m  axolotl.cli.train $CONFIG_PATH
# done

# RUN_IDS=""
# for gap in 0.0 0.25 0.5; do
#     CURR_RUN_ID="${RUN_IDENTIFIER}_gridsearch_gap_${gap}"
#     RUN_IDS="$RUN_IDS $CURR_RUN_ID"
#     ./improver run pipeline --run_id $CURR_RUN_ID  --annotation  --informal --examples 4 --metric length --prompt_id final_test   --split test --model $MODELS_DIR/gap_${gap}     --num_blocks 64     --config $GRID_CONFIG
# done

# # select best
# echo "Selecting best model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     $RUN_IDS \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_gap

BEST_GAP_MODEL=$(cat $ABLATION_DIR/.best_gap)
echo "Best Gap Model: $BEST_GAP_MODEL"
BEST_GAP_VALUE=$(echo $BEST_GAP_MODEL | cut -d'_' -f9)
echo "Best Gap: $BEST_GAP_VALUE"


# # ============ HARDNESS WEIGHT GRID SEARCH ============
# # This section searches over different hardness_weight values to upweight hard proofs in DPO training

# echo "Starting Hardness Weight Grid Search..."

# for hardness_weight in 1.0 2.0 5.0; do
#     echo "=== Generating training data with hardness_weight=$hardness_weight ==="
    
#     ./improver run training_data --run_id $BASE_RUN_ID \
#         --output_path $SWEEP_DATASET_DIR/hardness_${hardness_weight}.jsonl \
#         --type dpo \
#         --num_invalid $BEST_L_VALUE \
#         --max_champions $BEST_W_VALUE \
#         --filter_threshold $BEST_THRESHOLD_VALUE \
#         --min_gap $BEST_GAP_VALUE \
#         --replay_buffer_split $BEST_REPLAY_VALUE \
#         --replay_type $BEST_REPLAY_TYPE \
#         --prev_run_id $PREV_RUN_IDS \
#         --hardness_weight $hardness_weight

#     MODEL_NAME="hardness_${hardness_weight}"
#     OUTPUT_DIR=$MODELS_DIR/$MODEL_NAME
#     CONFIG_PATH=$SWEEP_DIR/$MODEL_NAME.yaml

#     python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
#         --learning-rate $BEST_LR_VALUE \
#         --alpha $BEST_ALPHA_VALUE \
#         --beta $BEST_BETA_VALUE \
#         --dataset-path $SWEEP_DATASET_DIR/hardness_${hardness_weight}.jsonl \
#         --output-dir $OUTPUT_DIR \
#         --project-name $MODEL_NAME \
#         --config-path $CONFIG_PATH

#     accelerate launch -m axolotl.cli.train $CONFIG_PATH
# done

# echo "Running inference for hardness models..."
# RUN_IDS=""
# for hardness_weight in 1.0 2.0 5.0; do
#     CURR_RUN_ID="${RUN_IDENTIFIER}_gridsearch_hardness_${hardness_weight}"
#     RUN_IDS="$RUN_IDS $CURR_RUN_ID"
#     ./improver run pipeline --run_id $CURR_RUN_ID \
#         --annotation --informal --examples 4 \
#         --metric length --prompt_id final_test \
#         --split test \
#         --model $MODELS_DIR/hardness_${hardness_weight} \
#         --num_blocks 64 \
#         --config $GRID_CONFIG
# done

# # select best hardness model
# echo "Selecting best hardness model..."
# python $SCRIPTS_DIR/select_best_model.py \
#     $RUN_IDS \
#     --evals-path $EVALS_DIR \
#     --models-path $MODELS_DIR \
#     --output-file $ABLATION_DIR/.best_hardness

BEST_HARDNESS_MODEL=$(cat $ABLATION_DIR/.best_hardness)
echo "Best Hardness Model: $BEST_HARDNESS_MODEL"
BEST_HARDNESS_VALUE=$(echo $BEST_HARDNESS_MODEL | grep -oP 'hardness_\K[0-9.]+')
echo "Best Hardness Weight: $BEST_HARDNESS_VALUE"

echo "Hardness Weight Grid Search Complete!"





# ============ Final Training with Best Hyperparameters ===========

# ./improver run training_data --run_id $BASE_RUN_ID --output_path $DATA_DIR/final.jsonl --type dpo --num_invalid $BEST_L_VALUE --max_champions $BEST_W_VALUE --filter_threshold $BEST_THRESHOLD_VALUE  --min_gap $BEST_GAP_VALUE  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS

# MODEL_NAME="IRPO_${METRIC}_iter_2"
# OUTPUT_DIR=/data/user_data/riyaza/saved_models/$MODEL_NAME
# CONFIG_PATH=$CONFIG_DIR/$MODEL_NAME.yaml

# # python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
# #     --learning-rate $BEST_LR_VALUE \
# #     --alpha $BEST_ALPHA_VALUE \
# #     --beta $BEST_BETA_VALUE \
# #     --dataset-path $DATA_DIR/final.jsonl \
# #     --output-dir $OUTPUT_DIR \
# #     --project-name $MODEL_NAME \
# #     --config-path $CONFIG_PATH

# # accelerate launch -m  axolotl.cli.train $CONFIG_PATH

# ./improver run pipeline --run_id ${MODEL_NAME}_train_gap_0.0     --annotation  --informal --examples 4     --metric $METRIC --prompt_id final_train     --split train --model /data/user_data/riyaza/saved_models/length/iter_2/gridsearch/gap_0.0     --num_blocks 64     --config $TEST_CONFIG




# --- W/L ---
W=1
L=4

./improver run training_data --run_id $BASE_RUN_ID --output_path $SWEEP_DATASET_DIR/WL_${W}_${L}.jsonl --type dpo --num_invalid $L --max_champions $W --filter_threshold $DEFAULT_THRESHOLD  --min_gap $DEFAULT_GAP  --replay_buffer_split $BEST_REPLAY_VALUE --replay_type $BEST_REPLAY_TYPE --prev_run_id $PREV_RUN_IDS

MODEL_NAME="WL_${W}_${L}"
OUTPUT_DIR=$MODELS_DIR/$MODEL_NAME

CONFIG_PATH=$SWEEP_DIR/$MODEL_NAME.yaml

python experiments/final/length/make_train_config.py --base-model $BASE_MODEL \
    --learning-rate $DEFAULT_LR \
    --alpha $BEST_ALPHA_VALUE \
    --beta $BEST_BETA_VALUE \
    --dataset-path $SWEEP_DATASET_DIR/WL_${W}_${L}.jsonl \
    --output-dir $OUTPUT_DIR \
    --project-name $MODEL_NAME \
    --config-path $CONFIG_PATH

accelerate launch -m  axolotl.cli.train $CONFIG_PATH

CURR_RUN_ID="${RUN_IDENTIFIER}_gridsearch_WL_${W}_${L}_AGAIN"
RUN_IDS="$RUN_IDS $CURR_RUN_ID"
./improver run pipeline --run_id $CURR_RUN_ID  --annotation  --informal --examples 4 --metric length --prompt_id final_test   --split test --model $MODELS_DIR/WL_${W}_${L}     --num_blocks 64     --config $GRID_CONFIG


