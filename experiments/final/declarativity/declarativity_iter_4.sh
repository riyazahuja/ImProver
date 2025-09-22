#!/bin/bash
#SBATCH --job-name=declarativity_iter_4
#SBATCH --output=logs/final/declarativity/declarativity_iter_4.out
#SBATCH --error=logs/final/declarativity/declarativity_iter_4.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=250G
#SBATCH --exclude=babel-15-36,babel-1-23

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/$USER/HF"
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
mkdir -p /data/user_data/$USER/ray_tmp
export RAY_TMPDIR=/data/user_data/$USER/ray_tmp

export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576


cd /home/$USER/eval_improver/improver
lake build eval_improver
sleep 5


# eval prev iter model on train set

./improver run pipeline --run_id IRPO_declarativity_iter_3_train     --annotation --context 10  --informal --examples 4     --metric declarativity --prompt_id /home/$USER/eval_improver/improver/prompts/final_train     --split train --model /data/user_data/riyaza/saved_models/IRPO_declarativity_iter_3     --num_blocks 512     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml


# first get wSFT data

./improver run training_data --run_id IRPO_declarativity_iter_3_train --tau 0.5 --output_path /home/riyaza/eval_improver/improver/experiments/final/declarativity/data/wSFT_declarativity_iter_4.jsonl     --type weighted_sft --epsilon 0.1 --variance_threshold 0.8 --filter_threshold 1.1     --replay_buffer_split 0.4 --replay_type replace --prev_run_id base_declarativity_train,IRPO_declarativity_iter_1_train,IRPO_declarativity_iter_2_train

    
# convert wSFT data

python /home/$USER/eval_improver/improver/experiments/final/preprocess_weights.py     /home/riyaza/eval_improver/improver/experiments/final/declarativity/data/wSFT_declarativity_iter_4.jsonl /home/riyaza/eval_improver/improver/experiments/final/declarativity/data/wSFT_declarativity_iter_4

# train wSFT model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/declarativity/configs/wSFT_declarativity_iter_4.yaml

# merge wSFT LoRA with base to get final wSFT model

python /home/$USER/eval_improver/improver/experiments/final/merge.py     --ref /data/user_data/riyaza/saved_models/IRPO_declarativity_iter_3     --adapter /data/user_data/riyaza/saved_models/wSFT_declarativity_iter_4_lora     --output /data/user_data/riyaza/saved_models/wSFT_declarativity_iter_4

# eval wSFT model on test set

./improver run pipeline --run_id wSFT_declarativity_iter_4_test     --annotation --context 10  --informal --examples 4     --metric declarativity --prompt_id /home/$USER/eval_improver/improver/prompts/final_test     --split test --model /data/user_data/riyaza/saved_models/wSFT_declarativity_iter_4     --num_blocks 64     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml

# get IRPO data



./improver run training_data --run_id IRPO_declarativity_iter_3_train --output_path /home/riyaza/eval_improver/improver/experiments/final/declarativity/data/IRPO_declarativity_iter_4.jsonl     --type dpo --num_invalid -1 --max_champions -1 --filter_threshold 1.1     --replay_buffer_split 0.4 --replay_type replace --prev_run_id base_declarativity_train,IRPO_declarativity_iter_1_train,IRPO_declarativity_iter_2_train

    
# train IRPO model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/declarativity/configs/IRPO_declarativity_iter_4.yaml


# eval IRPO model on test set

./improver run pipeline --run_id IRPO_declarativity_iter_4_test     --annotation --context 10  --informal --examples 4     --metric declarativity --prompt_id /home/$USER/eval_improver/improver/prompts/final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_declarativity_iter_4     --num_blocks 64     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml

