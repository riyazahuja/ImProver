#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/final/dependency/baselines.out
#SBATCH --error=logs/final/dependency/baselines.err
#SBATCH --cpus-per-task=128
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
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

# get IRPO data


./improver run training_data --run_id base_length_train --output_path /home/riyaza/eval_improver/improver/experiments/final/length/data/IRPO_length_iter_1.jsonl     --type dpo --num_invalid 4 --max_champions 4 --filter_threshold 0.8  --min_gap 0  --replay_buffer_split 0.2 --replay_type replace

    
# # train IRPO model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/length/configs/IRPO_length_iter_1.yaml


# # eval IRPO model on test set

./improver run pipeline --run_id IRPO_length_iter_1_test     --annotation  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1     --num_blocks 64     --config experiments/final/test_eval.yaml

