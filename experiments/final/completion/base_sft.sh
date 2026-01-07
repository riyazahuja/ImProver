#!/bin/bash
#SBATCH --job-name=length_iter_1
#SBATCH --output=logs/final/length/length_iter_1.out
#SBATCH --error=logs/final/length/length_iter_1.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L40S:6
#SBATCH --mem=250G
#SBATCH --exclude=babel-v9-28

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


cd /home/$USER/ImProver
# lake build eval_improver
# sleep 5


# axolotl train /home/trowney/ImProver/experiments/final/completion/base_sft_conf.yaml


# python experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    --adapter /data/user_data/trowney/saved_models/completion_lora     --output /data/user_data/trowney/saved_models/completion

# rm -rf /data/user_data/trowney/saved_models/completion_lora

./improver run pipeline --run_id SFT_cmp_test     --metric completion --prompt_id final_test     --split test --model /data/user_data/trowney/saved_models/completion     --num_blocks 32     --config experiments/final/test_eval.yaml