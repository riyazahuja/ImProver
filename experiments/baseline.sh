#!/bin/bash

#SBATCH --job-name=baseline_eval
#SBATCH --output=logs/final/baseline_eval.out
#SBATCH --error=logs/final/baseline_eval.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=150G





source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/trowney/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONUNBUFFERED=1
mkdir -p /data/user_data/$USER/ray_tmp
export RAY_TMPDIR=/data/user_data/$USER/ray_tmp


cd /home/trowney/ImProver
lake build eval_improver
sleep 5


export prompt_id="final_test"
export split="test"
export model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export num_blocks=64



./improver run pipeline --run_id "baseline_read" --metric "readability" --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/trowney/ImProver/experiments/final/test_eval.yaml



