#!/bin/bash

#SBATCH --job-name=r1_qwen7b_32k_full_weighted
#SBATCH --output=logs/EI_tests/r1_qwen7b_32k_full_weighted.out
#SBATCH --error=logs/EI_tests/r1_qwen7b_32k_full_weighted.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:7
#SBATCH --mem=150G

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONUNBUFFERED=1

cd /home/riyaza/eval_improver/improver
lake build eval_improver
sleep 5

export run_id="r1_qwen7b_32k_full_weighted"
export prompt_id="final_final_test"
export split="test"
export model="/data/user_data/riyaza/saved_models/r1_qwen7b_32k_full_weighted"

./improver run pipeline --run_id $run_id --prompt_id $prompt_id --split $split --model $model --config /home/riyaza/eval_improver/improver/experiments/results/length/base_infer_a6000_temp.yaml
