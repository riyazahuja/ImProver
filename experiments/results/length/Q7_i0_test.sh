#!/bin/bash

#SBATCH --job-name=Q7_i0_train
#SBATCH --output=logs/EI_tests/Q7_i0_train.out
#SBATCH --error=logs/EI_tests/Q7_i0_train.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=150G

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONUNBUFFERED=1

cd /home/riyaza/eval_improver/improver
lake build eval_improver
sleep 5

export run_id="Q7_i0_test"
export prompt_id="really_final_rag_test2"
export split="test"
export model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

./improver run pipeline --run_id $run_id --prompt_id $prompt_id --split $split --model $model --config /home/riyaza/eval_improver/improver/experiments/results/length/base_infer_a6000_smaller.yaml
