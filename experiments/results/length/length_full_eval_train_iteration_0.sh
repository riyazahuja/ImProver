#!/bin/bash

#SBATCH --job-name=length_full_eval_train_iteration_0
#SBATCH --output=logs/length_full/length_full_eval_train_iteration_0.out
#SBATCH --error=logs/length_full/length_full_eval_train_iteration_0.err
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:7
#SBATCH --mem=150G

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export PYTHONUNBUFFERED=1

cd /home/riyaza/eval_improver/improver
lake build eval_improver
sleep 5

export run_id="length_full_eval_train_iteration_0"
export prompt_id="final_final_train"
export split="train"
export model="deepseek-ai/DeepSeek-Prover-V2-7B"

./improver run pipeline --run_id $run_id --prompt_id $prompt_id --split $split --model $model --config /home/riyaza/eval_improver/improver/experiments/results/length/base_infer_a6000_temp.yaml
