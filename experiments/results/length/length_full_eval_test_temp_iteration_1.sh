#!/bin/bash

#SBATCH --job-name=length_full_eval_test_temp_iteration_1_base
#SBATCH --output=logs/length_full/length_full_eval_test_temp_iteration_1_base.out
#SBATCH --error=logs/length_full/length_full_eval_test_temp_iteration_1_base.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=150G

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONUNBUFFERED=1

cd /home/riyaza/eval_improver/improver
lake build eval_improver
sleep 5

export run_id="length_full_eval_test_temp_iteration_1_base"
export prompt_id="final_final_test"
export split="test"
export model="/data/user_data/riyaza/saved_models/length_full_iteration_1_base"

./improver run pipeline --run_id $run_id --prompt_id $prompt_id --split $split --model $model --config /home/riyaza/eval_improver/improver/experiments/results/length/base_infer_a6000_smaller.yaml
