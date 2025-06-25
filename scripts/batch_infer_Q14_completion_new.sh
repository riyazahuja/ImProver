#!/bin/bash

#SBATCH --job-name=tuned_Q14_infer
#SBATCH --output=logs/tuned_Q14_infer.out
#SBATCH --error=logs/tuned_Q14_infer.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=150G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/inference.py completion /home/riyaza/eval_improver/improver/train/data/tt_split_data_iter0.json prompts_test --split test --cpus 12 --gpus 6 --n 32 --model /data/user_data/riyaza/saved_models/Q14-completion_inf/checkpoint-1364
