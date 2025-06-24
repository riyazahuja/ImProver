#!/bin/bash

#SBATCH --job-name=completion_base
#SBATCH --output=logs/completion_base.out
#SBATCH --error=logs/completion_base.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:3
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/inference.py completion /home/riyaza/eval_improver/improver/train/data/tt_split_data.json prompts_test --split test --cpus 12 --gpus 3 --n 32 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B