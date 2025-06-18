#!/bin/bash

#SBATCH --job-name=conjecturing_base
#SBATCH --output=logs/conjecturing_base.out
#SBATCH --error=logs/conjecturing_base.err
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/efficient/inference.py conjecturer /home/riyaza/eval_improver/improver/scripts/data/tt_split_data.json --split test --prompts_dir prompts_test/ --cpus 4 --gpus 2 --n 32 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B