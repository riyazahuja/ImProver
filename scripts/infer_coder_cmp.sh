#!/bin/bash

#SBATCH --job-name=coder_cmp
#SBATCH --output=logs/coder_cmp.out
#SBATCH --error=logs/coder_cmp.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=150G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/inference.py completion /home/riyaza/eval_improver/improver/train/data/tt_split_data.json prompts_reformat_train --split train --cpus 12 --gpus 6 --n 32 --model Qwen/Qwen2.5-Coder-7B-Instruct
