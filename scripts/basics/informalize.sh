#!/bin/bash

#SBATCH --job-name=informalize
#SBATCH --output=logs/informalize.out
#SBATCH --error=logs/informalize.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=150G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/KG/informalize.py /home/riyaza/eval_improver/improver/train/data/train/final_dataset.json final_train --split train --cpus 12 --gpus 8 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
