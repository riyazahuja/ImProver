#!/bin/bash

#SBATCH --job-name=informalize
#SBATCH --output=logs/informalize.out
#SBATCH --error=logs/informalize.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:A6000:6



source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

python ImProver/KG/informalize_direct.py /home/riyaza/eval_improver/improver/train/data/tt_split_data.json prompts --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --cpus 12 --gpus 6