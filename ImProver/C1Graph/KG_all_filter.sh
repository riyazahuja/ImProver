#!/bin/bash

#SBATCH --job-name=KG_all
#SBATCH --output=logs/KG_all.out
#SBATCH --error=logs/KG_all.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A6000:8
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

sleep 5

python /home/riyaza/eval_improver/improver/ImProver/C1Graph/heuristic_filter.py /home/riyaza/eval_improver/improver/data/tt_split_data.json --KG_dir KG_ALL --cpus 12 --gpus 8 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
