#!/bin/bash

#SBATCH --job-name=filter
#SBATCH --output=logs/filter.out
#SBATCH --error=logs/filter.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A6000:6
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

sleep 5

python /home/riyaza/eval_improver/improver/ImProver/C1Graph/heuristic_filter2.py --KG_dir KG3_i0 --cpus 12 --gpus 6 --n 3 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
