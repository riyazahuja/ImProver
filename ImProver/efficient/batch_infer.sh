#!/bin/bash

#SBATCH --job-name=infer_iter0-q
#SBATCH --output=logs/infer_iter0_lenq.out
#SBATCH --error=logs/infer_iter0_lenq.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/efficient/inference.py length /home/riyaza/eval_improver/improver/scripts/data/tt_split_data_iter0.json --cpus 12 --gpus 6 --examples 2 --n 32 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
