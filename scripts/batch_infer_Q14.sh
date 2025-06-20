#!/bin/bash

#SBATCH --job-name=Q14_infer
#SBATCH --output=logs/Q14_infer.out
#SBATCH --error=logs/Q14_infer.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/efficient/inference.py conjecturer /home/riyaza/eval_improver/improver/scripts/data/tt_split_data.json --split test --prompts_dir prompts_test/ --cpus 12 --gpus 6 --n 32 --model /data/user_data/riyaza/saved_models/Q14_conjecturer
