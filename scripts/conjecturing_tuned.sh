#!/bin/bash

#SBATCH --job-name=conjecturing_tuned
#SBATCH --output=logs/conjecturing_tuned.out
#SBATCH --error=logs/conjecturing_tuned.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/efficient/inference.py conjecturer /home/riyaza/eval_improver/improver/scripts/data/tt_split_data.json --split test --prompts_dir prompts_test/ --cpus 12 --gpus 6 --n 32 --model /data/user_data/riyaza/saved_models/Q7_conjecturer/checkpoint-504