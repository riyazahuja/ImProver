#!/bin/bash

#SBATCH --job-name=prompts_train
#SBATCH --output=logs/prompts_train.out
#SBATCH --error=logs/prompts_train.err
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --mem=150G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

lake build get_prompts

sleep 5

python /home/riyaza/eval_improver/improver/ImProver/get_prompts.py /home/riyaza/eval_improver/improver/train/data/train/final_dataset.json --prompt_id final-train --cpus 24
