#!/bin/bash

#SBATCH --job-name=prompt_decl
#SBATCH --output=logs/prompt_decl.out
#SBATCH --error=logs/prompt_decl.err
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

lake build ImProver.efficient.get_prompts

sleep 5

python /home/riyaza/eval_improver/improver/ImProver/efficient/get_prompts.py declarativity /home/riyaza/eval_improver/improver/scripts/data/tt_split_data.json --cpus 24
