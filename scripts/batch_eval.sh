#!/bin/bash

#SBATCH --job-name=base_eval
#SBATCH --output=logs/base_eval.out
#SBATCH --error=logs/base_eval.err
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

lake build eval_improver

sleep 5

python /home/riyaza/eval_improver/improver/ImProver/eval_improver.py Q14-cmp --cpus 24
