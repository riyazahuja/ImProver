#!/bin/bash

#SBATCH --job-name=Q14_eval
#SBATCH --output=logs/Q14_eval.out
#SBATCH --error=logs/Q14_eval.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

lake build ImProver.efficient.eval_improver

sleep 5

python /home/riyaza/eval_improver/improver/ImProver/efficient/eval_improver.py RUN_20250604_190740 --cpus 12
