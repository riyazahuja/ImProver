#!/bin/bash

#SBATCH --job-name=eval_0
#SBATCH --output=logs/eval_len-iter0-14.out
#SBATCH --error=logs/eval_len-iter0-14.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

lake build ImProver.efficient.eval_improver

sleep 5

python /home/riyaza/eval_improver/improver/ImProver/efficient/eval_improver.py RUN_20250603_142135 --cpus 12
