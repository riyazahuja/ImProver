#!/bin/bash

#SBATCH --job-name=Q14_infer
#SBATCH --output=logs/Q14_infer.out
#SBATCH --error=logs/Q14_infer.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/efficient/inference.py completion /home/riyaza/eval_improver/improver/scripts/data/tt_split_data_iter0.json --split test --cpus 12 --gpus 8 --n 32 --model taterowney/prover_completion_v2
