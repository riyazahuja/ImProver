#!/bin/bash

#SBATCH --job-name=cotrain
#SBATCH --output=logs/cotrain.out
#SBATCH --error=logs/cotrain.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A6000:6
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver


lake build ImProver.efficient.eval_improver

sleep 5


python /home/riyaza/eval_improver/improver/ImProver/offline_cotraining.py /data/user_data/riyaza/saved_models/Q14_conjecturer taterowney/prover_completion_v2 