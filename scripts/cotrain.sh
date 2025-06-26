#!/bin/bash

#SBATCH --job-name=cotrain
#SBATCH --output=logs/cotrain.out
#SBATCH --error=logs/cotrain.err
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:A6000:7
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver


lake build ImProver.eval_improver

sleep 5


python /home/riyaza/eval_improver/improver/ImProver/cotraining/offline_cotraining.py /data/user_data/riyaza/saved_models/Q14_conjecturer_inf/checkpoint-434 /data/user_data/riyaza/saved_models/Q14-completion_inf/checkpoint-1364