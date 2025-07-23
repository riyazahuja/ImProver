#!/bin/bash

#SBATCH --job-name=prompts
#SBATCH --output=logs/prompts.out
#SBATCH --error=logs/prompts.err
#SBATCH --cpus-per-task=25
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:7
#SBATCH --mem=150G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver

lake build get_prompts
sleep 5


./improver prompts get --config /home/riyaza/eval_improver/improver/configs/prompts.yaml