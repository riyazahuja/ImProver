#!/bin/bash

#SBATCH --job-name=informalize
#SBATCH --output=logs/informalize.out
#SBATCH --error=logs/informalize.err
#SBATCH --cpus-per-task=25
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=150G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver


./improver prompts informalize --config /home/riyaza/eval_improver/improver/configs/prompts.yaml