#!/bin/bash

#SBATCH --job-name=kg
#SBATCH --output=logs/kg.out
#SBATCH --error=logs/kg.err
#SBATCH --cpus-per-task=25
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=150G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"

cd ~/eval_improver/improver


./improver kg full --config /home/riyaza/eval_improver/improver/configs/kg.yaml