#!/bin/bash

#SBATCH --job-name=kg
#SBATCH --output=logs/kg2.out
#SBATCH --error=logs/kg2.err
#SBATCH --cpus-per-task=25
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=150G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG            # verbose compile log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONUNBUFFERED=1   
cd ~/eval_improver/improver


./improver kg full --config /home/riyaza/eval_improver/improver/configs/kg2.yaml