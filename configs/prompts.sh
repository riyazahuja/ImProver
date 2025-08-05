#!/bin/bash

#SBATCH --job-name=prompts
#SBATCH --output=logs/prompts.out
#SBATCH --error=logs/prompts.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=200G
    

source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG            # verbose compile log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export PYTHONUNBUFFERED=1   

cd ~/eval_improver/improver

lake build get_prompts
sleep 5
ulimit -s 65536

./improver prompts get --config /home/riyaza/eval_improver/improver/configs/prompts.yaml