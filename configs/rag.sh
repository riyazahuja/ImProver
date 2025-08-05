#!/bin/bash

#SBATCH --job-name=rag
#SBATCH --output=logs/build_rag.out
#SBATCH --error=logs/build_rag.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=150G
    

source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG            # verbose compile log
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export PYTHONUNBUFFERED=1   

cd ~/eval_improver/improver

lake build preprocess_rag
sleep 5
ulimit -s 65536

./improver rag informalize --config /home/riyaza/eval_improver/improver/configs/rag.yaml