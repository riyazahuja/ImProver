#!/bin/bash

#SBATCH --job-name=r1_qwen7b_32k_dpo
#SBATCH --output=logs/EI_tests/r1_qwen7b_32k_dpo.out
#SBATCH --error=logs/EI_tests/r1_qwen7b_32k_dpo.err
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=150G
#SBATCH --exclude=babel-15-36,babel-1-23

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/riyaza/HF"
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5


cd /home/riyaza/eval_improver/improver



sleep 5
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/results/length/EI_tests/dpo.yaml --no_save_optimizer_state

