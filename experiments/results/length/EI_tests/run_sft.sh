#!/bin/bash

#SBATCH --job-name=sft_i1
#SBATCH --output=logs/EI_tests/sft_i1.out
#SBATCH --error=logs/EI_tests/sft_i1.err
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

./improver run training_data [RUN_ID]

sleep 5

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/results/length/EI_tests/sft.yaml
