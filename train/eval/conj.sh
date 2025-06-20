#!/bin/bash

#SBATCH --job-name=conjecturer2
#SBATCH --output=logs/conjecturer2.out
#SBATCH --error=logs/conjecturer2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --mem=150G
#SBATCH --exclude=babel-0-31,babel-5-31


source $HOME/miniconda3/bin/activate env

export HF_HOME="/data/user_data/riyaza/HF"

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --main-process-port=29501 -m \
    axolotl.cli.train /home/riyaza/eval_improver/improver/scripts/train/conj_inf.yml \
    --deepspeed /home/riyaza/deepspeed_configs/zero3_bf16_cpuoffload_params.json
