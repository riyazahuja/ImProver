#!/bin/bash

#SBATCH --job-name=inf_conjecturer
#SBATCH --output=logs/inf_conjecturer.out
#SBATCH --error=logs/inf_conjecturer.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A100_80GB:8
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env

export HF_HOME="/data/user_data/riyaza/HF"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --main-process-port=29501 -m \
    axolotl.cli.train /home/riyaza/eval_improver/improver/scripts/train/conj_inf.yml \
    --deepspeed /home/riyaza/deepspeed_configs/zero3_bf16_cpuoffload_params.json
