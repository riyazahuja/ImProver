#!/bin/bash

#SBATCH --job-name=inf_completion
#SBATCH --output=logs/inf_completion.out
#SBATCH --error=logs/inf_completion.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:7
#SBATCH --mem=160G
#SBATCH --exclude=babel-1-27,babel-4-37



source $HOME/miniconda3/bin/activate env
export NCCL_DEBUG=INFO

export HF_HOME="/data/user_data/riyaza/HF"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

accelerate launch --main-process-port=29501 -m \
    axolotl.cli.train /home/riyaza/eval_improver/improver/train/configs/Q14_inf.yml \
    --deepspeed /home/riyaza/deepspeed_configs/zero3_bf16_cpuoffload_params.json
