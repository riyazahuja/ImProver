#!/bin/bash

#SBATCH --job-name=improver
#SBATCH --output=logs/improver.out
#SBATCH --error=logs/improver.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --main-process-port=29501 -m \
    axolotl.cli.train /home/riyaza/eval_improver/improver/scripts/train/Q14.yml \
    --deepspeed /home/riyaza/deepspeed_configs/zero3_bf16_cpuoffload_params.json