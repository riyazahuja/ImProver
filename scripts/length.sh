#!/bin/bash

#SBATCH --job-name=Q14_len
#SBATCH --output=logs/Q14_len.out
#SBATCH --error=logs/Q14_len.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --mem=150G


# --gres=gpu:A6000:8

source $HOME/miniconda3/bin/activate ImProver_env


export HF_HOME="/data/user_data/$USER/HF"

cd ~/ImProver

# ./improver run pipeline --gpus 8 --n 32 length ./train/data/train/final_dataset.json final_train
./improver run eval RUN_20250710_110154
# ./improver run analysis RUN_20250710_110154
