#!/bin/bash

#SBATCH --job-name=DS2_infer
#SBATCH --output=logs/DS2_infer.out
#SBATCH --error=logs/DS2_infer.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/efficient/inference.py length /home/riyaza/eval_improver/improver/scripts/data/tt_split_data_iter0.json --split test --cpus 12 --gpus 8 --examples 2 --n 32 --model /data/user_data/riyaza/saved_models/DS2_lora_0_merged