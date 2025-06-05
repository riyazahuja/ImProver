#!/bin/bash

#SBATCH --job-name=DS2_infer
#SBATCH --output=logs/DS2_infer.out
#SBATCH --error=logs/DS2_infer.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/efficient/inference.py length /home/riyaza/eval_improver/improver/scripts/data/tt_split_data_iter0.json --cpus 12 --gpus 6 --examples 2 --n 32 --model deepseek-ai/DeepSeek-Prover-V2-7B