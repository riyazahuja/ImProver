#!/bin/bash

#SBATCH --job-name=informalize
#SBATCH --output=logs/informalize.out
#SBATCH --error=logs/informalize.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:A6000:6



source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/C1Graph/c3/informalize.py /home/riyaza/eval_improver/improver/data/tt_split_data.json --KG_dir KG3 --cpus 12