#!/bin/bash

#SBATCH --job-name=KG_all
#SBATCH --output=logs/KG_all.out
#SBATCH --error=logs/KG_all.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver

lake build ImProver.C1Graph.getKG_2

sleep 5

python /home/riyaza/eval_improver/improver/ImProver/C1Graph/getKG_2.py /home/riyaza/eval_improver/improver/data/tt_split_data.json --output_dir KG_ALL --cpus 12
