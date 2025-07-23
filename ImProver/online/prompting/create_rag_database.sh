#!/bin/bash

#SBATCH --job-name=create_rag_database
#SBATCH --output=/home/trowney/ImProver/logs/create_rag_database.out
#SBATCH --error=/home/trowney/ImProver/logs/create_rag_database.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2

source ~/miniconda3/bin/activate ImProver_env

export HF_HOME=/data/user_data/$USER/HF
cd ~/ImProver/ImProver/online/prompting

python create_rag_database.py
# python rag_batched.py '{"queries": [{"module": "Carleson.Antichain.AntichainOperator", "name": "antichain_operator_le_volume"}], "k": 5, "imports": ["Mathlib.Tactic.Ring"]}' --prompt_id final_final_train