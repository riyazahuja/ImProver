#!/bin/bash

#SBATCH --job-name=create_rag_database
#SBATCH --output=logs/create_rag_database.out
#SBATCH --error=logs/create_rag_database.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2

source ~/miniconda3/bin/activate env

export HF_HOME=/data/user_data/$USER/HF
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1
cd ~/eval_improver/improver

python /home/riyaza/eval_improver/improver/ImProver/online/prompting/create_rag_database.py
# python rag_batched.py '{"queries": [{"module": "Carleson.Antichain.AntichainOperator", "name": "antichain_operator_le_volume"}], "k": 5, "imports": ["Mathlib.Tactic.Ring"]}' --prompt_id final_final_train