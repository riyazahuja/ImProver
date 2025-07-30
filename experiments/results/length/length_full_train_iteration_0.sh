#!/bin/bash

#SBATCH --job-name=length_full_train_iteration_0
#SBATCH --output=logs/length_full/length_full_train_iteration_0.out
#SBATCH --error=logs/length_full/length_full_train_iteration_0.err
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A100_80GB:6
#SBATCH --mem=150G

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/riyaza/HF"
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

export base_model="deepseek-ai/DeepSeek-Prover-V2-7B"
export model_name="length_full_iteration_0"
export datasets="[{"path": "evals/length_full_eval_train_iteration_0/analysis/BoN/train.jsonl", "type": "alpaca"}]"
export output_dir="/data/user_data/riyaza/saved_models/length"
export hub_model_id="riyazahuja/length_full_iteration_0"
export wandb_project="length_full_iteration_0"

cd /home/riyaza/eval_improver/improver

accelerate launch -m     axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/results/length/base_train_hyper.yaml     --base_model $base_model     --datasets $datasets     --output_dir $output_dir     --hub_model_id $hub_model_id     --wandb_project $wandb_project
