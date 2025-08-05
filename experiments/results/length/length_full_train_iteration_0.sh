#!/bin/bash

#SBATCH --job-name=length_full_train_iteration_1_base
#SBATCH --output=logs/length_full/length_full_train_iteration_1_base.out
#SBATCH --error=logs/length_full/length_full_train_iteration_1_base.err
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=150G
#SBATCH --exclude=babel-15-36,babel-1-23

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/riyaza/HF"
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

export base_model="deepseek-ai/DeepSeek-Prover-V2-7B"
export model_name="length_full_iteration_1"
# export datasets="[{\"path\": \"/home/riyaza/eval_improver/improver/evals/ablation_model_train_DS2/analysis/BoN/train.jsonl\", \"type\": \"alpaca\"}]"
export output_dir="/data/user_data/riyaza/saved_models/length"
export hub_model_id="riyazahuja/length_full_iteration_1"
export wandb_project="length_full_iteration_1"

cd /home/riyaza/eval_improver/improver

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/results/length/base_train_hyper.yaml  
