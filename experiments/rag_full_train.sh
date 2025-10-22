#!/bin/bash

#SBATCH --job-name=rag_full_train
#SBATCH --output=logs/rag_full_train.out
#SBATCH --error=logs/rag_full_train.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --mem=150G
#SBATCH --exclude=babel-15-36,babel-1-23



source $HOME/miniconda/bin/activate venv
export HF_HOME="/data/user_data/shivansg/HF"
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5


cd $HOME/ImProver

ulimit -s 65532

export ACCELERATE_USE_REENTRANT_CHECKPOINT=0

# accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/results/length/replay_buffers/wSFT_mark_big.yaml --no_save_optimizer_state
./improver rag build --dataset_path /home/shivansg/ImProver/data/final_dataset_decontaminated.json --rag_id rag_full_train --max_depth 1 --split train --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cpus 64 --gpus 8 --num_blocks 512 --config /home/shivansg/ImProver/experiments/final/test_eval.yaml