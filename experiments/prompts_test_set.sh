#!/bin/bash

#SBATCH --job-name=prompts_informalize_test
#SBATCH --output=logs/prompts_informalize_test.out
#SBATCH --error=logs/prompts_informalize_test.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --mem=150G
#SBATCH --exclude=babel-15-36,babel-1-23
#SBATCH --dependency=afterok:5405343



source $HOME/miniconda/bin/activate venv
export HF_HOME="/data/user_data/shivansg/HF"
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5


cd $HOME/ImProver

export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
ulimit -s 65532
# accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/results/length/replay_buffers/wSFT_mark_big.yaml --no_save_optimizer_state
./improver prompts get --dataset_path /home/shivansg/ImProver/train/data/train/final_dataset_fixed.json --rag_id rag_test_set --split test --cpus 64