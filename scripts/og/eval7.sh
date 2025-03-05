#!/bin/bash

#SBATCH --job-name=improver_eval
#SBATCH --output=logs/improver_eval7_2.out
#SBATCH --error=logs/improver_eval7_2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G
#SBATCH --exclude=babel-1-31

source $HOME/miniconda3/bin/activate .venv10

export MODEL_PATH=nutPace/Improver-DeepSeek-R1-Distill-Qwen-7B_full
export MODEL_NAME=Qwen-7B-short

#vllm serve $MODEL_PATH --served-model-name $MODEL_NAME &
vllm serve $MODEL_PATH --download-dir /data/user_data/riyaza/HF/big_one --served-model-name $MODEL_NAME  &

cd ~/eval_improver/ImProver

lake build scripts.verifier

echo "Waiting for vLLM server to start..."
until curl -s http://localhost:8000/model_info > /dev/null; do
    sleep 5
done
echo "vLLM server is up and running."


python3 scripts/eval.py 60 $MODEL_NAME 8000



