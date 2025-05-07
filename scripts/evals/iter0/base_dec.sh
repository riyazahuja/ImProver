#!/bin/bash

#SBATCH --job-name=d_base_i0
#SBATCH --output=logs/d_b_i0.out
#SBATCH --error=logs/d_b_i0.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env

export METRIC=declarativity

export MODEL_PATH="deepseek-ai/DeepSeek-Prover-V2-7B"
export MODEL_NAME=Deepseek-v2-decl


export ANNOTATION=false
export CONTEXT=false
export RAG=0
export DATASET_PATH=$HOME/eval_improver/improver/scripts/data/tt_split_data.json
export PORT=8000
export BEST_OF_N=32


export DOWNLOAD_PATH=/data/user_data/riyaza/HF
export HF_HOME="/data/user_data/riyaza/HF"


cd ~/eval_improver/improver
echo "Lake building"

lake build ImProver.improver

echo "Starting vLLM server..."
vllm serve $MODEL_PATH --served-model-name $MODEL_NAME \
    --port $PORT --tensor-parallel-size 2 &


echo "Waiting for vLLM server to start..."
until curl -s http://localhost:${PORT}/v1/models > /dev/null; do
    sleep 5
    echo "Still waiting for vLLM..."

done
echo "vLLM server is up and running."





python3 scripts/eval_improver_async.py $METRIC $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG "BASE"



