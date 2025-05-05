#!/bin/bash

#SBATCH --job-name=b_Tr0
#SBATCH --output=logs/base_train0.out
#SBATCH --error=logs/base_train0.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env

export METRIC=declarativity

# export MODEL_PATH=/data/user_data/riyaza/saved_models/ImProver_BASE_iter0/checkpoint-210
export MODEL_PATH=/data/user_data/riyaza/HF/models--Qwen--Qwen-7B
export MODEL_NAME=BASE


export ANNOTATION=true
export CONTEXT=true
export RAG=7
export DATASET_PATH=$HOME/eval_improver/improver/scripts/data/tt_split_data.json
export PORT=8000
export BEST_OF_N=10


export DOWNLOAD_PATH=/data/user_data/riyaza/HF

echo "Starting vLLM server..."
vllm serve $MODEL_PATH --served-model-name $MODEL_NAME \
    --port $PORT \
    --tensor-parallel-size 2 \
    &


cd ~/eval_improver/improver
echo "Lake building"

lake build ImProver.improver





echo "Waiting for vLLM server to start..."
until curl -s http://localhost:${PORT}/model_info > /dev/null; do
    sleep 5
    echo "Still waiting for vLLM..."

done
echo "vLLM server is up and running."





python3 scripts/eval_improver_async.py $METRIC $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG "iter0_train"



