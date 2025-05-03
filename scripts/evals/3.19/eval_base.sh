#!/bin/bash

#SBATCH --job-name=base
#SBATCH --output=logs/3.19/base.out
#SBATCH --error=logs/3.19/base.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export MODEL_PATH=/data/user_data/riyaza/saved_models/ImProver_BASE_iter0/checkpoint-210
export MODEL_NAME=Qwen-7B-BASE


export ANNOTATION=false
export CONTEXT=false
export RAG=0
export DATASET_PATH=$HOME/eval_improver/improver/scripts/data/test/test_set_no_inst.json
export PORT=8000
export BEST_OF_N=64



echo "Starting vLLM server..."
vllm serve $MODEL_PATH --served-model-name $MODEL_NAME --port $PORT &


cd ~/eval_improver/improver
echo "Lake building"

lake build ImProver.improver





echo "Waiting for vLLM server to start..."
until curl -s http://localhost:${PORT}/model_info > /dev/null; do
    sleep 5
    echo "Still waiting for vLLM..."

done
echo "vLLM server is up and running."





python3 scripts/eval_improver_async.py $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG "base_iter0"



