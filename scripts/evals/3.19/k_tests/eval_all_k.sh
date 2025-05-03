#!/bin/bash

#SBATCH --job-name=all_k
#SBATCH --output=logs/3.19/all_k.out
#SBATCH --error=logs/3.19/all_k.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export MODEL_PATH=/data/user_data/riyaza/saved_models/DeepSeek-R1-Distill-Qwen-7B-improverSFT
export MODEL_NAME=Qwen-7B-ALL


export ANNOTATION=true
export CONTEXT=true
export DATASET_PATH=$HOME/eval_improver/improver/scripts/data/test/test_set_no_inst.json
export PORT=8005
export BEST_OF_N=32



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



export RAG=3

python3 scripts/eval_improver.py $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG $RAG

export RAG=7

python3 scripts/eval_improver.py $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG $RAG

export RAG=11

python3 scripts/eval_improver.py $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG $RAG

export RAG=15

python3 scripts/eval_improver.py $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG $RAG

