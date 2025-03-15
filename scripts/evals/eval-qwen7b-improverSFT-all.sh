#!/bin/bash

#SBATCH --job-name=all
#SBATCH --output=logs/all.out
#SBATCH --error=logs/all.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env



# export MODEL_PATH=/data/user_data/riyaza/saved_models/DeepSeek-R1-Distill-Llama-8B_4096
# export MODEL_NAME=Llama-8B

# export MODEL_PATH = /data/user_data/riyaza/saved_models/DeepSeek-R1-Distill-Qwen-7B_full_4096
# export MODEL_NAME=Qwen-7B

export MODEL_PATH=/data/user_data/riyaza/saved_models/DeepSeek-R1-Distill-Qwen-7B-improverSFT
export MODEL_NAME=Qwen-7B-ImproverSFT-All

# export MODEL_PATH=nutPace/Improver-DeepSeek-R1-Distill-Qwen-7B_dpo1
# export MODEL_NAME=Qwen-7B-DPO

# export MODEL_PATH=nutPace/Improver-DeepSeek-R1-Distill-Qwen-7B-full_dpo1
# export MODEL_NAME=Qwen-7B-SFT-DPO



export ANNOTATION=true
export CONTEXT=true
export RAG=7
export DATASET_PATH=$HOME/eval_improver/improver/scripts/data/test/test_set_no_inst.json
export PORT=8001
export BEST_OF_N=64


#vllm serve $MODEL_PATH --served-model-name $MODEL_NAME &
echo "Starting vLLM server..."
vllm serve $MODEL_PATH --served-model-name $MODEL_NAME --port $PORT &


# Start Ollama server
echo "Starting Ollama server..."
ollama serve &


cd ~/eval_improver/improver
echo "Lake building"

lake build ImProver.improver



echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/health > /dev/null 2>&1; do
    sleep 5
    echo "Still waiting for Ollama..."
done
echo "Ollama server is up and running."



echo "Waiting for vLLM server to start..."
until curl -s http://localhost:${PORT}/model_info > /dev/null; do
    sleep 5
    echo "Still waiting for vLLM..."

done
echo "vLLM server is up and running."





python3 scripts/eval_improver.py $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG



