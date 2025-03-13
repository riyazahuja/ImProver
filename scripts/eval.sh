#!/bin/bash

#SBATCH --job-name=improver_eval
#SBATCH --output=logs/improver_eval.out
#SBATCH --error=logs/improver_eval.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G
#SBATCH --exclude=babel-1-31

source $HOME/miniconda3/bin/activate env



# export MODEL_PATH=/data/user_data/riyaza/saved_models/DeepSeek-R1-Distill-Llama-8B_4096
# export MODEL_NAME=Llama-8B

# export MODEL_PATH = /data/user_data/riyaza/saved_models/DeepSeek-R1-Distill-Qwen-7B_full_4096
# export MODEL_NAME=Qwen-7B

# export MODEL_PATH=/data/user_data/riyaza/saved_models/DeepSeek-R1-Distill-Qwen-7B-improverSFT
# export MODEL_NAME=Qwen-7B-ImproverSFT

# export MODEL_PATH=nutPace/Improver-DeepSeek-R1-Distill-Qwen-7B_dpo1
# export MODEL_NAME=Qwen-7B-DPO

# export MODEL_PATH=nutPace/Improver-DeepSeek-R1-Distill-Qwen-7B-full_dpo1
# export MODEL_NAME=Qwen-7B-SFT-DPO



export ANNOTATION=false
export CONTEXT=false
export DATASET_PATH=$HOME/eval_improver/improver/scripts/data/test_set_no_inst.json
export PORT=8000
export BEST_OF_N=64


#vllm serve $MODEL_PATH --served-model-name $MODEL_NAME &
vllm serve $MODEL_PATH --served-model-name $MODEL_NAME &

cd ~/eval_improver/improver

lake build ImProver.improver

echo "Waiting for vLLM server to start..."
until curl -s http://localhost:${PORT}/model_info > /dev/null; do
    sleep 5
done
echo "vLLM server is up and running."


python3 scripts/eval_improver.py $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT

python3 scripts/analyze_data.py $MODEL_NAME

