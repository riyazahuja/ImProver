#!/bin/bash

#SBATCH --job-name=all_iter0
#SBATCH --output=logs/3.19/all_iter0.out
#SBATCH --error=logs/3.19/all_iter0.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --mem=100G


source $HOME/miniconda3/bin/activate env


export MODEL_PATH=/data/user_data/riyaza/saved_models/ImProver_BASE_iter0/checkpoint-210
export MODEL_NAME=Qwen-7B-ALL


export ANNOTATION=true
export CONTEXT=true
export RAG=7
export DATASET_PATH=$HOME/eval_improver/improver/scripts/data/test/test_set_no_inst.json
export PORT=8003
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




# lake exe improver --best_of_n 10 --proofAsSorry false --model Qwen-7B-ALL --json_path improver_outputs_new/MIL/Qwen-7B-ALL_iter0/MIL_C04_Sets_and_Functions_solutions_Solutions_S01_Sets.json --endpoint http://0.0.0.0:8003/v1/chat/completions --annotation true --context true --rag 7 --example_file prompt_examples/Qwen-7B-ALL.txt MIL.C04_Sets_and_Functions.solutions.Solutions_S01_Sets

python3 scripts/eval_improver_async.py $BEST_OF_N $MODEL_NAME $PORT $DATASET_PATH $ANNOTATION $CONTEXT $RAG "base_iter0"



