#!/bin/bash

#SBATCH --job-name=neurosymbolic_base_eval3
#SBATCH --output=logs/final/neurosymbolic_base_eval3.out
#SBATCH --error=logs/final/neurosymbolic_base_eval3.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=150G





source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/$USER/HF"
export DEEPSPEED_LOG_LEVEL=DEBUG
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONUNBUFFERED=1
mkdir -p /data/user_data/$USER/ray_tmp
export RAY_TMPDIR=/data/user_data/$USER/ray_tmp


cd /home/$USER/eval_improver/improver
lake build eval_improver
sleep 5


export prompt_id="final_test"
export split="test"
export model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export num_blocks=48
export metric="length"


# ./improver run pipeline --run_id "neurosymbolic_ablation_none" --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml


# # ./improver run pipeline --run_id "neurosymbolic_ablation_examples" --examples 3 --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml


# ./improver run pipeline --run_id "neurosymbolic_ablation_CoS2" --annotation --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml



# ./improver run pipeline --run_id "neurosymbolic_ablation_context2" --examples 3 --annotation --context 5 --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml

# # ./improver prompts get --dataset_path "/home/riyaza/eval_improver/improver/train/data/train/final_dataset_fixed.json" --split test --prompts_id "final_test" --cpus 64 --k 10 --rag_id "rag_test_set"

# ./improver run pipeline --run_id "neurosymbolic_ablation_informal2" --annotation --context 5 --informal --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml

# ./improver run pipeline --run_id "neurosymbolic_ablation_rag2"  --annotation --context 5  --rag 5 --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml


# ./improver run pipeline --run_id "neurosymbolic_ablation_examples_again" --examples 4 --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml


# ./improver run pipeline --run_id "neurosymbolic_ablation_CoS_again" --examples 4 --annotation --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml



# ./improver run pipeline --run_id "neurosymbolic_ablation_context_again" --examples 4 --annotation --context 10 --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml


./improver run pipeline --run_id "neurosymbolic_ablation_final_informal"  --examples 4 --annotation --context 10  --informal --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml
# ./improver run pipeline --run_id "neurosymbolic_ablation_rag_again_k10"  --examples 4 --annotation --context 10  --rag 10 --metric $metric --prompt_id $prompt_id --split $split --model $model --num_blocks $num_blocks --config /home/riyaza/eval_improver/improver/experiments/final/test_eval.yaml




# ./improver prompts get --dataset_path "/home/riyaza/eval_improver/improver/train/data/train/final_dataset_fixed.json" --split train --prompts_id "final_train" --cpus 64 --k 10 --rag_id "rag_train_set"
