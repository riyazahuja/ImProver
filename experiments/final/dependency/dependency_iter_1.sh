#!/bin/bash
#SBATCH --job-name=dependency_iter_1
#SBATCH --output=logs/final/dependency/dependency_iter_1.out
#SBATCH --error=logs/final/dependency/dependency_iter_1.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=150G
#SBATCH --exclude=babel-15-36,babel-1-23

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/$USER/HF"
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
mkdir -p /data/user_data/$USER/ray_tmp
export RAY_TMPDIR=/data/user_data/$USER/ray_tmp


cd /home/$USER/eval_improver/improver
lake build eval_improver
sleep 5


# eval prev iter model on train set

./improver run eval --run_id base_dependency_train     --annotation --context 10  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_train     --split train --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 512     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml
./improver run analysis --run_id base_dependency_train     --annotation --context 10  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_train     --split train --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 512     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml



# first get wSFT data

./improver run training_data --run_id base_dependency_train --tau 0.5 --output_path /home/riyaza/eval_improver/improver/experiments/final/dependency/data/wSFT_dependency_iter_1.json     --type weighted_sft --epsilon 0.1 --variance_threshold 0.8 --filter_threshold 1.1     

    
# convert wSFT data

python /home/$USER/eval_improver/improver/experiments/final/preprocess_weights.py     /home/riyaza/eval_improver/improver/experiments/final/dependency/data/wSFT_dependency_iter_1.jsonl /home/riyaza/eval_improver/improver/experiments/final/dependency/data/wSFT_dependency_iter_1

# train wSFT model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/dependency/configs/wSFT_dependency_iter_1.yaml

# merge wSFT LoRA with base to get final wSFT model

python /home/$USER/eval_improver/improver/experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --adapter /data/user_data/riyaza/saved_models/wSFT_dependency_iter_1_lora     --output /data/user_data/riyaza/saved_models/wSFT_dependency_iter_1

# eval wSFT model on test set

./improver run pipeline --run_id wSFT_dependency_iter_1_test     --annotation --context 10  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_test     --split test --model /data/user_data/riyaza/saved_models/wSFT_dependency_iter_1     --num_blocks 64     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml

# get IRPO data



./improver run training_data --run_id base_dependency_train --output_path /home/riyaza/eval_improver/improver/experiments/final/dependency/data/IRPO_dependency_iter_1     --type dpo --num_invalid -1 --max_champions -1 --filter_threshold 1.1     

    
# train IRPO model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/dependency/configs/IRPO_dependency_iter_1.yaml


# eval IRPO model on test set

./improver run pipeline --run_id IRPO_dependency_iter_1_test     --annotation --context 10  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_test     --split test --model /data/user_data/USER/saved_models/IRPO_dependency_iter_1     --num_blocks 64     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml

