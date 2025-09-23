#!/bin/bash
#SBATCH --job-name=dependency_iter_5
#SBATCH --output=logs/final/dependency/dependency_iter_5.out
#SBATCH --error=logs/final/dependency/dependency_iter_5.err
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

./improver run pipeline --run_id IRPO_dependency_iter_4_train     --annotation --context 10  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_train     --split train --model /data/user_data/riyaza/saved_models/IRPO_dependency_iter_4     --num_blocks 512     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml


# first get wSFT data

./improver run training_data --run_id IRPO_dependency_iter_4_train --tau 0.5 --output_path /home/riyaza/eval_improver/improver/experiments/final/dependency/data/wSFT_dependency_iter_5.jsonl     --type weighted_sft --epsilon 0.1 --variance_threshold 0.8 --filter_threshold 1.1     --replay_buffer_split 0.4 --replay_type replace --prev_run_id base_dependency_train,IRPO_dependency_iter_1_train,IRPO_dependency_iter_2_train,IRPO_dependency_iter_3_train

    
# convert wSFT data

python /home/$USER/eval_improver/improver/experiments/final/preprocess_weights.py     /home/riyaza/eval_improver/improver/experiments/final/dependency/data/wSFT_dependency_iter_5.jsonl /home/riyaza/eval_improver/improver/experiments/final/dependency/data/wSFT_dependency_iter_5

# train wSFT model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/dependency/configs/wSFT_dependency_iter_5.yaml

# merge wSFT LoRA with base to get final wSFT model

python /home/$USER/eval_improver/improver/experiments/final/merge.py     --ref /data/user_data/riyaza/saved_models/IRPO_dependency_iter_4     --adapter /data/user_data/riyaza/saved_models/wSFT_dependency_iter_5_lora     --output /data/user_data/riyaza/saved_models/wSFT_dependency_iter_5

# eval wSFT model on test set

./improver run pipeline --run_id wSFT_dependency_iter_5_test     --annotation --context 10  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_test     --split test --model /data/user_data/riyaza/saved_models/wSFT_dependency_iter_5     --num_blocks 64     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml

# get IRPO data



./improver run training_data --run_id IRPO_dependency_iter_4_train --output_path /home/riyaza/eval_improver/improver/experiments/final/dependency/data/IRPO_dependency_iter_5.jsonl     --type dpo --num_invalid -1 --max_champions -1 --filter_threshold 1.1     --replay_buffer_split 0.4 --replay_type replace --prev_run_id base_dependency_train,IRPO_dependency_iter_1_train,IRPO_dependency_iter_2_train,IRPO_dependency_iter_3_train

    
# train IRPO model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/dependency/configs/IRPO_dependency_iter_5.yaml


# eval IRPO model on test set

./improver run pipeline --run_id IRPO_dependency_iter_5_test     --annotation --context 10  --informal --examples 4     --metric dependency --prompt_id /home/$USER/eval_improver/improver/prompts/final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_dependency_iter_5     --num_blocks 64     --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml

