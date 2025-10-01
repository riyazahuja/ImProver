#!/bin/bash
#SBATCH --job-name=length_iter_1
#SBATCH --output=logs/final/length/length_iter_1.out
#SBATCH --error=logs/final/length/length_iter_1.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --mem=250G
#SBATCH --exclude=babel-11-25

source $HOME/miniconda3/bin/activate env
export HF_HOME="/data/user_data/$USER/HF"
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
mkdir -p /data/user_data/$USER/ray_tmp
export RAY_TMPDIR=/data/user_data/$USER/ray_tmp

export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576


cd /home/riyaza/eval_improver/improver
lake build eval_improver
sleep 5


# eval prev iter model on train set

# ./improver run eval --run_id base_length_train --config experiments/final/test_eval.yaml
# ./improver run analysis --run_id base_length_train

# # first get wSFT data

# ./improver run training_data --run_id base_length_train --tau 0.5 --output_path /home/riyaza/eval_improver/improver/experiments/final/length/data/wSFT_length_iter_1.jsonl     --type weighted_sft --epsilon 0.1 --variance_threshold 0.8 --filter_threshold 1.1     

    
# # convert wSFT data

# python experiments/final/preprocess_weights.py     /home/riyaza/eval_improver/improver/experiments/final/length/data/wSFT_length_iter_1.jsonl /home/riyaza/eval_improver/improver/experiments/final/length/data/wSFT_length_iter_1

# # train wSFT model

# accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/length/configs/wSFT_length_iter_1.yaml

# # merge wSFT LoRA with base to get final wSFT model

# python experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --adapter /data/user_data/riyaza/saved_models/wSFT_length_iter_1_lora     --output /data/user_data/riyaza/saved_models/wSFT_length_iter_1

# # eval wSFT model on test set

# ./improver run pipeline --run_id wSFT_length_iter_1_test_new_params     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/wSFT_length_iter_1     --num_blocks 64     --config experiments/final/test_eval.yaml

# # get IRPO data



./improver run training_data --run_id base_length_train --output_path /home/riyaza/eval_improver/improver/experiments/final/length/data/IRPO_length_iter_1.jsonl     --type dpo --num_invalid 2 --max_champions 4 --filter_threshold 0.6      

    
# train IRPO model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/length/configs/IRPO_length_iter_1.yaml


# eval IRPO model on test set

./improver run pipeline --run_id IRPO_length_iter_1_test_new_params_irpo4 --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1_rerun2     --num_blocks 64     --config experiments/final/test_eval.yaml

# ./improver run pipeline --run_id IRPO_length_iter_1_test_new_params_irpo_no_ns     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1_rerun2     --num_blocks 64     --config experiments/final/test_eval.yaml

# ./improver run pipeline --run_id IRPO_length_iter_1_test_new_params_ep2     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1_rerun2/checkpoint-322     --num_blocks 64     --config experiments/final/test_eval.yaml

# ./improver run pipeline --run_id IRPO_length_iter_1_test_new_params_ep3     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1_rerun2/checkpoint-484     --num_blocks 64     --config experiments/final/test_eval.yaml