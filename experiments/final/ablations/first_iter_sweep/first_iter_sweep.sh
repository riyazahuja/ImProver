#!/bin/bash
#SBATCH --job-name=length_iter_1
#SBATCH --output=logs/final/length/length_iter_1.out
#SBATCH --error=logs/final/length/length_iter_1.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=250G

source $HOME/miniconda3/bin/activate env
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export RAY_TMPDIR=/home/riyaz/ray_tmp

export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576


cd /home/riyaz/ImProver
lake build eval_improver
sleep 5



# ./improver run pipeline --run_id IRPO_length_iter_1_train     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_train     --split train --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 64     --config experiments/final/test_eval.yaml


# ============ DATA ===========

#SFT 

./improver run training_data --run_id base_length_train --output_path /home/riyaz/ImProver/experiments/final/length/data/wSFT_length_iter_1.jsonl     --type sft

#wSFT

./improver run training_data --run_id base_length_train --output_path /home/riyaz/ImProver/experiments/final/length/data/wSFT_length_iter_1.jsonl     --type weighted_sft 



#DPO

./improver run training_data --run_id base_length_train --output_path /home/riyaz/ImProver/experiments/final/length/data/IRPO_length_iter_1.jsonl     --type dpo --num_invalid 2 --max_champions 4 --filter_threshold 0.8  --min_gap 2  --replay_buffer_split 0.2 --replay_type replace


#IRPO

./improver run training_data --run_id base_length_train --output_path /home/riyaz/ImProver/experiments/final/length/data/IRPO_length_iter_1.jsonl     --type dpo --num_invalid 2 --max_champions 4 --filter_threshold 0.8  --min_gap 2  --replay_buffer_split 0.2 --replay_type replace

# ============ Basic ===========




#each on best of 8: epsilon = 0.1, tau = 0.8

# for each, sweep across LR = [1e-6,5e-5,1e-5,2e-5] and effective BS = [1,4,8] (alpha = 0 and beta =0.05) and hardness = [0.75,1.0]






# =========== Additional DPO/IRPO ===========


# for best, sweep across IRPO/DPO beta = [0.02,0.05,0.1] and for IRPO alpha = [0.2, 0.5, 1.0]. Pairs given by (W,L)=(2,1).

# for best, sweep across W,L = [1,4,8]. for best, test min_gap = [0,1,2]


# =========== Combined ===========


# Take best (w)SFT and (IR/D)PO combination and run back to back, sweeping variance threshold = [0.5,0.8,1.0]





















# eval prev iter model on train set

# ./improver run pipeline --run_id IRPO_length_iter_1_test     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 16     --config experiments/final/test_eval.yaml
# ./improver run pipeline --run_id IRPO_length_iter_1_train     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_train     --split train --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 64     --config experiments/final/test_eval.yaml


# first get wSFT data

# ./improver run training_data --run_id base_length_train --tau 0.5 --output_path /home/riyaz/ImProver/experiments/final/length/data/wSFT_length_iter_1.jsonl     --type weighted_sft --epsilon 0.1 --variance_threshold 0.8 --filter_threshold 0.8     --replay_buffer_split 0.4 --replay_type replace --prev_run_id base_length_train

    
# # convert wSFT data

# python experiments/final/preprocess_weights.py     /home/riyaz/ImProver/experiments/final/length/data/wSFT_length_iter_1.jsonl /home/riyaz/ImProver/experiments/final/length/data/wSFT_length_iter_1

# # train wSFT model

# accelerate launch -m  axolotl.cli.train /home/riyaz/ImProver/experiments/final/length/configs/wSFT_length_iter_1.yaml

# # merge wSFT LoRA with base to get final wSFT model

# python experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --adapter /home/riyaz/saved_models/wSFT_length_iter_1_lora     --output /home/riyaz/saved_models/wSFT_length_iter_1

# eval wSFT model on test set

# python experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --adapter /home/riyaz/saved_models/wSFT_length_iter_1_lora/checkpoint-9     --output /home/riyaz/saved_models/wSFT_length_iter_1_ep1

# python experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --adapter /home/riyaz/saved_models/wSFT_length_iter_1_lora/checkpoint-18     --output /home/riyaz/saved_models/wSFT_length_iter_1_ep2

# ./improver run pipeline --run_id wSFT_length_iter_1_test_ep1     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /home/riyaz/saved_models/wSFT_length_iter_1_ep1     --num_blocks 16     --config experiments/final/test_eval.yaml



# ./improver run pipeline --run_id wSFT_length_iter_1_test_ep2    --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /home/riyaz/saved_models/wSFT_length_iter_1_ep2     --num_blocks 16     --config experiments/final/test_eval.yaml


# get IRPO data


./improver run training_data --run_id base_length_train --output_path /home/riyaz/ImProver/experiments/final/length/data/IRPO_length_iter_1.jsonl     --type dpo --num_invalid 2 --max_champions 4 --filter_threshold 0.8  --min_gap 2  --replay_buffer_split 0.2 --replay_type replace

    
# # train IRPO model

accelerate launch -m  axolotl.cli.train /home/riyaz/ImProver/experiments/final/length/configs/IRPO_length_iter_1.yaml


# # eval IRPO model on test set

./improver run pipeline --run_id IRPO_length_iter_1_test     --annotation --context 10  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /home/riyaz/saved_models/IRPO_length_iter_1     --num_blocks 16     --config experiments/final/test_eval.yaml

