#!/bin/bash
#SBATCH --job-name=i1
#SBATCH --output=logs/final/length/i1.out
#SBATCH --error=logs/final/length/i1.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=250G

source $HOME/miniconda3/bin/activate env
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export RAY_TMPDIR=/data/user_data/riyaza/ray_tmp
export HF_HOME="/data/user_data/riyaza/HF"

export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576


cd /home/riyaza/eval_improver/improver
lake build eval_improver
sleep 5





# ./improver run training_data --run_id base_length_train --output_path /home/riyaza/eval_improver/improver/experiments/final/length/data/SFT_length_iter_1.jsonl     --type sft --num_invalid 4 --max_champions 4 --filter_threshold 0.8 --variance_threshold 0.8 --variance_threshold_type bottom


axolotl train /home/riyaza/eval_improver/improver/experiments/final/length/configs/SFT_length_iter_1.yaml


python experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    --adapter /data/user_data/riyaza/saved_models/SFT_length_iter_1_lora     --output /data/user_data/riyaza/saved_models/SFT_length_iter_1

rm -rf /data/user_data/riyaza/saved_models/IRPO_length_iter_1_lora

./improver run pipeline --run_id SFT_length_iter_1_test     --annotation  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/SFT_length_iter_1     --num_blocks 32     --config experiments/final/test_eval.yaml

./improver run training_data --run_id base_length_train --output_path /home/riyaza/eval_improver/improver/experiments/final/length/data/IRPO_length_iter_1.jsonl     --type dpo --filter_threshold 0.8  --min_gap 0 


# # train IRPO model

accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/length/configs/IRPO_length_iter_1.yaml


# # eval IRPO model on test set

./improver run pipeline --run_id IRPO_length_iter_1_test     --annotation  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1     --num_blocks 32     --config experiments/final/test_eval.yaml


# ./improver run pipeline --run_id IRPO_length_iter_1_train     --annotation  --informal --examples 4     --metric length --prompt_id final_train     --split train --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1     --num_blocks 72     --config experiments/final/test_eval.yaml
