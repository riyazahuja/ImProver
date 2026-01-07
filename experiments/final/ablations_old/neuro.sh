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




# ./improver run pipeline --run_id ablation_base_dep --examples 4     --metric dependency --prompt_id final_test     --split test --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 16     --config experiments/final/test_eval.yaml


# ./improver run pipeline --run_id ablation_base_dep_cos  --annotation  --examples 4     --metric dependency --prompt_id final_test     --split test --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 16     --config experiments/final/test_eval.yaml


./improver run pipeline --run_id ablation_base_dep_cos_ctx --annotation --context 10 --examples 4     --metric dependency --prompt_id final_test     --split test --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 16     --config experiments/final/test_eval.yaml



./improver run pipeline --run_id ablation_base_dep_cos_ctx_inf  --annotation --context 10 --informal --examples 4     --metric dependency --prompt_id final_test     --split test --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --num_blocks 16     --config experiments/final/test_eval.yaml

