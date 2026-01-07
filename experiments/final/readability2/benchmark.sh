#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/final/readability/baselines.out
#SBATCH --error=logs/final/readability/baselines.err
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=200G


source $HOME/miniconda3/bin/activate env
export NCCL_DEBUG=INFO
export NCCL_BLOCKING=1
export ACCELERATE_USE_REENTRANT_CHECKPOINT=0
export DEEPSPEED_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export RAY_TMPDIR=/data/user_data/trowney/ray_tmp
export HF_HOME="/data/user_data/trowney/HF"

export DEEPSPEED_COMM=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

export AZURE_OPENAI_API_KEY="7Ct6awSMrJg65ywG8SJEVeiqlootUzaITAjbrLiWHoK96wsSvQsDJQQJ99BIACHYHv6XJ3w3AAAAACOGzuBg"


# cd /home/trowney/ImProver
# lake build eval_improver
# sleep 5



# ====== Readability Baselines ======



# # gpt-5-nano
# echo "Running gpt-5-nano evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5-nano/chat/completions?api-version=2025-01-01-preview"

# ./improver run pipeline --run_id gpt_5_nano_read_test --examples 4   --azure true --server_concurrency 768 --server_rate_limit 4096 --metric readability2 --prompt_id final_test     --split test --judge_model gpt-5-nano  --judge_n 3   --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml



# gpt-5-mini

# echo "Running gpt-5-mini evaluation"
# # export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview"

# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview" 

# ./improver run pipeline --run_id gpt_5_mini_read_test --examples 4   --azure true --server_concurrency 1600 --server_rate_limit 8192 --metric readability2 --prompt_id final_test     --split test --judge_model gpt-5-nano  --judge_n 3  --num_blocks 16   --max_tokens 2048  --config experiments/final/test_eval.yaml 




# # gpt-oss-120b
echo "Running gpt-oss-120b evaluation"
export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

./improver run pipeline --run_id gpt_oss_120b_read_test --examples 4   --azure true --server_concurrency 2048 --server_rate_limit 8192 --metric readability2 --prompt_id final_test     --split test --judge_model gpt-5-nano  --judge_n 3     --num_blocks 16   --max_tokens 2048  --config experiments/final/test_eval.yaml &


# gpt-5-chat

echo "Running gpt-5-chat evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5-chat/chat/completions?api-version=2025-01-01-preview"

AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5-chat/chat/completions?api-version=2025-01-01-preview" ./improver run pipeline --run_id gpt_5_chat_read_test --examples 4   --azure true --server_concurrency 1600 --server_rate_limit 8192 --metric readability2 --prompt_id final_test     --split test --judge_model gpt-5-nano  --judge_n 3     --num_blocks 16   --max_tokens 2048  --config experiments/final/test_eval.yaml &


# Deepseek
echo "Running DeepSeek-R1-0528 evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview" ./improver run pipeline --run_id DS_read_test --examples 4   --azure true  --server_concurrency 1600 --server_rate_limit 8192 --metric readability2 --prompt_id final_test     --split test --judge_model gpt-5-nano  --judge_n 3     --num_blocks 16   --max_tokens 2048   --config experiments/final/test_eval.yaml &


wait


# # ====== Readability w/ neuro Baselines ======

# # gpt-5-mini

# echo "Running gpt-5-mini evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview"
# ./improver run pipeline --run_id gpt_5_mini_read_neuro_test --informal --annotation --context 5  --examples 4   --azure true --server_concurrency 2048 --server_rate_limit 8192 --metric readability2 --prompt_id final_test     --split test --model gpt-5-mini     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml




# # gpt-5-nano
# echo "Running gpt-5-nano evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5-nano/chat/completions?api-version=2025-01-01-preview"

# ./improver run pipeline --run_id gpt_5_nano_read_neuro_test  --informal --annotation --context 5 --examples 4   --azure true --server_concurrency 768 --server_rate_limit 4096 --metric readability2 --prompt_id final_test     --split test --model gpt-5-nano     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml


# # gpt-oss-120b
# echo "Running gpt-oss-120b evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

# ./improver run pipeline --run_id gpt_oss_120b_read_neuro_test --informal --annotation --context 5 --examples 4   --azure true --server_concurrency 2048 --server_rate_limit 8192 --metric readability2 --prompt_id final_test     --split test --model gpt-oss-120b     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml


# # gpt-5-chat

# echo "Running gpt-5-chat evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5-chat/chat/completions?api-version=2025-01-01-preview"

# ./improver run pipeline --run_id gpt_5_chat_read_neuro_test --informal --annotation --context 5 --examples 4   --azure true --server_concurrency 1600 --server_rate_limit 8192 --metric readability2 --prompt_id final_test     --split test --model gpt-5-chat     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml


# # Deepseek
# echo "Running DeepSeek-R1-0528 evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

# ./improver run pipeline --run_id DS_read_neuro_test --informal --annotation --context 5 --examples 4   --azure true  --server_concurrency 2048 --server_rate_limit 8192 --metric readability2 --prompt_id final_test     --split test --model DeepSeek-R1-0528     --num_blocks 16   --max_tokens 2048  --config experiments/final/test_eval.yaml