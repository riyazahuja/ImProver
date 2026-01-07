#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/final/dependency/baselines.out
#SBATCH --error=logs/final/dependency/baselines.err
#SBATCH --cpus-per-task=128
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=1-00:00:00
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

export AZURE_OPENAI_API_KEY="7Ct6awSMrJg65ywG8SJEVeiqlootUzaITAjbrLiWHoK96wsSvQsDJQQJ99BIACHYHv6XJ3w3AAAAACOGzuBg"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"



cd /home/riyaza/eval_improver/improver
lake build eval_improver
sleep 5




echo "Running gpt-5-chat evaluation (ON DEP)"
export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5-chat/chat/completions?api-version=2025-01-01-preview"

./improver run pipeline --run_id gpt_5_chat_dep_neuro_test --informal --annotation --context 5 --examples 4   --azure true --server_concurrency 1600 --server_rate_limit 8192 --metric dependency --prompt_id final_test     --split test --model gpt-5-chat     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml



# ====== LENGTH

# gpt-5-mini

echo "Running gpt-5-mini evaluation"
export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview"
./improver run pipeline --run_id gpt_5_mini_length_neuro_test --informal --annotation --context 5  --examples 4   --azure true --server_concurrency 2048 --server_rate_limit 8192 --metric length --prompt_id final_test     --split test --model gpt-5-mini     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml


# Deepseek
echo "Running DeepSeek-R1-0528 evaluation"
export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

./improver run pipeline --run_id DS_length_neuro_test --informal --annotation --context 5 --examples 4   --azure true  --server_concurrency 2048 --server_rate_limit 8192 --metric length --prompt_id final_test     --split test --model DeepSeek-R1-0528     --num_blocks 16   --max_tokens 2048  --config experiments/final/test_eval.yaml


# gpt-5-nano
echo "Running gpt-5-nano evaluation"
export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5-nano/chat/completions?api-version=2025-01-01-preview"

./improver run pipeline --run_id gpt_5_nano_length_neuro_test  --informal --annotation --context 5 --examples 4   --azure true --server_concurrency 768 --server_rate_limit 4096 --metric length --prompt_id final_test     --split test --model gpt-5-nano     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml


# gpt-oss-120b
echo "Running gpt-oss-120b evaluation"
export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

./improver run pipeline --run_id gpt_oss_120b_length_neuro_test --informal --annotation --context 5 --examples 4   --azure true --server_concurrency 2048 --server_rate_limit 8192 --metric length --prompt_id final_test     --split test --model gpt-oss-120b     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml


# gpt-5-chat

echo "Running gpt-5-chat evaluation"
export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5-chat/chat/completions?api-version=2025-01-01-preview"

./improver run pipeline --run_id gpt_5_chat_length_neuro_test --informal --annotation --context 5 --examples 4   --azure true --server_concurrency 1600 --server_rate_limit 8192 --metric length --prompt_id final_test     --split test --model gpt-5-chat     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml








# # o1
# echo "Running o1 evaluation"
# export AZURE_OPENAI_ENDPOINT="https://riyaz-mfrbnakc-eastus2.cognitiveservices.azure.com/openai/deployments/o1/chat/completions?api-version=2025-01-01-preview"

# ./improver run pipeline --run_id o1_dep_test  --examples 4   --azure true --server_concurrency 2048 --server_rate_limit 8192 --metric dependency --prompt_id final_test     --split test --model o1     --num_blocks 16   --max_tokens 8192  --config experiments/final/test_eval.yaml


# # first get wSFT data

# ./improver run training_data --run_id base_length_train --tau 0.8 --output_path /home/riyaza/eval_improver/improver/experiments/final/length/data/wSFT_length_iter_1.jsonl     --type weighted_sft --epsilon 0.1 --variance_threshold 0.8 --filter_threshold 0.8     --replay_buffer_split 0.2 --replay_type replace --prev_run_id base_length_train

    
# # # convert wSFT data

# python experiments/final/preprocess_weights.py     /home/riyaza/eval_improver/improver/experiments/final/length/data/wSFT_length_iter_1.jsonl /home/riyaza/eval_improver/improver/experiments/final/length/data/wSFT_length_iter_1

# # # train wSFT model

# accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/length/configs/wSFT_length_iter_1.yaml

# # # merge wSFT LoRA with base to get final wSFT model

# python experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --adapter /data/user_data/riyaza/saved_models/wSFT_length_iter_1_lora     --output /data/user_data/riyaza/saved_models/wSFT_length_iter_1

# # eval wSFT model on test set
# ./improver run pipeline --run_id wSFT_length_iter_1_test     --annotation --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/wSFT_length_iter_1     --num_blocks 64     --config experiments/final/test_eval.yaml


# # get IRPO data


# ./improver run training_data --run_id base_length_train --output_path /home/riyaza/eval_improver/improver/experiments/final/length/data/IRPO_length_iter_1.jsonl     --type dpo --num_invalid 4 --max_champions 4 --filter_threshold 0.8  --min_gap 0  --replay_buffer_split 0.2 --replay_type replace

    
# # # train IRPO model

# accelerate launch -m  axolotl.cli.train /home/riyaza/eval_improver/improver/experiments/final/length/configs/IRPO_length_iter_1.yaml


# # # eval IRPO model on test set

# ./improver run pipeline --run_id IRPO_length_iter_1_test     --annotation  --informal --examples 4     --metric length --prompt_id final_test     --split test --model /data/user_data/riyaza/saved_models/IRPO_length_iter_1     --num_blocks 64     --config experiments/final/test_eval.yaml

