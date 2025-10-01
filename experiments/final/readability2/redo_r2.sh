#!/bin/bash
#SBATCH --job-name=readability2_redo_consensus
#SBATCH --output=logs/final/readability2/readability2_redo_consensus.out
#SBATCH --error=logs/final/readability2/readability2_redo_consensus.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --mem=250G
#SBATCH --gres=gpu:L40S:1

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


cd /home/trowney/ImProver
lake build eval_improver
sleep 5


for run_id in {"base_readability2_train","wSFT_readability2_iter_1_test","IRPO_readability2_iter_1_test"}
do
    echo "starting LLM metric for $run_id"
    ./improver run llm_metric --run_id $run_id  --num_blocks 512 --config experiments/final/test_eval.yaml
    ./improver run analysis --run_id $run_id
done



# echo "starting wSFT"
# # first get wSFT data

# ./improver run training_data --run_id base_readability2_train --tau 0.5 --output_path /home/trowney/ImProver/experiments/final/readability2/data/wSFT_readability2_iter_1.jsonl     --type weighted_sft --epsilon 0.1 --variance_threshold 0.8 --filter_threshold 1.1     

    
# # convert wSFT data

# python experiments/final/preprocess_weights.py     /home/trowney/ImProver/experiments/final/readability2/data/wSFT_readability2_iter_1.jsonl /home/trowney/ImProver/experiments/final/readability2/data/wSFT_readability2_iter_1

# # train wSFT model

# accelerate launch -m  axolotl.cli.train /home/trowney/ImProver/experiments/final/readability2/configs/wSFT_readability2_iter_1.yaml

# # merge wSFT LoRA with base to get final wSFT model

# python experiments/final/merge.py     --ref deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --adapter /data/user_data/trowney/saved_models/wSFT_readability2_iter_1_lora     --output /data/user_data/trowney/saved_models/wSFT_readability2_iter_1

# # eval wSFT model on test set

# ./improver run pipeline --run_id wSFT_readability2_iter_1_test     --annotation --context 10  --informal --examples 4     --metric readability2 --prompt_id final_test     --split test --model /data/user_data/trowney/saved_models/wSFT_readability2_iter_1     --num_blocks 64     --config experiments/final/test_eval.yaml

# get IRPO data



# ./improver run training_data --run_id base_readability2_train --output_path /home/trowney/ImProver/experiments/final/readability2/data/IRPO_readability2_iter_1.jsonl     --type dpo --num_invalid 8 --max_champions 1 --filter_threshold 1.1     

    
# # train IRPO model

# accelerate launch -m  axolotl.cli.train /home/trowney/ImProver/experiments/final/readability2/configs/IRPO_readability2_iter_1.yaml


# # eval IRPO model on test set

# ./improver run pipeline --run_id IRPO_readability2_iter_1_test     --annotation --context 10  --informal --examples 4     --metric readability2 --prompt_id final_test     --split test --model /data/user_data/trowney/saved_models/IRPO_readability2_iter_1     --num_blocks 64     --config experiments/final/test_eval.yaml

