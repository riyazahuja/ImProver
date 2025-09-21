import subprocess
import os

from preprocess_weights import convert
USER=os.getenv('USER','riyaza')

def make_wSFT_config(name, config_path, data_path, base_model):
    output_path = os.path.join(config_path, f"{name}.yaml")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_model_path = f"/data/user_data/{USER}/saved_models/{name}_lora"
    dataset_path = os.path.join(data_path, name)

    yaml_config = f"""
base_model: {base_model}
trust_remote_code: true

load_in_4bit: true
bnb4_quant_type: nf4
bnb4_compute_dtype: bf16


adapter: lora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]

merge_lora: true

sequence_len: 32768
flash_attn_2: true
gradient_checkpointing: true

micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 3
learning_rate: 2e-5
optimizer: adamw_torch
lr_scheduler: cosine
warmup_steps: 5
weight_decay: 0.0

max_grad_norm: 1.0


deepspeed: /home/USER/deepspeed_configs/zero3_bf16_cpuoffload_params_16b_fls.json
bf16: true

datasets:
  - path: {dataset_path}
    type: arrow

val_set_size: 0.05
output_dir: {output_model_path}

# Optional trackers
wandb_project: "{name}"
logging_steps: 2
evals_per_epoch: 2
save_strategy: "no"

plugins:
  - "axolotl.integrations.weightedSFT.WeightedSFTPlugin"
normalize_batch_weights: false
remove_unused_columns: false
"""
    with open(output_path, "w") as f:
        f.write(yaml_config)
    return output_path, output_model_path


def make_IRPO_config(name, config_path, data_path, wSFT_model):
    output_path = os.path.join(config_path, f"{name}.yaml")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_model_path = f"/data/user_data/USER/saved_models/{name}"
    dataset_path = os.path.join(data_path, name)

    yaml_config = f"""


base_model: {wSFT_model}
ref_model: {wSFT_model}

trust_remote_code: true

rl: dpo
rl_beta: 0.07
rpo_alpha: 1.0


sequence_len: 32768
flash_attn_2: true
gradient_checkpointing: true

micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 5e-6
optimizer: adamw_torch
lr_scheduler: cosine
warmup_steps: 5
weight_decay: 0.0

max_grad_norm: 1.0


deepspeed: /home/{USER}/deepspeed_configs/zero3_bf16_cpuoffload_params.json
bf16: true

datasets:
  - path: {dataset_path}
    type: arrow

val_set_size: 0.05
output_dir: {output_model_path}

# Optional trackers
wandb_project: "{name}"
logging_steps: 2
evals_per_epoch: 2
save_strategy: "no"
"""
    with open(output_path, "w") as f:
        f.write(yaml_config)
    return output_path, output_model_path


def make_slurm(
    wSFT_name,
    IRPO_name,
    output_path,
    i,
    metric,
    curr_run_id,
    prev_run_ids,
    wSFT_config,
    wSFT_model,
    IRPO_config,
    IRPO_model,
    prev_iteration_model,
):
    run_name = f"{metric}_iter_{i}"
    output_path = os.path.join(output_path, f"{run_name}.sh")
    IRPO_dataset_path = os.path.join(data_path, f"{IRPO_name}")
    wSFT_dataset_path = os.path.join(data_path, f"{wSFT_name}")
    wSFT_model_final = wSFT_model.replace("_lora", "")

    script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --output=logs/final/{metric}/{run_name}.out
#SBATCH --error=logs/final/{metric}/{run_name}.err
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

./improver run pipeline --run_id {curr_run_id}_train \
    --annotation --context 10  --informal --examples 4 \
    --metric {metric} --prompt_id /home/$USER/eval_improver/improver/prompts/final_train \
    --split train --model {prev_iteration_model} \
    --num_blocks 2048 \
    --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml


# first get wSFT data

./improver run training_data --run_id {curr_run_id}_train --tau 0.5 --output_path {wSFT_dataset_path}.json \
    --type weighted_sft --epsilon 0.1 --variance_threshold 0.8 --filter_threshold 1.1 \
    {f'--replay_buffer_split 0.4 --replay_type replace --prev_run_id {",".join(prev_run_ids)}' if len(prev_run_ids) > 0 else ""}

    
# convert wSFT data

python /home/$USER/eval_improver/improver/experiments/final/preprocess_weights.py \
    {wSFT_dataset_path}.jsonl {wSFT_dataset_path}

# train wSFT model

accelerate launch -m  axolotl.cli.train {wSFT_config}

# merge wSFT LoRA with base to get final wSFT model

python /home/$USER/eval_improver/improver/experiments/final/merge.py \
    --ref {base_model} \
    --adapter {wSFT_model} \
    --output {wSFT_model_final}

# eval wSFT model on test set

./improver run pipeline --run_id {wSFT_name}_test \
    --annotation --context 10  --informal --examples 4 \
    --metric {metric} --prompt_id /home/$USER/eval_improver/improver/prompts/final_test \
    --split test --model {wSFT_model_final} \
    --num_blocks 32 \
    --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml

# get IRPO data



./improver run training_data --run_id {curr_run_id}_train --output_path {IRPO_dataset_path} \
    --type dpo --num_invalid -1 --max_champions -1 --filter_threshold 1.1 \
    {f'--replay_buffer_split 0.4 --replay_type replace --prev_run_id {",".join(prev_run_ids)}' if len(prev_run_ids) > 0 else ""}

    
# train IRPO model

accelerate launch -m  axolotl.cli.train {IRPO_config}


# eval IRPO model on test set

./improver run pipeline --run_id {IRPO_name}_test \
    --annotation --context 10  --informal --examples 4 \
    --metric {metric} --prompt_id /home/$USER/eval_improver/improver/prompts/final_test \
    --split test --model {IRPO_model} \
    --num_blocks 32 \
    --config /home/$USER/eval_improver/improver/experiments/final/test_eval.yaml

"""
    with open(output_path, "w") as f:
        f.write(script)
    return output_path


params = ["length", "declarativity", "dependency"]
max_iterations = 5
base_path = f"/home/{USER}/eval_improver/improver/experiments/final"


for metric in params:

    experiments_path = os.path.join(base_path, metric)

    print("=" * 20)
    print(f"Processing parameter set: {metric}")
    print("-" * 20)

    base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    past_run_ids = []
    curr_run_id = f"base_{metric}"

    for i in range(1, max_iterations + 1):
        print(f"Starting iteration {i}")
        output_path = experiments_path
        config_path = os.path.join(output_path, "configs")
        data_path = os.path.join(output_path, "data")

        os.makedirs(config_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        wSFT_name = f"wSFT_{metric}_iter_{i}"
        IRPO_name = f"IRPO_{metric}_iter_{i}"

        wSFT_config, wSFT_model = make_wSFT_config(
            wSFT_name, config_path, data_path, base_model
        )

        IRPO_config, IRPO_model = make_IRPO_config(
            IRPO_name, config_path, data_path, wSFT_model
        )

        slurm_script_path = make_slurm(
            wSFT_name,
            IRPO_name,
            output_path,
            i,
            metric,
            curr_run_id,
            past_run_ids,
            wSFT_config,
            wSFT_model,
            IRPO_config,
            IRPO_model,
            base_model,
        )

        base_model = IRPO_model
        past_run_ids.append(curr_run_id)
        curr_run_id = f"{IRPO_name}_train"
