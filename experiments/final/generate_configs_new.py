import os


USER = os.getenv("USER", "riyaza")

improver_base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
models_path = f"/data/user_data/{USER}/saved_models"
deepspeed_path = os.path.join(improver_base_path, "deepspeed_configs")
conda_dir = "$HOME/miniconda3"
conda_env = "env"


def make_irpo_base_config(base_config_path: str, data_dir: str, base_model: str):
    os.makedirs(os.path.dirname(base_config_path), exist_ok=True)
    dataset_path = os.path.join(data_dir, "IRPO_train.jsonl")

    yaml_config = f"""
group_by_length: true

base_model: {base_model}
ref_model: {base_model}

trust_remote_code: true

rl: dpo
rl_beta: 0.07
rpo_alpha: 1.0


sequence_len: 8192
flash_attn_2: true
gradient_checkpointing: true

micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 1
learning_rate: 5e-6
optimizer: adamw_torch
lr_scheduler: cosine
warmup_steps: 5
weight_decay: 0.0

max_grad_norm: 1.0


deepspeed: {os.path.join(deepspeed_path,'zero3_bf16_cpuoffload_all_custom.json')}
bf16: true

datasets:
  - path: {dataset_path}
    type: chatml.prompt_pairs

val_set_size: 0.05
output_dir: {os.path.join(models_path, 'IRPO_sweep')}

wandb_project: "IRPO_sweep"
logging_steps: 10
evals_per_epoch: 1
save_strategy: "no"
"""
    with open(base_config_path, "w") as f:
        f.write(yaml_config)


def make_irpo_sweep_configs(sweeps_dir: str):
    os.makedirs(sweeps_dir, exist_ok=True)

    # Alpha/Beta sweep (only for first iteration, on default data)
    alpha_beta_sweep = """
parameters:
  alpha:
    values: [0.2, 0.5, 1.0]
  beta:
    values: [0.02, 0.05, 0.1]
"""
    with open(os.path.join(sweeps_dir, "IRPO_alpha_beta.yaml"), "w") as f:
        f.write(alpha_beta_sweep)

    # W/L sweep (only for first iteration, uses copied results for later metrics)
    wl_sweep = """
parameters:
  w:
    values: [1, 1, 4, 4, 2, 4, 2]
  l:
    values: [1, 4, 1, 4, 4, 2, 2]
"""
    with open(os.path.join(sweeps_dir, "IRPO_wl.yaml"), "w") as f:
        f.write(wl_sweep)

    # Learning rate sweep (each iteration)
    lr_sweep = """
parameters:
  learning_rate:
    values: [1e-6, 2e-6, 5e-6, 1e-5]
"""
    with open(os.path.join(sweeps_dir, "IRPO_lr.yaml"), "w") as f:
        f.write(lr_sweep)


def make_slurm(metric: str, output_dir: str):
    job_name = f"{metric}_iter_sweep"
    slurm_path = os.path.join(output_dir, f"{job_name}.sh")
    configs_dir = os.path.join(output_dir, "configs")
    sweeps_dir = os.path.join(output_dir, "sweeps")
    data_dir = os.path.join(output_dir, "data")

    base_config = os.path.join(configs_dir, "IRPO_base.yaml")

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/final/{metric}/{job_name}.out
#SBATCH --error=logs/final/{metric}/{job_name}.err
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --mem=250G
#SBATCH --exclude=babel-15-36,babel-1-23

set -e

source {os.path.join(conda_dir, "bin", "activate")} {conda_env}
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

cd {improver_base_path}
lake build eval_improver
sleep 5

# ---------------------------------------------------------------------
# 1) Generate IRPO training data with sweeps over hardness, gap, replay
#    (see ImProver/basic/training_data.py for CLI args)
# ---------------------------------------------------------------------

./improver run training_data \
    --run_id base_{metric}_train \
    --output_path {os.path.join(data_dir, 'IRPO_train.jsonl')} \
    --type dpo \
    --num_invalid -1 \
    --max_champions -1 \
    --filter_threshold 1.1


# ---------------------------------------------------------------------
# 2) Axolotl sweeps
# ---------------------------------------------------------------------

# (a) Alpha/Beta sweep (first iteration / default data)
axolotl train {base_config} --sweep {os.path.join(sweeps_dir, 'IRPO_alpha_beta.yaml')}

# (b) W/L sweep (first iteration; W/L combos)
axolotl train {base_config} --sweep {os.path.join(sweeps_dir, 'IRPO_wl.yaml')}

# (c) LR sweep (each iteration)
axolotl train {base_config} --sweep {os.path.join(sweeps_dir, 'IRPO_lr.yaml')}

"""
    with open(slurm_path, "w") as f:
        f.write(script)


params = ["length"]
base_path = os.path.join(improver_base_path, "experiments", "final")


for metric in params:
    experiments_path = os.path.join(base_path, metric)
    configs_dir = os.path.join(experiments_path, "configs")
    sweeps_dir = os.path.join(experiments_path, "sweeps")
    data_dir = os.path.join(experiments_path, "data")

    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(sweeps_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 20)
    print(f"Preparing sweeps for metric: {metric}")
    print("-" * 20)

    irpo_base_config = os.path.join(configs_dir, "IRPO_base.yaml")
    make_irpo_base_config(
        irpo_base_config, data_dir, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    make_irpo_sweep_configs(sweeps_dir)
    make_slurm(metric, experiments_path)
