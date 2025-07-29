#!/usr/bin/env python3
"""
expert_iteration.py

A self-contained script to automate an expert-iteration workflow consisting of:
1. Two inference runs (train/test) using a provided core inference config template.
2. A training run using a provided core training config template.
3. Repeating the above for N iterations, chaining the newly trained model as the base
   model for the next iteration.

The script builds specialised YAML config files and SBATCH shell scripts, submits
jobs to Slurm, tracks their job-ids and enforces ordering via dependencies.

Example usage
-------------
python experiments/expert_iteration.py \
    --inf-template   configs/core/inference_template.yaml \
    --train-template configs/core/train_template.yml \
    --base-model     deepseek-ai/DeepSeek-Prover-V2-7B \
    --base-name      DS2 \
    --iterations     3 \
    --cpus           64 \
    --gres           gpu:A6000:8 \
    --mem            150G

The script must be launched from the repository root so that relative paths match
those produced elsewhere in the code-base.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import json

try:
    import yaml  # type: ignore
except ImportError as e:  # pragma: no cover
    sys.stderr.write("PyYAML is required: pip install pyyaml\n")
    raise

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def dump_yaml(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def submit_and_get_jobid(script_path: Path, dependency: str | None = None) -> str:
    """Submit the sbatch script and return the Slurm job-id as a string."""
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(script_path))
    result = subprocess.check_output(cmd, text=True).strip()
    # Expected format: "Submitted batch job 123456"
    job_id = result.split()[-1]
    print(f"[Slurm] {script_path.name} -> job {job_id}")
    return job_id


def wait_for_job(job_id: str, poll_seconds: int = 60):
    """Block until the given Slurm job finishes (i.e. disappears from squeue)."""
    while True:
        try:
            out = subprocess.check_output(["squeue", "-j", job_id, "-h"])
            # When the job is finished squeue prints nothing and exits 0.
            if not out.strip():
                break
        except subprocess.CalledProcessError:
            # Non-zero exit means job not found – treat as finished.
            break
        time.sleep(poll_seconds)
    print(f"[Slurm] job {job_id} finished")




# -----------------------------------------------------------------------------
# SBATCH script generation helpers
# -----------------------------------------------------------------------------

def sbatch_header(job_name: str, cpus: int, gres: str, mem: str, extra: str = "") -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --cpus-per-task={cpus}
#SBATCH --time=1-00:00:00
#SBATCH --gres={gres}
#SBATCH --mem={mem}
{extra}
"""


def make_inference_sbatch(inference_config: str, run_id: str, prompt_id: str, split: str, model: str, script_path: Path, stdout_path: Path, stderr_path: Path, job_name: str, cpus: int, gres: str, mem: str, additional_vars : str = ""):

    num_gpus = int(gres.split(":")[-1])
    cuda_visible_devices = ",".join(list(range(num_gpus)))
    

    repo_root = Path(__file__).resolve().parent.parent

    body = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={stdout_path}
#SBATCH --error={stderr_path}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time=1-00:00:00
#SBATCH --gres={gres}
#SBATCH --mem={mem}

{additional_vars}
export DEEPSPEED_LOG_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES={cuda_visible_devices}
export PYTHONUNBUFFERED=1

cd {repo_root}
lake build eval_improver
sleep 5

export run_id=\"{run_id}\"
export prompt_id=\"{prompt_id}\"
export split=\"{split}\"
export model=\"{model}\"

./improver run pipeline --run_id $run_id --prompt_id $prompt_id --split $split --model $model --config {inference_config}
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w") as f:
        f.write(body)
    script_path.chmod(0o755)

    
    
    
def make_training_sbatch(train_config_path: Path, base_model: str, model_name: str, datasets: list, output_dir: str, hub_model_id: str, wandb_project: str, script_path: Path, stdout_path: Path, stderr_path: Path, job_name: str, cpus: int, gres: str, mem: str, additional_vars: str = ""):

    num_gpus = int(gres.split(":")[-1])
    cuda_visible_devices = ",".join(list(range(num_gpus)))
    

    repo_root = Path(__file__).resolve().parent.parent
    
    body = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={stdout_path}
#SBATCH --error={stderr_path}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time=1-00:00:00
#SBATCH --gres={gres}
#SBATCH --mem={mem}

{additional_vars}
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES={cuda_visible_devices}

export base_model=\"{base_model}\"
export model_name=\"{model_name}\"
export datasets=\"{datasets}\"
export output_dir=\"{output_dir}\"
export hub_model_id=\"{hub_model_id}\"
export wandb_project=\"{wandb_project}\"

cd {repo_root}

accelerate launch -m \
    axolotl.cli.train {train_config_path} \
    --base_model $base_model \
    --datasets $datasets \
    --output_dir $output_dir \
    --hub_model_id $hub_model_id \
    --wandb_project $wandb_project
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w") as f:
        f.write(body)
    script_path.chmod(0o755)






#     body = sbatch_header(job_name, cpus, gres, mem, extra="#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1")
#     body += """
# source $HOME/miniconda3/bin/activate env
# export NCCL_DEBUG=INFO
# export HF_HOME="/data/user_data/$USER/HF"
# export PYTHONUNBUFFERED=1

# accelerate launch --main-process-port=29501 -m \
#     axolotl.cli.train {TRAIN_CONFIG_PATH} \
#     --deepspeed /home/$USER/deepspeed_configs/zero3_bf16_cpuoffload_params.json
# """.replace("{TRAIN_CONFIG_PATH}", str(train_config_path))
#     script_path.parent.mkdir(parents=True, exist_ok=True)
#     script_path.write_text(body)
#     script_path.chmod(0o755)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Automate expert iteration with Slurm")
    parser.add_argument("--inf-template", required=True, type=Path, help="Path to core inference config YAML template")
    parser.add_argument("--train-template", required=True, type=Path, help="Path to core training config YAML template")
    
    parser.add_argument("--base-model", required=True, help="Initial base model identifier/path")
    parser.add_argument("--base-name", required=True, help="Short human-readable identifier, e.g. DS2")
    parser.add_argument("--prompt-id-test", default="final_final_test", help="Prompt ID to use for test inference")
    parser.add_argument("--prompt-id-train", default="final_final_train", help="Prompt ID to use for train inference")
    
    
    
    parser.add_argument("--iterations", required=True, type=int, help="Number of expert iterations to perform")
    
    parser.add_argument("--cpus", required=True, type=int, help="cpus-per-task for SBATCH")
    parser.add_argument("--gres", required=True, help="Slurm gres string, e.g. gpu:A6000:8")
    parser.add_argument("--mem", required=True, help="Memory spec for SBATCH, e.g. 150G")
    
    parser.add_argument("--output-dir", default="/data/user_data/riyaza/saved_models", help="Output directory for the model")
    parser.add_argument("--HF-username", default="riyazahuja", help="HF username")
    
    parser.add_argument("--start_iteration", default=0, type=int, help="starting iteration")
    
    args = parser.parse_args()

    inf_template = load_yaml(args.inf_template)
    train_template = load_yaml(args.train_template)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    current_model = args.base_model
    
    prompt_id_map = {"test": args.prompt_id_test, "train": args.prompt_id_train}
    additional_vars = "source $HOME/miniconda3/bin/activate env\nexport HF_HOME=\"/data/user_data/riyaza/HF\""
    
    for iteration in range(args.start_iteration, args.iterations):
        iteration_tag = f"iteration_{iteration}"
        print(f"\n=== Iteration {iteration}/{args.iterations} ({iteration_tag}) ===")

        # ------------------------------------------------------------------
        # 1. Inference (train & test)
        # ------------------------------------------------------------------
        inference_jobids = []
        
        
        train_run_id = ""
        for split in ("test", "train"):
            run_id = f"{args.base_name}_{split}_{iteration_tag}"
            if split == "train":
                train_run_id = run_id
            prompt_id = prompt_id_map[split]
            # we already have split and current_model
            
            #for the sbatch, we need to make a job_name, stdout path and stderr path, and a script path
            
            inf_cfg_dir = Path(args.inf_template).parent
            script_path = inf_cfg_dir / f"{run_id}.sh"
            inf_cfg_dir.mkdir(exist_ok=True)
            
            log_dir = Path("logs") / args.base_name
            log_dir.mkdir(exist_ok=True)
            stdout_path = log_dir / f"{run_id}.out"
            stderr_path = log_dir / f"{run_id}.err"
            
            job_name = run_id
            
            make_inference_sbatch(inf_template, run_id, prompt_id, split, current_model, script_path, stdout_path, stderr_path, job_name,args.cpus, args.gres, args.mem, additional_vars)
            
            
            jobid = submit_and_get_jobid(script_path)
            inference_jobids.append(jobid)

        dependency_str = ":".join(inference_jobids)

        # ------------------------------------------------------------------
        # 2. Training
        # ------------------------------------------------------------------
        
        # for training config, we need a base_model, model_name, dataset path
        base_model = current_model
        model_name = f"{args.base_name}_{iteration_tag}"
        run_id = f"{args.base_name}_train_{iteration_tag}"
        training_dataset_path = f"evals/{train_run_id}/analysis/BoN/train.jsonl"
        
        datasets = json.loads([{"path": training_dataset_path, "type": "alpaca"}])
        
        output_dir = args.output_dir
        
        hub_model_id = f"{args.HF_username}/{model_name}"
        wandb_project = model_name
        
        # for the sbatch, we need a job_name, stdout path and stderr path, and a script path
        train_cfg_dir = Path(args.train_template).parent
        script_path = train_cfg_dir / f"{run_id}.sh"
        train_cfg_dir.mkdir(exist_ok=True)
        
        log_dir = Path("logs") / args.base_name
        log_dir.mkdir(exist_ok=True)
        stdout_path = log_dir / f"{run_id}.out"
        stderr_path = log_dir / f"{run_id}.err"
        
        job_name = run_id
        
        make_training_sbatch(train_template, base_model, model_name, datasets, output_dir, hub_model_id, wandb_project, script_path, stdout_path, stderr_path, job_name, args.cpus, args.gres, args.mem, additional_vars)

        train_jobid = submit_and_get_jobid(script_path, dependency=dependency_str)

        # Wait for training to finish before next iteration
        wait_for_job(train_jobid)

        # Update model path for next iteration
        current_model = str(output_dir)

    print("\nAll iterations completed successfully.")


if __name__ == "__main__":
    main() 