import yaml
import os
import argparse


def create_train_config(
    base_model: str,
    beta: float,
    alpha: float,
    lr: float,
    dataset_path: str,
    output_dir: str,
    project_name: str,
    config_path: str,
    epochs: int = 1,
):
    """
    Create a training configuration file with the specified parameters.

    Args:
        base_model: Path to the base model
        beta: RL beta parameter
        alpha: RPO alpha parameter
        lr: Learning rate
        dataset_path: Path to the dataset
        output_dir: Output directory for training artifacts
        project_name: Weights & Biases project name
        config_path: Path where the config file will be created
    """
    config = {
        "group_by_length": True,
        "base_model": base_model,
        "ref_model": base_model,
        "trust_remote_code": True,
        "rl": "dpo",
        "rl_beta": beta,
        "rpo_alpha": alpha,
        "sequence_len": 8192,
        "flash_attn_2": True,
        "gradient_checkpointing": True,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_epochs": epochs,
        "learning_rate": lr,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "warmup_steps": 5,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "deepspeed": "/home/riyaza/eval_improver/improver/deepspeed_configs/zero3_bf16_cpuoffload_all_custom.json",
        "bf16": True,
        "datasets": [{"path": dataset_path, "type": "chatml.prompt_pairs"}],
        "val_set_size": 0.05,
        "output_dir": output_dir,
        "wandb_project": project_name,
        "logging_steps": 10,
        "evals_per_epoch": 1,
        "save_strategy": "no",
        "save_only_model": True,
        "seed": 42,
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Write config to file
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a training configuration file")
    parser.add_argument(
        "--base-model", type=str, required=True, help="Path to the base model"
    )
    parser.add_argument("--beta", type=float, required=True, help="RL beta parameter")
    parser.add_argument(
        "--alpha", type=float, required=True, help="RPO alpha parameter"
    )
    parser.add_argument(
        "--learning-rate", type=float, required=True, help="Learning rate"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for training artifacts",
    )
    parser.add_argument(
        "--project-name", type=str, required=True, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path where the config file will be created",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )

    args = parser.parse_args()

    create_train_config(
        base_model=args.base_model,
        beta=args.beta,
        alpha=args.alpha,
        lr=args.learning_rate,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        project_name=args.project_name,
        config_path=args.config_path,
        epochs=args.epochs,
    )

    print(f"Training config created at: {args.config_path}")
