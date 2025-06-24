#!/usr/bin/env python3
"""Driver script for the full proof evaluation pipeline."""
import argparse
import datetime
import os
import sys
import yaml

from improver_cli import (
    run_get_prompts,
    run_inference,
    run_eval,
    run_analysis,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_prompts(cfg: dict) -> None:
    """Generate prompts if needed and update prompts_dir."""
    base_dir = cfg.get("prompts_dir", ".prompts")
    prompt_id = cfg.get("prompt_id")
    if not prompt_id:
        prompt_id = f"PROMPTS_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cfg["prompt_id"] = prompt_id
        run_get_prompts(cfg)
    cfg["prompts_dir"] = os.path.join(base_dir, prompt_id)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_prompts(cfg)

    if "runID" not in cfg or cfg["runID"] is None:
        cfg["runID"] = f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_inference(cfg)
    run_eval(cfg)
    run_analysis(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full ImProver proof evaluation pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args)
