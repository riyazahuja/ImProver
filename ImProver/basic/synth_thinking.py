#!/usr/bin/env python

import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
import os
import torch
import pandas as pd
import json
import datetime
import multiprocessing
import argparse
import duckdb

# --- Minimal changes for synthetic reasoning trace generation ---
def load_train_jsonl(run_id):
    path = f"evals/{run_id}/analysis/BoN/train.jsonl"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training data not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines

def build_reasoning_prompt(instruction, output):
    # You can further customize this prompt as needed
    return (
        "You are a helpful mathematician whose goal is to retroactively construct an informal reasoning trace. "
        "Namely, your mathematician peer was given an instruction to optimize a current Lean4 theorem for some specified metric, and also given some metadata and context on that theorem. "
        "They successfully optimized the theorem and produced an output, but did not include any reasoning on how or why they did it, nor did they detail their thought process in any way. "
        "As a fellow mathematician, your goal is to reconstruct the reasoning that your mathematician peer likely used in their thought process to optimize the input instruction to produce the model_output, as if you were the original mathematician yourself. "
        "You will be provided the original instruction (containing the theorem, optimization metric, etc.) that the peer recieved, as well as their output (tagged in <INSTRUCTION> and <MODEL_OUTPUT> respectively). "
        "Analyze both and reconstruct the reasoning that your mathematician peer used to optimize or prove the input instruction to produced the model_output, as if you were the original model. "
        "Return ONLY the full chain-of-thought text of the reasoning process (no extra prose, no tags). Be detailed and thorough.\n\n"
        f"<INSTRUCTION>\n{instruction}\n</INSTRUCTION>\n\n<MODEL_OUTPUT>\n{output}\n</MODEL_OUTPUT>\n"
    )

def main(args):
    # Load input data
    train_data = load_train_jsonl(args.run_id)
    df = pd.DataFrame([
        {
            "idx": i,
            "instruction": ex["instruction"],
            "original_output": ex["output"],
            "raw_prompt": build_reasoning_prompt(ex["instruction"], ex["output"]),
        }
        for i, ex in enumerate(train_data)
    ])

    # Ray/vLLM setup (copied from inference.py)
    if args.nccl_p2p:
        os.environ["NCCL_P2P_DISABLE"] = "0"
    else:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    ray.init(num_cpus=args.cpus, num_gpus=args.gpus)
    DataContext.get_current().wait_for_min_actors_s = args.ray_timeout
    assert Version(ray.__version__) >= Version("2.44.1")

    ds = ray.data.from_pandas(df).repartition(args.num_blocks)
    print(ds.schema())
    print(f"Size of dataset: {ds.count()} prompts")

    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_resources={"CPU": args.engine_cpu_resources, "GPU": args.engine_gpu_resources},
        concurrency=args.concurrency,
        engine_kwargs={
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
        max_concurrent_batches=args.max_concurrent_batches,
        batch_size=args.batch_size,
    )

    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["raw_prompt"]}],
            sampling_params=dict(
                truncate_prompt_tokens=args.truncate_prompt_tokens,
                max_tokens=args.max_tokens,
            ),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )
    ds = vllm_processor(ds).materialize()

    # Write output in the required format
    output_path = f"evals/{args.run_id}/analysis/BoN/train_thinking.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds.iter_rows():
            cot = row["answer"]
            rec = {
                "instruction": row["instruction"],
                # "original_output": row["original_output"],
                # "cot": cot,
                "augmented_output": f"<think>\n{cot.replace('<think>', '').replace('</think>', '')}\n</think>\n{row['original_output']}"
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {ds.count()} records to {output_path}")

def get_parser():
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning traces using vLLM")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID (input: evals/<run_id>/analysis/BoN/train.jsonl)")
    parser.add_argument("--model", type=str, required=True, help="vLLM model path or name")
    parser.add_argument("--cpus", type=int, default=multiprocessing.cpu_count(), help="Number of CPUs to use")
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--num_blocks", type=int, default=16, help="Number of blocks to repartition the dataset into")
    parser.add_argument("--engine_cpu_resources", type=int, default=multiprocessing.cpu_count(), help="Number of CPU resources for the engine")
    parser.add_argument("--engine_gpu_resources", type=int, default=1, help="Number of GPU resources for the engine")
    parser.add_argument("--concurrency", type=int, default=torch.cuda.device_count(), help="Concurrency for the engine")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for the engine")
    parser.add_argument("--enable_chunked_prefill", type=bool, default=True, help="Enable chunked prefill for the engine")
    parser.add_argument("--max_model_len", type=int, default=16384, help="Maximum model length for the engine")
    parser.add_argument("--max_num_batched_tokens", type=int, default=65536, help="Maximum number of batched tokens for the engine")
    parser.add_argument("--max_concurrent_batches", type=int, default=32, help="Maximum number of concurrent batches for the engine")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the engine")
    parser.add_argument("--truncate_prompt_tokens", type=int, default=16384-2048, help="Number of prompt tokens to truncate")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--nccl_p2p", type=bool, default=False, help="Enable NCCL P2P")
    parser.add_argument("--ray_timeout", type=int, default=1800, help="Ray timeout in seconds")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
