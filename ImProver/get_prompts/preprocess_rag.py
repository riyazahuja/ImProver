import os
import json
import re
import argparse
from tqdm import tqdm
import pandas as pd
import duckdb
import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
import multiprocessing
import torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import AutoTokenizer
from .informalize import collect_prompts, run_inference, populate_database


#rag process first preprocesses the lean files to get the informal data and importgraphs
# we then run the inference on the importgraphs to get the informal data
# and then we initialize the rag database 


def preprocess_prompts(args):


    with open(args.dataset_path, "r") as f:
        all_ds = json.load(f)
        dataset = all_ds[args.split]
    files = []
    for repo in dataset.values():
        files.extend(repo)
    files_real = [file_info if type(file_info) is str else file_info["file"] for file_info in files]
    modules = set(f.replace(".lean", "").replace("/", ".") for f in files_real)
    modules_str = ",".join(sorted(modules))

    prompts_dir = os.path.join("prompts", args.prompts_id)


    cmd = f"lake exe preprocess_rag \"{modules_str}\" \"{prompts_dir}\""
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {cmd}")


def main(args):
    
    MAX_PROMPT_TOKENS = 16384 - 2048   # model context minus generation tokens
    
    preprocess_prompts(args)
    
    #now dedup all the newly preprocessed items
    #and use this to call some lean script to get the source code of each statement in each file
    #
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Informalize theorems")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("prompts_id", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--include_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    try:
        available_gpus = torch.cuda.device_count()
    except (ImportError, AttributeError):
        available_gpus = 0

    parser.add_argument(
        "--gpus",
        type=int,
        default=available_gpus,
        help="Number of GPUs to use (default: all available)",
    )
    args = parser.parse_args()
    
    main(args)