import os
import json
import re
import argparse
from tqdm import tqdm
import pandas as pd
import duckdb
# import ray
# from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
import multiprocessing
# import torch
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# from transformers import AutoTokenizer
# from .informalize import collect_prompts, run_inference, populate_database, build_prompt


#rag process first preprocesses the lean files to get the informal data and importgraphs
# we then run the inference on the importgraphs to get the informal data
# and then we initialize the rag database 


def preprocess_prompts(args):


    with open(args.dataset_path, "r") as f:
        all_ds = json.load(f)
        all_splits = all_ds.keys()
        dataset = []
        for split in all_splits:
            dataset.extend(all_ds[split].values())
    files = []
    print(dataset)
    for repo in dataset:
        files.extend(repo)
    files_real = [file_info if type(file_info) is str else file_info["file"] for file_info in files]
    modules = set(f.replace(".lean", "").replace("/", ".") for f in files_real)
    modules_str = ",".join(sorted(modules))

    prompts_dir = os.path.join("rag", args.rag_id)

    os.makedirs(prompts_dir, exist_ok=True)

    cmd = f"lake exe build_rag {modules_str} {prompts_dir}"
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {cmd}")


def get_prompts(args):
    decls_path = os.path.join("rag", args.rag_id, "decl_data.json")
    with open(decls_path, "r") as f:
        decls = json.load(f)
        
    all_prompts = [{"prompt":build_prompt(decl),
                    "module" : decl['module'],
                    "name" : decl['decl'],
                    "text" : decl['content']} for decl in decls]
    with open(os.path.join("rag", args.rag_id, "prompts.json"), "w") as f:
        json.dump(all_prompts, f)
    
    df = pd.DataFrame(all_prompts)
    return df
    
    
    
    


    

def main(args):
    
    # MAX_PROMPT_TOKENS = 16384 - 2048   # model context minus generation tokens
    
    preprocess_prompts(args)
    # df = get_prompts(args)
    # run_inference(df, args)
    # create_vectordb(args.rag_id)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize RAG source")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("rag_id", type=str)
    # parser.add_argument("--split", type=str, default="train")
    # parser.add_argument("--include_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    # try:
    #     available_gpus = torch.cuda.device_count()
    # except (ImportError, AttributeError):
    #     available_gpus = 0

    # parser.add_argument(
    #     "--gpus",
    #     type=int,
    #     default=available_gpus,
    #     help="Number of GPUs to use (default: all available)",
    # )
    args = parser.parse_args()
    
    main(args)