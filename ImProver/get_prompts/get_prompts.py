import pandas as pd
import json
import tqdm
import os
from pathlib import Path
import re
import subprocess
import multiprocessing
import asyncio
import sys
import time
from datetime import datetime
import argparse
from multiprocessing import cpu_count
import torch
import json
from .extraction import main as extraction_main
from .informalize import main as informalizer_main
from .rag import main as rag_main
    
    
def main(args):
    
    print("[IMPROVER: Extracting theorems...]")
    extraction_main(args)
    
    print("[IMPROVER: Informalizing theorems...]")
    informalizer_main(args)
    
    print("[IMPROVER: Running RAG...]")
    rag_main(args)
    
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("--prompts_id", type=str, default="prompts_" + datetime.now().strftime("%Y%m%d_%H%M%S")),

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    
    
    
    parser.add_argument("--include_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

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


