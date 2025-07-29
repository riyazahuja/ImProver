import argparse
import torch
import multiprocessing
import datetime
from .preprocess_rag import main as preprocess_rag_main
from .informalize import main as informalize_main
from .build_db import main as build_db_main




    

def main(args):
    
    # MAX_PROMPT_TOKENS = 16384 - 2048   # model context minus generation tokens

    # preprocess_rag_main(args)
    # informalize_main(args)
    build_db_main(args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize RAG source")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--rag_id", type=str, default=f"rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--max_depth", type=int, default=2)
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