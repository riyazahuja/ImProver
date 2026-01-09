import argparse
import torch
import multiprocessing
import datetime
from .preprocess_rag import main as preprocess_rag_main
from .informalize import main as informalize_main
from .build_db import main as build_db_main




    

def main(args):
    
    # MAX_PROMPT_TOKENS = 16384 - 2048   # model context minus generation tokens

    preprocess_rag_main(args)
    informalize_main(args)
    build_db_main(args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize RAG source")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--rag_id", type=str, default=f"rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--split",type=str, default=None)
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
    
    
    
    
    
    parser.add_argument(
        "--nccl_p2p",
        type=bool,
        default=False,
        help="Enable NCCL P2P - set to false if nvidia-smi topo -m shows SYS between gpus, or something or another about PCIE? A6000 -> false. (default: False)",
    )
    parser.add_argument(
        "--ray_timeout",
        type=int,
        default=1800,
        help="Ray timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=16,
        help="Number of blocks to repartition the dataset into (default: 16)",
    )
    parser.add_argument(
        "--engine_cpu_resources",
        type=int,
        default=multiprocessing.cpu_count() // available_gpus,
        help="Number of CPU resources for the engine (default: cpus // gpus)",
    )
    parser.add_argument(
        "--engine_gpu_resources",
        type=int,
        default=1,
        help="Number of GPU resources for the engine (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=available_gpus,
        help="Concurrency for the engine (default: gpus)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for the engine (default: 1)",
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        type=bool,
        default=True,
        help="Enable chunked prefill for the engine (default: True)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="Maximum model length for the engine (default: 16384)",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=65536,
        help="Maximum number of batched tokens for the engine (default: 65536)",
    )
    parser.add_argument(
        "--max_concurrent_batches",
        type=int,
        default=32,
        help="Maximum number of concurrent batches for the engine (default: 32)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the engine (default: 32)",
    )
    parser.add_argument(
        "--truncate_prompt_tokens",
        type=int,
        default=16384 - 2048,
        help="Number of prompt tokens to truncate (default: 16384 - 2048)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate (default: 2048)",
    )
    
    
    
    args = parser.parse_args()
    
    main(args)