import os
import torch
import datetime
import multiprocessing
import argparse
from .inference import main as inference_main
from .eval_improver import main_async as eval_main
from .analysis import main as analysis_main
from .llm_metric import main as llm_main
import json
import asyncio


def get_parser():
    parser = argparse.ArgumentParser(description="ImProver: Inference, Evaluation, and Analysis Tool")
    
    # Required arguments
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("prompt_id", type=str, help="Prompt ID to use")
    
    # Optional arguments
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B", help="Model to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    # parser.add_argument("--prompts_dir", type=str, default=".prompts/", help="Directory of prompt data")
    # parser.add_argument("--output_dir", type=str, default=".evals/", help="Directory to output runs")
    
    # System resource arguments
    try:
        available_gpus = torch.cuda.device_count()
    except (ImportError, AttributeError):
        available_gpus = 0
        
    parser.add_argument("--cpus", type=int, default=multiprocessing.cpu_count(), help="Number of CPUs to use")
    parser.add_argument("--gpus", type=int, default=available_gpus, help="Number of GPUs to use")
    
    # Generation settings
    parser.add_argument("--n", type=int, default=1, help="Best-of-n value")
    parser.add_argument("--annotation", action=argparse.BooleanOptionalAction, default=False, help="Enable annotation")
    parser.add_argument("--context", type=int, default=0, help="Number of context retrievals")
    parser.add_argument("--rag", type=int, default=0, help="Number of RAG retrievals")
    parser.add_argument("--examples", type=int, default=0, help="Number of few-shot example retrievals")
    parser.add_argument(
        "--goal_state", action=argparse.BooleanOptionalAction, default=False, help="Enable goal state"
    )
    parser.add_argument(
        "--file_context",
        type=int,
        default=0,
        help="Number of file context items (default: 0, -1 for all)",
    )
    
    #inference hyperparams
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
    
    
    # Run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--run_id", type=str, default=f"RUN_{timestamp}", help="Run identifier")
    
    # Analysis settings
    parser.add_argument("--training_data", action=argparse.BooleanOptionalAction, help="Whether to extract training data", default=True)
    parser.add_argument("--thinking", default="none", help="Thinking mode for analysis (default: none). Options: 'none', 'raw'.")
    return parser

def main(args):
        
    # Create output directory if it doesn't exist
    os.makedirs("evals", exist_ok=True)
    

    # 1. Run Inference
    print(f"[IMPROVER: Running inference for {args.metric} with prompt {args.prompt_id}...]")
    inference_args = argparse.Namespace(
        metric=args.metric,
        dataset_path=args.dataset_path,
        prompt_id=args.prompt_id,
        model=args.model,
        split=args.split,
        # prompts_dir=args.prompts_dir,
        # output_dir=args.output_dir,
        cpus=args.cpus,
        gpus=args.gpus,
        n=args.n,
        annotation=args.annotation,
        context=args.context,
        rag=args.rag,
        examples=args.examples,
        goal_state=args.goal_state,
        file_context=args.file_context,
        
        
        nccl_p2p=args.nccl_p2p,
        ray_timeout=args.ray_timeout,
        num_blocks=args.num_blocks,
        engine_cpu_resources=args.engine_cpu_resources,
        engine_gpu_resources=args.engine_gpu_resources,
        concurrency=args.concurrency,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_concurrent_batches=args.max_concurrent_batches,
        batch_size=args.batch_size,
        truncate_prompt_tokens=args.truncate_prompt_tokens,
        max_tokens=args.max_tokens,
        
        run_id=args.run_id
    )
    inference_main(inference_args)
    
    # 2. Run Evaluation
    print(f"[IMPROVER: Evaluating run {args.run_id}...]")
    eval_args = argparse.Namespace(
        run_id=args.run_id,
        # inference_dir=args.output_dir,
        cpus=args.cpus
    )
    asyncio.run(eval_main(eval_args))
    
    # 3. Run Analysis
    
    
    
    # Load metric configuration
    config_path = f"metrics/{args.metric}/config.json"
    try:
        with open(config_path, 'r') as f:
            metric_config = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        metric_config = {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in config file at {config_path}")
        metric_config = {}
    
    # 4. If metric is llm, run llm metric
    if metric_config.get("llm",{}).get("llm_metric",False):
        print(f"[IMPROVER: Running llm metric analysis for run {args.run_id}...]")
        llm_args = argparse.Namespace(
            run_id=args.run_id,
            prompts_id=args.prompt_id,
            inference=True,
            # output_dir=args.output_dir,
            model=args.model,
            split=args.split,
            # prompts_dir=args.prompts_dir,
            cpus=args.cpus,
            gpus=args.gpus,
            n=3  # Default for llm is 3
        )
        llm_main(llm_args)

    
    print(f"[IMPROVER: Analyzing run {args.run_id}...]")
    analysis_args = argparse.Namespace(
        run_id=args.run_id,
        # run_dir=args.output_dir,
        training_data=args.training_data,
        thinking=args.thinking
    )
    analysis_main(analysis_args)
    
    
    
    print(f"[IMPROVER: ImProver pipeline completed for run {args.run_id}]")

    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)


