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
    
    # Run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--runID", type=str, default=f"RUN_{timestamp}", help="Run identifier")
    
    # Analysis settings
    parser.add_argument("--training_data", action=argparse.BooleanOptionalAction, help="Whether to extract training data", default=True)
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
        runID=args.runID
    )
    inference_main(inference_args)
    
    # 2. Run Evaluation
    print(f"[IMPROVER: Evaluating run {args.runID}...]")
    eval_args = argparse.Namespace(
        runID=args.runID,
        # inference_dir=args.output_dir,
        cpus=args.cpus
    )
    eval_main(eval_args)
    
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
        print(f"[IMPROVER: Running llm metric analysis for run {args.runID}...]")
        llm_args = argparse.Namespace(
            runID=args.runID,
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

    
    print(f"[IMPROVER: Analyzing run {args.runID}...]")
    analysis_args = argparse.Namespace(
        RunID=args.runID,
        # run_dir=args.output_dir,
        training_data=args.training_data
    )
    analysis_main(analysis_args)
    
    
    
    print(f"[IMPROVER: ImProver pipeline completed for run {args.runID}]")

    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main()


