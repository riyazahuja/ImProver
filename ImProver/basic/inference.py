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

# NCCL_P2P (default is False, A6000)
# ray_timeout (def 1800s)
# num_blocks (default 16)
# engine_cpu_resources = args.cpus // args.gpus
# engine_gpu_resources = 1
# concurrency=args.gpus
# tensor_parallel_size=1
# enable_chunked_prefill=True
# max_model_len=16384
# max_num_batched_tokens=65536
# max_concurrent_batches=32
# batch_size=32
# truncate_prompt_tokens=16384-2048
# max_tokens=2048

def run_inference(df, args, ray_init=True):
    # assuming gpus sit behind different PCIe host bridges on separate
    # NUMA sockets (i.e. nvidia-smi topo -m shows SYS between gpus)
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    if args.NCCL_P2P:
        os.environ["NCCL_P2P_DISABLE"] = "0"
    else:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        
    if ray_init:
        ray.init(num_cpus=args.cpus, num_gpus=args.gpus)#, _temp_dir='/home/riyaza/ray_tmp')
    DataContext.get_current().wait_for_min_actors_s = args.ray_timeout
    ctx = DataContext.get_current()
    # ctx.progress_bar = True
    # ctx.execution_options.verbose_progress = True
    
    assert Version(ray.__version__) >= Version(
        "2.44.1"
    ), "Ray version must be at least 2.44.1"

    # ds = ray.data.from_pandas(df)
    # Create a new dataframe with duplicated rows, each with a unique prompt_idx
    df2_parts = []
    for i in range(args.n):
        df_copy = df.copy()
        df_copy['prompt_idx'] = i
        df2_parts.append(df_copy)

    df2 = pd.concat(df2_parts, ignore_index=True)
    # Use df2 instead of df for the Ray dataset
    ds = ray.data.from_pandas(df2).repartition(args.num_blocks)
    # ds = ray.data.from_pandas(df).repartition(args.gpus * 4)
    # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
    print(ds.schema())

    size = ds.count()
    print(f"Size of dataset: {size} prompts")

    # ctx.execution_options = ExecutionOptions(task_extra_resources={"CPU": 0.25})

    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_resources={"CPU": args.engine_cpu_resources, "GPU": args.engine_gpu_resources},
        concurrency=args.concurrency,
        engine_kwargs={
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            # "max_num_batched_tokens": 4096,
            # "max_model_len": 16384,
            
        },
        max_concurrent_batches=args.max_concurrent_batches,
        batch_size=args.batch_size,
    )

    
    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["raw_prompt"]}],
            sampling_params=dict(
                # n=args.n,
                truncate_prompt_tokens=args.truncate_prompt_tokens,
                # temperature=0.3,
                max_tokens=args.max_tokens,
            ),
        ),
        postprocess= lambda row : dict(answer=row["generated_text"], **row),
    )
    ds = vllm_processor(ds).materialize()

    run_output_dir = os.path.join("evals", args.runID)
    os.makedirs(run_output_dir, exist_ok=True)

    output_path = os.path.join(run_output_dir, "data")

    config = vars(args)

    config_path = os.path.join(run_output_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    ds.repartition(16).write_parquet(f"local://{output_path}")

    con = duckdb.connect(os.path.join(run_output_dir, "data.duckdb"))
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS run_data AS
        SELECT * FROM read_parquet('{output_path}/*.parquet');
    """
    )

    return run_output_dir


def construct_prompts(config_data, data, args):
    # config_data is metric config data
    # data is the prompt data
    idx = 0
    items = []
    
    
    
    for item in data:
        if item['id']['isExtracted'] or len(item['id']['errorMsgs']) !=0:
            continue
        
        name = item["id"]["name"]

        prompt = config_data["prompts"]["system_prompt"]+"\n"
        
        if args.examples != 0:
            prompt += config_data["prompts"]["example_prompt"] + "\n"

        if args.annotation:
            prompt += config_data["prompts"]["annotation_prompt"] + "\n"

        if args.context != 0:
            prompt += config_data["prompts"]["context_prompt"] + "\n"

        if args.rag != 0:
            prompt += config_data["prompts"]["rag_prompt"] + "\n"

        prompt += "\n"

        if args.examples != 0:
            example_data_path = config_data["examples"]["example_data"]

            with open(example_data_path, "r") as f:
                examples_data = json.load(f)
            
            
            prompt += f"<EXAMPLES>\n\n"
            for nameTag, example in examples_data.items()[: min(args.examples,len(examples_data.items()))]:
                try:
                    ex_prompt = "<EXAMPLE>\n\n"
                    if args.context:
                        ex_prompt += f"<CONTEXT>\n"
                        for context in example["C1_dependencies"]:
                            ex_prompt += f"<ITEM>\n--name={context['name']}\n--type={context['kind']}\n{context['content']}\n</ITEM>\n"
                        ex_prompt += f"</CONTEXT>\n\n"
                    if args.rag != 0:
                        ex_prompt += f"<RAG>\n"
                        for rag in example["rag"][: args.rag]:
                            ex_prompt += (
                                f"<DOC>\n{rag}\n</DOC>\n"
                            )
                        ex_prompt += f"</RAG>\n\n"
                    if args.annotation:
                        ex_prompt += (
                            f"<ANNOTATION>\n{example['annotation']}\n</ANNOTATION>\n\n"
                        )
                    ex_prompt += f"<CURRENT>\n{example['id']['content']}\n</CURRENT>\n\n"
                    ex_prompt += f"<IMPROVED>\n{example['improved']}\n</IMPROVED>\n\n"
                    ex_prompt += f"</EXAMPLE>\n\n"
                    prompt += ex_prompt
                except:
                    pass
            prompt += f"</EXAMPLES>\n\n"

        if args.context != 0:
            
            prompt += f"<CONTEXT>\n"
            for context in item["C1_dependencies"][: min(args.context,len(item["C1_dependencies"]))]:
                prompt += f"<ITEM>\n--name={context['name']}\n--type={context['kind']}\n{context['content']}\n</ITEM>\n"
            prompt += f"</CONTEXT>\n\n"

        if args.rag != 0:
            prompt += f"<RAG>\n"
            for rag in item["rag"][: min(args.rag,len(item["rag"]))]:
                prompt += f"<DOC>\n{rag}\n</DOC>\n"
            prompt += f"</RAG>\n\n"

        if args.annotation:
            prompt += f"<ANNOTATION>\n{item['annotation']}\n</ANNOTATION>\n\n"

        prompt += f"\n<CURRENT>\n{item['id']['content'] if args.metric!="completion" else item['content_sorry']}\n</CURRENT>\n\n"
        prompt += "<IMPROVED>"

        data = {
            "decl": name,
            "decl_idx": idx,
            "raw_prompt": prompt,
        }
        items.append(data)
        idx += 1
    return items


def main(args):    
    with open(args.dataset_path, "r") as f:
        all = json.load(f)
        dataset = all[args.split]
    files_to_process = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]
    prompt_root = os.path.join("prompts", args.prompt_id)
    metric_root = os.path.join("metrics", args.metric)
    config_path = os.path.join(metric_root, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    df = pd.DataFrame(columns=["module", "decl", "decl_idx", "raw_prompt"])
    for file_info in files_to_process:
        file = file_info if type(file_info) is str else file_info["file"]
        
        file_path = os.path.join(prompt_root, "src", file.replace(".lean", ".json"))
        # file_path = os.path.join(prompt_root, file.replace(".lean", ".json")) #LEGACY, REVERT!
        module = file.replace(".lean", "").replace("/", ".")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data_raw = json.load(f)
                prompt_data = construct_prompts(config_data, data_raw, args)
            print(f"Processing {file_path} with {len(prompt_data)} prompts")

            
            for item in prompt_data:
                df.loc[len(df)] = [module,
                        item["decl"],
                        item["decl_idx"],
                        item["raw_prompt"]]


    # returns the path to the directory containing run metadata and the parquet lake
    output_path = run_inference(df, args)

    # we should also initialize + index the duckDB stuff


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("prompt_id", type=str, help="Prompt ID to use")
    parser.add_argument(
        "--runID",
        type=str,
        default="RUN_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run ID to use for evaluation (default: run_<timestamp>)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Model to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
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
    parser.add_argument("--n", type=int, default=1, help="Best-of-n value (default: 1)")
    parser.add_argument(
        "--annotation", type=bool, default=False, help="Annotation? (default: False)"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Number of context retrievals (default: 0)",
    )
    parser.add_argument(
        "--rag", type=int, default=0, help="Number of RAG retrievals (default: 0)"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of few-shot example retrievals (default: 0)",
    )
    parser.add_argument(
        "--NCCL_P2P",
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
