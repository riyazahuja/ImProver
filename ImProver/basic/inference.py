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
from transformers import AutoTokenizer

# nccl_p2p (default is False, A6000)
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
    if args.nccl_p2p:
        os.environ["NCCL_P2P_DISABLE"] = "0"
    else:
        os.environ["NCCL_P2P_DISABLE"] = "1"

    # if ray_init:
    #     try:
    #         ray.init(num_cpus=args.cpus, num_gpus=args.gpus)
    #     except:
    #         ray.init(
    #             num_cpus=args.cpus,
    #             num_gpus=args.gpus,
    #             _temp_dir="/data/user_data/riyaza/ray_tmp",
    #         )
    tmp_dir = os.environ.get("RAY_TMPDIR", f"/data/user_data/{os.getenv('USER','user')}/ray_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ray.init(num_cpus=args.cpus, num_gpus=args.gpus, _temp_dir=tmp_dir)
        

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
        df_copy["prompt_idx"] = i
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
        engine_resources={
            "CPU": args.engine_cpu_resources,
            "GPU": args.engine_gpu_resources,
        },
        concurrency=args.concurrency,
        engine_kwargs={
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            # "max_num_batched_tokens": 4096,
            # "max_model_len": 16384,
            # "gpu_memory_utilization":0.85,
            "swap_space": 16,
        },
        max_concurrent_batches=args.max_concurrent_batches,
        batch_size=args.batch_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess_with_truncation(row):
        # Truncate the prompt to fit within the model's context window
        prompt = row["raw_prompt"]
        tokens = tokenizer.encode(prompt)

        # Calculate available space for prompt (reserve space for generation)
        max_prompt_tokens = args.max_model_len - args.max_tokens

        if len(tokens) > max_prompt_tokens:
            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:max_prompt_tokens]
            prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            print(
                f"Truncated prompt from {len(tokens)} to {len(truncated_tokens)} tokens"
            )

        return dict(
            messages=[{"role": "user", "content": prompt}],
            sampling_params=dict(
                # n=args.n,
                # truncate_prompt_tokens=args.truncate_prompt_tokens,  # Remove this as we're handling truncation manually
                # temperature=0.3,
                max_tokens=args.max_tokens,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.05,
                stop=["</IMPROVED>"],
                seed=int(row.get("prompt_idx", 0)),
            ),
        )

    vllm_processor = build_llm_processor(
        config,
        preprocess=preprocess_with_truncation,  # lambda row: dict(
        #     messages=[{"role": "user", "content": row["raw_prompt"]}],
        #     sampling_params=dict(
        #         # n=args.n,
        #         truncate_prompt_tokens=args.truncate_prompt_tokens,
        #         # temperature=0.3,
        #         max_tokens=args.max_tokens,
        #     ),
        # ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )
    ds = vllm_processor(ds).materialize()

    run_output_dir = os.path.join("evals", args.run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    output_path = os.path.join(run_output_dir, "data")

    config = vars(args)

    config_path = os.path.join(run_output_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    ds.repartition(16).write_parquet(f"local://{output_path}")

    con = duckdb.connect(os.path.join(run_output_dir, "data.duckdb"))
    # Check if run_data exists, drop if so, then create it
    force = True
    if (
        force
        and con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'run_data'"
        ).fetchone()[0]
        > 0
    ):
        con.execute("DROP TABLE run_data")
    con.execute(
        f"""
        CREATE TABLE run_data AS
        SELECT * FROM read_parquet('{output_path}/*.parquet');
        """
    )

    return run_output_dir


def construct_prompt_core(
    config_data,
    item,
    file_context,
    context,
    rag,
    annotation,
    informal,
    goal_state,
    system,
):
    prompt = ""

    if file_context != 0:
        prompt += f"<FILE_CONTEXT>\n"
        num_deps = (
            len(item["C0_dependencies"])
            if file_context == -1
            else min(file_context, len(item["C0_dependencies"]))
        )
        for context in item["C0_dependencies"][:num_deps]:
            prompt += f"<ITEM>\n--name={context['name']}\n--type={context['kind']}\n{context['content']}\n</ITEM>\n"
        prompt += f"</FILE_CONTEXT>\n\n"

    if context != 0:

        prompt += f"<CONTEXT>\n"
        num_deps = (
            len(item["C1_dependencies"])
            if context == -1
            else min(context, len(item["C1_dependencies"]))
        )
        for context in item["C1_dependencies"][:num_deps]:
            prompt += f"<ITEM>\n--name={context['name']}\n--type={context['kind']}\n{context['content']}\n</ITEM>\n"
        prompt += f"</CONTEXT>\n\n"

    if rag != 0:
        prompt += f"<RETRIEVED>\n"
        num_rag = len(item["rag"]) if rag == -1 else min(rag, len(item["rag"]))
        for rag in item["rag"][:num_rag]:
            prompt += f"<DOC>\n{rag}\n</DOC>\n"
        prompt += f"</RETRIEVED>\n\n"

    if annotation:
        prompt += f"<ANNOTATION>\n{item['annotation']}\n</ANNOTATION>\n\n"

    if informal:
        prompt += f"<INFORMAL>\nTheorem: {item['informal_statement']}\n\nProof:\n{item['informal_proof']}\n</INFORMAL>\n\n"

    if goal_state:
        prompt += f"<GOAL_STATE>\n{item['goal_state']}\n</GOAL_STATE>\n\n"

    if system:
        prompt += "As a reminder: " + config_data["prompts"]["system_prompt"] + "\n"

    prompt += f"\n<CURRENT>\n{item['content_sorry'] if config_data["scoring"]["input_sorry"] else item['id']['content']}\n</CURRENT>\n\n"
    # prompt += "<IMPROVED>"

    return prompt


def construct_prompts(config_data, data, args):
    # config_data is metric config data
    # data is the prompt data
    idx = 0
    items = []

    for item in data:
        if item["id"]["isExtracted"] or len(item["id"]["errorMsgs"]) != 0:
            continue

        name = item["id"]["name"]

        prompt = config_data["prompts"]["system_prompt"] + "\n"

        if args.examples != 0:
            prompt += config_data["prompts"]["example_prompt"] + "\n"

        if args.context != 0:
            prompt += config_data["prompts"]["context_prompt"] + "\n"

        if args.file_context != 0:
            prompt += config_data["prompts"]["file_context_prompt"] + "\n"

        if args.rag != 0:
            prompt += config_data["prompts"]["rag_prompt"] + "\n"

        if args.annotation:
            prompt += config_data["prompts"]["annotation_prompt"] + "\n"
        if args.informal:
            try:
                prompt += config_data["prompts"]["informal_prompt"] + "\n"
            except:
                prompt += " An informal (natural language) version of the current theorem and proof has also been provided for reference in your reasoning process to better understand and optimize the structure and intuition behind the theorem (wrapped in <INFORMAL>...</INFORMAL>)."

        if args.goal_state:
            prompt += config_data["prompts"]["goal_state_prompt"] + "\n"

        prompt += "\n"

        if args.examples != 0:
            example_data_path = config_data["examples"]["example_data"]

            with open(example_data_path, "r") as f:
                examples_data = json.load(f)

            prompt += f"<EXAMPLES>\n\n"
            num_examples = (
                len(examples_data.items())
                if args.examples == -1
                else min(args.examples, len(examples_data.items()))
            )
            for nameTag, example in list(examples_data.items())[:num_examples]:
                try:
                    ex_prompt = "<EXAMPLE>\n\n"

                    ex_prompt += construct_prompt_core(
                        config_data, example, 0, 0, 0, False, False, False
                    )

                    ex_prompt += f"\n<IMPROVED>\n{example['improved']}\n</IMPROVED>\n\n"
                    ex_prompt += f"</EXAMPLE>\n\n"
                    prompt += ex_prompt
                except:
                    pass
            prompt += f"</EXAMPLES>\n\n"

        prompt += construct_prompt_core(
            config_data,
            item,
            args.file_context,
            args.context,
            args.rag,
            args.annotation,
            args.informal,
            args.goal_state,
            True,
        )

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
    print("=" * 20)
    print(f"Processing {len(files_to_process)} files from {args.dataset_path} on {args.split} split.")
    print("-" * 20)
    for file in files_to_process:
        print(f"  - {file}")
    print("=" * 20)
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
                df.loc[len(df)] = [
                    module,
                    item["decl"],
                    item["decl_idx"],
                    item["raw_prompt"],
                ]

    # returns the path to the directory containing run metadata and the parquet lake
    output_path = run_inference(df, args)

    ray.shutdown()

    # we should also initialize + index the duckDB stuff


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("prompt_id", type=str, help="Prompt ID to use")
    parser.add_argument(
        "--run_id",
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
        "--informal", type=bool, default=False, help="Informal? (default: False)"
    )
    parser.add_argument(
        "--goal_state", type=bool, default=False, help="Goal state? (default: False)"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Number of context retrievals (default: 0, -1 for all)",
    )
    parser.add_argument(
        "--file_context",
        type=int,
        default=0,
        help="Number of file context items (default: 0, -1 for all)",
    )
    parser.add_argument(
        "--rag",
        type=int,
        default=0,
        help="Number of RAG retrievals (default: 0, max: 10)",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of few-shot example retrievals (default: 0, -1 for all)",
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
