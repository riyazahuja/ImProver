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


def run_inference(df, args):
    # assuming gpus sit behind different PCIe host bridges on separate
    # NUMA sockets (i.e. nvidia-smi topo -m shows SYS between gpus)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    ray.init(
        num_cpus=args.cpus, num_gpus=args.gpus
    )  # , _temp_dir='/home/riyaza/ray_tmp')
    DataContext.get_current().wait_for_min_actors_s = 1800
    ctx = DataContext.get_current()
    # ctx.progress_bar = True
    # ctx.execution_options.verbose_progress = True

    assert Version(ray.__version__) >= Version(
        "2.44.1"
    ), "Ray version must be at least 2.44.1"

    ds = ray.data.from_pandas(df)
    # Create a new dataframe with duplicated rows, each with a unique prompt_idx
    df2_parts = []
    for i in range(args.n):
        df_copy = df.copy()
        df_copy["prompt_idx"] = i
        df2_parts.append(df_copy)

    df2 = pd.concat(df2_parts, ignore_index=True)
    # Use df2 instead of df for the Ray dataset
    ds = ray.data.from_pandas(df2).repartition(args.gpus * 4)
    # ds = ray.data.from_pandas(df).repartition(args.gpus * 4)
    # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
    print(ds.schema())

    size = ds.count()
    print(f"Size of dataset: {size} prompts")

    # ctx.execution_options = ExecutionOptions(task_extra_resources={"CPU": 0.25})

    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_resources={"CPU": args.cpus // args.gpus, "GPU": 1},
        concurrency=args.gpus,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "enable_chunked_prefill": True,
            "max_model_len": 8192,
            "max_num_batched_tokens": 65536,
            # "max_num_batched_tokens": 4096,
            # "max_model_len": 16384,
        },
        max_concurrent_batches=32,
        batch_size=32,
    )

    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["raw_prompt"]}],
            sampling_params=dict(
                # n=args.n,
                truncate_prompt_tokens=7168,
                # temperature=0.3,
                max_tokens=1024,
            ),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )
    ds = vllm_processor(ds).materialize()

    # id = f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = os.path.join(args.KG_dir, "filtered")
    os.makedirs(run_output_dir, exist_ok=True)

    output_path = os.path.join(run_output_dir, "data")

    config = {
        "dataset": args.dataset_path,
        "split": args.split,
        "n": args.n,
        "model": args.model,
    }

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


def get_thm_prompt(thm):
    THRESHOLD = 2
    # if we want to filter down the number of inference prompts, uncomment
    if (
        thm["isExtracted"] == True
        or thm["isOriginal"] == False
        or len(thm["C1_dependencies"]) + len(thm["C2_dependencies"]) <= THRESHOLD
    ):
        return None

    all_dependencies_data = thm["C1_dependencies"] + thm["C2_dependencies"]
    dependencies_raw = [t["text"].strip() for t in all_dependencies_data]

    dependencies = "\n\n".join(
        [
            f"<DEPENDENCY index={i}>\n{dep}\n</DEPENDENCY>"
            for i, dep in enumerate(dependencies_raw)
        ]
    )

    prompt = f"""

Given the following Lean4 theorem (wrapped with <CURRENT>...</CURRENT>), and all its dependencies/lemmas (each wrapped in <DEPENDENCY>, with an index) return the indices of the "core" lemmas. Namely, a core lemma is a dependency that intuitively embodies fundamental ideas relevant to the proof of the current theorem. This is in contrast to purely technical or helper lemmas.

Return your answer as a comma-seperated list of indices, wrapped in <CORE_DEPENDENCIES>...</CORE_DEPENDENCIES> tags. For example,




<CURRENT>

{thm['text'].strip()}

</CURRENT>

{dependencies}

"""

    return {"raw_prompt": prompt, **thm}


def main(args):

    prompts = []

    for root, _, files in os.walk(args.KG_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            module_path = os.path.relpath(os.path.join(root, file), args.KG_path)
            module = module_path.replace("/", ".").replace(".json", "")
            with open(os.path.join(root, file), "r") as f:
                theorems = json.load(f)
            # with open(os.path.join(root, file).replace("KG", "KG2.5"), "r") as f:
            #     C2Data = json.load(f)
            for thm in theorems:
                thm_prompt = get_thm_prompt(thm)
                if thm_prompt:
                    prompts.append(thm_prompt)

    # Convert the list of prompts to a pandas DataFrame
    df = pd.DataFrame(prompts)
    output_path = run_inference(df, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter KG for ImProver")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--KG_dir",
        type=str,
        default="KG2.75",
        help="Directory to get KG (default: KG2.75)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Model to use",
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
    parser.add_argument(
        "--n", type=int, default=1, help="Majority vote value (default: 1)"
    )
    # parser.add_argument(
    #     "--annotation", type=bool, default=False, help="Annotation? (default: False)"
    # )
    # parser.add_argument(
    #     "--context",
    #     type=int,
    #     default=0,
    #     help="Number of context retrievals (default: 0)",
    # )
    # parser.add_argument(
    #     "--rag", type=int, default=0, help="Number of RAG retrievals (default: 0)"
    # )
    # parser.add_argument(
    #     "--examples",
    #     type=int,
    #     default=0,
    #     help="Number of few-shot example retrievals (default: 0)",
    # )

    args = parser.parse_args()

    main(args)
