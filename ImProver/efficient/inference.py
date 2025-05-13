import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
import os
import torch
import pandas as pd
import json
import datetime


def run_inference(df, args):
    # assuming gpus sit behind different PCIe host bridges on separate
    # NUMA sockets (i.e. nvidia-smi topo -m shows SYS between gpus)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    ray.init(num_cpus=args.cpus, num_gpus=args.gpus)
    DataContext.get_current().wait_for_min_actors_s = 1800

    assert Version(ray.__version__) >= Version(
        "2.44.1"
    ), "Ray version must be at least 2.44.1"

    ds = ray.data.from_pandas(df)
    # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
    print(ds.schema())

    size = ds.count()
    print(f"Size of dataset: {size} prompts")

    # ctx.execution_options = ExecutionOptions(task_extra_resources={"CPU": 0.25})

    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_resources={"CPU": args.cpus, "GPU": 1},
        concurrency=args.gpus,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 16384,
        },
        max_concurrent_batches=16,
        batch_size=128,
    )

    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["prompt"]}],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=250,
            ),
        ),
        postprocess=lambda row: dict(answer=row["model_output"], **row),
    )

    ds = vllm_processor(ds).materialize()

    id = f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = os.path.join(args.output_dir, id)
    os.makedirs(run_output_dir, exist_ok=True)

    output_path = os.path.join(run_output_dir, "data")

    config = {
        "dataset": args.dataset_path,
        "split": args.split,
        "metric": args.metric,
        "n": args.n,
        "annotation": args.annotation,
        "context": args.context,
        "rag": args.rag,
        "model": args.model,
    }

    config_path = os.path.join(run_output_dir, "config.json")

    run_output_dir = os.path.join(args.output_dir, id)
    ds.write_csv(f"local://{output_path}")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return run_output_dir
    # print(f"Output written to {output_path}")


def construct_prompts(data, args):
    idx = 0
    items = []
    for name, decl_data in data.items():
        prompt = decl_data["system"] + "\n\n"
        prompt += decl_data["example_prompt"] + "\n"

        if args.annotation:
            prompt += decl_data["annotation_prompt"] + "\n"

        if args.context:
            prompt += decl_data["context_prompt"] + "\n"

        if args.rag:
            prompt += decl_data["rag_prompt"] + "\n"

        prompt += "\n"

        if args.annotation:
            prompt += f"<EXAMPLES>\n{decl_data["annotation"]}\n</EXAMPLES>\n\n"

        if args.context:
            prompt += f"<CONTEXT>\n"
            for context in decl_data["context"]:
                prompt += f"<ITEM>\n--name={context['name']}\n--type={context['context_item_type']}\n{context['content']}\n</ITEM>\n"
            prompt += f"</CONTEXT>\n\n"

        if args.rag:
            prompt += f"<RAG>\n"
            for rag in decl_data["rag"]:
                prompt += f"<DOC>\n--src={rag['src']}\n{rag['content']}\n</DOC>\n"
            prompt += f"</RAG>\n\n"

        prompt += f"\n<CURRENT>\n{decl_data['current']}\n</CURRENT>\n\n"
        prompt += "<IMPROVED>"

        data = {
            "decl": name,
            "decl_idx": idx,
            "prompt": prompt,
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

    prompt_root = os.path.join(args.prompts_dir, args.metric)

    df = pd.DataFrame(columns=["file_path", "decl", "decl_idx", "prompt"])

    for file in files_to_process:
        file_path = os.path.join(prompt_root, file.replace(".lean", ".json"))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data_raw = json.load(f)
                prompt_data = construct_prompts(data_raw, args)

            for item in prompt_data:
                df = df.append(
                    {
                        "file_path": file_path,
                        "decl": item["decl"],
                        "decl_idx": item["decl_idx"],
                        "prompt": item["prompt"],
                    },
                    ignore_index=True,
                )

    output_path = run_inference(df, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
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
        "--prompts_dir",
        type=str,
        default="prompts/",
        help="Directory to output runs (default: prompts/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/",
        help="Directory to output runs (must be absolute) (default: runs/)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpu_count(),
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
        "--context", type=bool, default=False, help="Context? (default: False)"
    )
    parser.add_argument(
        "--rag", type=int, default=False, help="Number of RAG retrievals (default: 0)"
    )

    args = parser.parse_args()

    main(args)
