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

        if item["id"]["kind"] != "theorem":
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
            "solution": item["id"]["content"],
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

    df = pd.DataFrame(columns=["module", "decl", "decl_idx", "raw_prompt", "solution"])
    print("=" * 20)
    print(
        f"Processing {len(files_to_process)} files from {args.dataset_path} on {args.split} split."
    )
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
                    item["solution"],
                ]

    # returns the path to the directory containing run metadata and the parquet lake
    # Output DataFrame as JSONL file
    jsonl_data = []
    for _, row in df.iterrows():
        jsonl_data.append({"instruction": row["raw_prompt"], "output": row["solution"]})

    with open(args.output_path, "w") as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(jsonl_data)} items to {args.output_path}")

    # we should also initialize + index the duckDB stuff


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("prompt_id", type=str, help="Prompt ID to use")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Run ID to use for evaluation (default: run_<timestamp>)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
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

    args = parser.parse_args()

    main(args)
