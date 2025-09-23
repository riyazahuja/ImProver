import os
import torch
import pandas as pd
import json
import datetime
import multiprocessing
import argparse
import duckdb
from transformers import AutoTokenizer


def get_stats(df: pd.DataFrame, args):
    # print(df.head())
    # print(df.describe())
    # print(df.info())
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize all prompts efficiently
    print("Tokenizing prompts...")
    prompts = df["raw_prompt"].tolist()

    # Tokenize in batches for efficiency
    batch_size = 32
    token_lengths = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        # Tokenize batch without padding to get true lengths
        encoded = tokenizer(
            batch, add_special_tokens=True, truncation=False, padding=False
        )
        batch_lengths = [len(tokens) for tokens in encoded["input_ids"]]
        token_lengths.extend(batch_lengths)

    # Add token lengths to dataframe
    df["token_length"] = token_lengths

    # Calculate detailed statistics
    print("\n=== PROMPT LENGTH STATISTICS ===")
    print(f"Total prompts: {len(df)}")
    print(f"Total tokens: {sum(token_lengths):,}")
    print(f"Mean tokens per prompt: {df['token_length'].mean():.2f}")
    print(f"Median tokens per prompt: {df['token_length'].median():.2f}")
    print(f"Standard deviation: {df['token_length'].std():.2f}")
    print(f"Min tokens: {df['token_length'].min()}")
    print(f"Max tokens: {df['token_length'].max()}")

    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        value = df["token_length"].quantile(p / 100)
        print(f"  {p:2d}th percentile: {value:.0f} tokens")

    # Histogram bins
    print("\nDistribution by ranges:")
    bins = [0, 1000, 2000, 4000, 8000, 16000, 32000, float("inf")]
    bin_labels = ["<1K", "1K-2K", "2K-4K", "4K-8K", "8K-16K", "16K-32K", ">32K"]
    df["length_bin"] = pd.cut(
        df["token_length"], bins=bins, labels=bin_labels, right=False
    )
    bin_counts = df["length_bin"].value_counts().sort_index()
    for bin_label, count in bin_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {bin_label}: {count} prompts ({percentage:.1f}%)")

    # Module-level statistics
    print("\nPer-module statistics:")
    module_stats = (
        df.groupby("module")["token_length"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(2)
    )
    print(module_stats)

    # Find extremes
    print("\nExtreme cases:")
    shortest_idx = df["token_length"].idxmin()
    longest_idx = df["token_length"].idxmax()
    print(
        f"Shortest prompt: {df.loc[shortest_idx, 'decl']} ({df.loc[shortest_idx, 'token_length']} tokens)"
    )
    print(
        f"Longest prompt: {df.loc[longest_idx, 'decl']} ({df.loc[longest_idx, 'token_length']} tokens)"
    )

    # Save statistics to output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"stats_{args.prompt_id}_{args.metric}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed dataframe
    df.to_parquet(os.path.join(output_dir, "prompt_stats.parquet"))

    # Save summary statistics
    stats_summary = {
        "total_prompts": len(df),
        "total_tokens": int(sum(token_lengths)),
        "mean_tokens": float(df["token_length"].mean()),
        "median_tokens": float(df["token_length"].median()),
        "std_tokens": float(df["token_length"].std()),
        "min_tokens": int(df["token_length"].min()),
        "max_tokens": int(df["token_length"].max()),
        "percentiles": {
            f"p{p}": float(df["token_length"].quantile(p / 100)) for p in percentiles
        },
        "distribution": bin_counts.to_dict(),
        # "module_stats": module_stats.to_dict(),
        "extremes": {
            "shortest": {
                "decl": df.loc[shortest_idx, "decl"],
                "tokens": int(df.loc[shortest_idx, "token_length"]),
            },
            "longest": {
                "decl": df.loc[longest_idx, "decl"],
                "tokens": int(df.loc[longest_idx, "token_length"]),
            },
        },
    }

    with open(os.path.join(output_dir, "stats_summary.json"), "w") as f:
        json.dump(stats_summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return output_dir


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

    for file_info in files_to_process:
        file = file_info if type(file_info) is str else file_info["file"]

        file_path = os.path.join(prompt_root, "src", file.replace(".lean", ".json"))
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
    output_path = get_stats(df, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("prompt_id", type=str, help="Prompt ID to use")
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

    args = parser.parse_args()

    main(args)
