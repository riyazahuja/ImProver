import json
import tiktoken
from pathlib import Path
import numpy as np
from typing import List, Dict, Union


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))


def analyze_jsonl_tokens(file_path: str) -> Dict:
    """Analyze token statistics from a JSONL file in either Alpaca or Zephyr.nectar format."""
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    instruction_tokens: List[int] = []
    input_tokens: List[int] = []
    output_tokens: List[int] = []
    total_tokens: List[int] = []

    # Read and process the JSONL file
    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)

            # Determine the format (Alpaca or Zephyr.nectar)
            if "instruction" in entry:  # Alpaca format
                # Count tokens for each field
                instr_count = count_tokens(entry["instruction"], tokenizer)
                input_count = count_tokens(entry.get("input", ""), tokenizer)
                output_count = count_tokens(entry["output"], tokenizer)

            elif "prompt" in entry and "answers" in entry:  # Zephyr.nectar format
                # In Zephyr format, prompt is equivalent to instruction+input
                instr_count = count_tokens(entry["prompt"], tokenizer)
                input_count = 0  # Input is already included in prompt

                # Take the first answer as the output (or you could process all answers)
                if entry["answers"] and len(entry["answers"]) > 0:
                    output_count = count_tokens(
                        entry["answers"][0]["answer"], tokenizer
                    )
                else:
                    output_count = 0
            else:
                # Skip entries that don't match expected formats
                continue

            total = instr_count + input_count + output_count

            instruction_tokens.append(instr_count)
            input_tokens.append(input_count)
            output_tokens.append(output_count)
            total_tokens.append(total)

    # Calculate threshold statistics
    thresholds = [256, 512, 1024, 2048, 4096]
    output_threshold_counts = {}
    total_threshold_counts = {}

    for threshold in thresholds:
        if threshold <= 2048:
            output_threshold_counts[threshold] = sum(
                1 for count in output_tokens if count < threshold
            )
        if threshold >= 2048:
            total_threshold_counts[threshold] = sum(
                1 for count in total_tokens if count < threshold
            )

    stats = {
        "num_examples": len(total_tokens),
        "instruction_tokens": {
            "mean": np.mean(instruction_tokens),
            "std": np.std(instruction_tokens),
            "min": np.min(instruction_tokens),
            "max": np.max(instruction_tokens),
        },
        "input_tokens": {
            "mean": np.mean(input_tokens),
            "std": np.std(input_tokens),
            "min": np.min(input_tokens),
            "max": np.max(input_tokens),
        },
        "output_tokens": {
            "mean": np.mean(output_tokens),
            "std": np.std(output_tokens),
            "min": np.min(output_tokens),
            "max": np.max(output_tokens),
        },
        "total_tokens": {
            "mean": np.mean(total_tokens),
            "std": np.std(total_tokens),
            "min": np.min(total_tokens),
            "max": np.max(total_tokens),
            "sum": np.sum(total_tokens),
        },
        "thresholds": {
            "output": {
                threshold: count / len(output_tokens) * 100
                for threshold, count in output_threshold_counts.items()
            },
            "total": {
                threshold: count / len(total_tokens) * 100
                for threshold, count in total_threshold_counts.items()
            },
        },
    }

    return stats


if __name__ == "__main__":
    # Replace with your JSONL file path
    jsonl_paths = [
        "length_human_train.jsonl",
        "improver_basic.jsonl",
        "improver_dpo.jsonl",
    ]
    for jsonl_path in jsonl_paths:
        print(f"======================================\nAnalyzing {jsonl_path}...\n")
        stats = analyze_jsonl_tokens(jsonl_path)

        print("Dataset Statistics:")
        print(f"Number of examples: {stats['num_examples']}")
        print("\nToken Statistics:")
        for field in [
            "instruction_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ]:
            print(f"\n{field.replace('_', ' ').title()}:")
            for metric, value in stats[field].items():
                print(f"  {metric}: {value:.2f}")

        print("\nThreshold Statistics:")
        print("Output Tokens < Threshold (for thresholds ≤ 2048):")
        for threshold, percentage in stats["thresholds"]["output"].items():
            print(f"  < {threshold}: {percentage:.2f}%")

        print("Total Tokens < Threshold (for thresholds ≥ 2048):")
        for threshold, percentage in stats["thresholds"]["total"].items():
            print(f"  < {threshold}: {percentage:.2f}%")
