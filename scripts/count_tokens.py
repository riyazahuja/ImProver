import os
import pandas as pd
import argparse
import seaborn as sns
from transformers import AutoTokenizer

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate token counts for prompts")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model"
    )
    parser.add_argument(
        "--repos", nargs="+", required=True, help="List of repositories"
    )
    return parser.parse_args()


def load_and_combine_dataframes(model_name, repos):
    all_dfs = []

    for repo in repos:
        csv_path = (
            f"improver_outputs_new/{repo}/{model_name}/improver_combined_results.csv"
        )
        try:
            df = pd.read_csv(csv_path)
            if "original_prompt" in df.columns:
                all_dfs.append(df[["original_prompt"]])
                print(f"Loaded {len(df)} rows from {csv_path}")
            else:
                print(f"Warning: 'original_prompt' column not found in {csv_path}")
        except FileNotFoundError:
            print(f"Warning: File not found: {csv_path}")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

    if not all_dfs:
        raise ValueError("No valid dataframes found")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    unique_prompts = combined_df.drop_duplicates(
        subset=["original_prompt"]
    ).reset_index(drop=True)

    print(f"Combined {len(combined_df)} total prompts")
    print(f"Found {len(unique_prompts)} unique prompts after deduplication")

    return unique_prompts


def calculate_token_counts(prompts_df, tokenizer):
    token_counts = []

    for prompt in prompts_df["original_prompt"]:
        tokens = tokenizer.encode(prompt)
        token_counts.append(len(tokens))

    prompts_df["token_count"] = token_counts
    return prompts_df


def plot_distribution(prompts_df, model_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(prompts_df["token_count"], bins=30, kde=True)
    plt.title("Distribution of Token Counts")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    avg_tokens = prompts_df["token_count"].mean()
    plt.axvline(
        x=avg_tokens,
        color="r",
        linestyle="--",
        label=f"Average: {avg_tokens:.2f} tokens",
    )
    plt.legend()

    output_path = f"token_distribution_{model_name}.png"
    plt.savefig(output_path)
    print(f"Distribution plot saved to {output_path}")

    print("\nToken Count Statistics:")
    print(f"Average token count: {avg_tokens:.2f}")
    print(f"Median token count: {prompts_df['token_count'].median():.2f}")
    print(f"Min token count: {prompts_df['token_count'].min()}")
    print(f"Max token count: {prompts_df['token_count'].max()}")

    return avg_tokens


def main():
    args = parse_args()

    # Load and combine dataframes
    unique_prompts_df = load_and_combine_dataframes(args.model_name, args.repos)

    # Load tokenizer
    model_path = (
        "/data/user_data/riyaza/saved_models/DeepSeek-R1-Distill-Qwen-7B-improverSFT"
    )
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Calculate token counts
    print("Calculating token counts...")
    prompts_with_tokens = calculate_token_counts(unique_prompts_df, tokenizer)

    # Plot distribution and get average
    avg_tokens = plot_distribution(prompts_with_tokens, args.model_name)

    # Save token count data
    output_csv = f"token_counts_{args.model_name}.csv"
    prompts_with_tokens.to_csv(output_csv, index=False)
    print(f"Token count data saved to {output_csv}")


if __name__ == "__main__":
    main()
