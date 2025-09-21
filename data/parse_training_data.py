import pandas as pd
import json
import os
from pathlib import Path
import re
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path
import tiktoken


def get_full_prompts(repos, full_prompts_name="Qwen-7B-BASE_all2"):

    # Create a list to store all DataFrames
    all_dfs = []

    # Read data from each repo
    for repo in repos:
        csv_path = f"improver_outputs_new/{repo}/{full_prompts_name}/improver_combined_results.csv"
        if os.path.exists(csv_path):
            df_repo = pd.read_csv(csv_path)
            all_dfs.append(df_repo)
        else:
            print(f"Warning: CSV file not found for repo {repo}: {csv_path}")

    # Combine all DataFrames into one
    if all_dfs:
        df_test = pd.concat(all_dfs, ignore_index=True)
    else:
        print(
            f"Error: No data found for any repositories with name {full_prompts_name}"
        )
        return {}

    filtered_df_test = df_test.copy()

    # Group by module and decl to handle each declaration separately
    filtered_df_test["row_num"] = filtered_df_test.groupby(
        ["module", "decl"]
    ).cumcount()

    # Keep only first n_prime rows for each decl
    filtered_df_test = filtered_df_test[filtered_df_test["row_num"] < 1]

    # For each decl, keep only the best row based on criteria
    def select_best_row(group):
        # If any row has new_correct = True, select from those
        correct_rows = group[group["new_correct"] == True]
        if len(correct_rows) > 0:
            # Among correct rows, return the one with minimal new_score
            return correct_rows.nsmallest(1, "new_score")
        # If no correct rows, return the first row
        return group.iloc[:1]

    # Apply the selection process to each group
    filtered_df_test = (
        filtered_df_test.groupby(["module", "decl"])
        .apply(select_best_row)
        .reset_index(drop=True)
    )

    filtered_merged_df = filtered_df_test

    prompt_all = filtered_merged_df["original_prompt"].values

    # prompt_all = [
    #     re.sub(r"Shorten the current theorem.*?tag\.", "", prompt, flags=re.DOTALL)
    #     for prompt in prompt_all
    # ]

    # Create prompt_rag by removing specified sections from prompt_all
    prompt_rag = []
    prompt_base = []
    prompt_norag = []

    for prompt in prompt_all:
        # Remove the annotation explanation text

        rag_instruction = "A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response. The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>. "
        norag_instruction = "The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>. "

        rag_prompt = prompt.replace(
            rag_instruction,
            "",
        )
        norag_prompt = prompt.replace(
            norag_instruction,
            "",
        )

        base_prompt = prompt.replace(rag_instruction, "").replace(norag_instruction, "")

        # rag_prompt = prompt
        # norag_prompt = prompt
        # Remove everything between <CONTEXT> and </CONTEXT>
        rag_prompt = re.sub(r"<CONTEXT>.*?</CONTEXT>", "", rag_prompt, flags=re.DOTALL)

        # Remove everything between <ANNOTATION> and </ANNOTATION>
        rag_prompt = re.sub(
            r"<ANNOTATION>.*?</ANNOTATION>", "", rag_prompt, flags=re.DOTALL
        )

        # Remove everything between <RETRIEVED> and </RETRIEVED>
        norag_prompt = re.sub(
            r"<RETRIEVED>.*?</RETRIEVED>", "", norag_prompt, flags=re.DOTALL
        )

        base_prompt = re.sub(
            r"<CONTEXT>.*?</CONTEXT>", "", base_prompt, flags=re.DOTALL
        )  # Remove context for base prompt as well
        base_prompt = re.sub(
            r"<ANNOTATION>.*?</ANNOTATION>", "", base_prompt, flags=re.DOTALL
        )  # Remove annotation for base prompt as well
        base_prompt = re.sub(
            r"<RETRIEVED>.*?</RETRIEVED>", "", base_prompt, flags=re.DOTALL
        )  # Remove retrieved for base prompt as well

        prompt_base.append(base_prompt)
        prompt_rag.append(rag_prompt)
        prompt_norag.append(norag_prompt)

    result_df = pd.DataFrame(
        {
            "base_prompt": prompt_base,
            "norag_prompt": prompt_norag,
            "rag_prompt": prompt_rag,
            "all_prompt": prompt_all,
            "decl": filtered_merged_df["decl"],
        }
    )
    # Save the result dataframe to a CSV file
    output_dir = "improver_outputs_new/parsed_data"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/{full_prompts_name}_prompts.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved prompts data to {csv_path}")

    # Convert dataframe to dictionary format
    prompts_dict = {}
    # i = 0
    for _, row in result_df.iterrows():

        decl = row["decl"]
        # i += 1
        # if i < 100:
        #     print(decl)
        prompts_dict[decl] = {
            "base_prompt": row["base_prompt"],
            "norag_prompt": row["norag_prompt"],
            "rag_prompt": row["rag_prompt"],
            "all_prompt": row["all_prompt"],
        }
    # check_key = "Bulgaria1998P1.lemma2"
    # if check_key in prompts_dict:
    #     print(f"Found {check_key} in prompts_dict")
    #     # print(prompts_dict[check_key])
    # else:
    #     print(
    #         f"Warning: {check_key} not found in prompts_dict. This may indicate an issue with the data processing."
    #     )

    return prompts_dict


def calc_stats(repos, df, ns):
    n = max(ns)
    ns = [n]

    # First run ImProver to get data
    # get_data(test_set, repo, n)

    # Convert results to CSV

    # Import extract functions
    sys.path.append("benchmark")
    full_prompts = get_full_prompts(repos)
    # Load and analyze the data
    # For each n in ns, take first n trajectories from the max n run
    metrics_by_n = {}
    print(f"Data for {model} on all repos:")
    for n_prime in ns:
        # Create a copy of the dataframe
        filtered_df = df.copy()

        # Group by module and decl to handle each declaration separately
        filtered_df["row_num"] = filtered_df.groupby(["module", "decl"]).cumcount()

        # Keep only first n_prime rows for each decl
        filtered_df = filtered_df[filtered_df["row_num"] < n_prime]

        # For each decl, keep only the best row based on criteria
        def select_best_row(group):
            # If any row has new_correct = True, select from those
            correct_rows = group[
                (group["new_correct"] == True) & (group["new_raw"] != "")
            ]
            if len(correct_rows) > 0:
                # Among correct rows, return the one with minimal new_score
                return correct_rows.nsmallest(1, "new_score")
            # If no correct rows, return the first row
            return group.iloc[:1]

        # Apply the selection process to each group
        filtered_df = (
            filtered_df.groupby(["module", "decl"])
            .apply(select_best_row)
            .reset_index(drop=True)
        )

        # Filter out rows where new_correct is False or delta is 0
        filtered_df = filtered_df[
            (filtered_df["new_correct"] == True) & (filtered_df["delta"] != 0)
        ]
        print(
            f"After filtering for correct solutions with non-zero delta: {len(filtered_df)} rows"
        )
        # Save the filtered DataFrame to a CSV file
        output_csv_file = f"improver_outputs_new/filtered_df_n{n_prime}.csv"
        filtered_df.to_csv(output_csv_file, index=False)
        print(f"Filtered DataFrame saved to {output_csv_file}")

        # Convert to Alpaca format
        alpaca_data = {}

        for _, row in filtered_df.iterrows():
            decl = row.get("decl", "")
            fp = full_prompts.get(decl, None)

            if fp is None:
                print(f"Warning: No full prompts found for decl {decl}. Skipping.")
                continue
            # Get the prompts

            new_raw = row.get("new_raw", "")

            # Skip if prompt or new_raw is missing or empty
            if not new_raw:
                continue

            for id in ["all", "rag", "norag", "base"]:
                prompt = fp.get(f"{id}_prompt", None)
                if not prompt:
                    print(f"Warning: Empty prompt found for decl {decl}. Skipping.")
                    continue

                # Split on the last instance of "<CURRENT>"
                splitter = "Include the output in the <IMPROVED>...</IMPROVED> tag."
                parts = prompt.rsplit(splitter, 1)
                if len(parts) != 2:
                    continue

                instruction = (parts[0] + splitter).strip()
                inp = parts[1].strip()
                outp = new_raw.strip()
                if id not in alpaca_data.keys():
                    alpaca_data[id] = []

                alpaca_data[id].append(
                    {"instruction": instruction, "input": inp, "output": outp}
                )

        # Save as JSONL
        for id, data in alpaca_data.items():
            if not data:
                print(f"No data found for id {id}. Skipping JSONL save.")
                continue
            output_jsonl_file = (
                f"improver_outputs_new/alpaca_data_n{n_prime}_{id}.jsonl"
            )
            with open(output_jsonl_file, "w") as f:
                for entry in data:
                    f.write(json.dumps(entry) + "\n")
            print(f"Alpaca format data saved to {output_jsonl_file}")

            # Calculate token statistics
            enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

            input_tokens = []
            output_tokens = []
            total_tokens = []

            for entry in data:
                # Calculate tokens for instruction + input
                input_text = entry["instruction"] + entry["input"]
                input_token_count = len(enc.encode(input_text))
                input_tokens.append(input_token_count)

                # Calculate tokens for output
                output_token_count = len(enc.encode(entry["output"]))
                output_tokens.append(output_token_count)

                # Calculate total tokens
                total_token_count = input_token_count + output_token_count
                total_tokens.append(total_token_count)

            # Calculate averages
            avg_input_tokens = (
                sum(input_tokens) / len(input_tokens) if input_tokens else 0
            )
            avg_output_tokens = (
                sum(output_tokens) / len(output_tokens) if output_tokens else 0
            )
            avg_total_tokens = (
                sum(total_tokens) / len(total_tokens) if total_tokens else 0
            )

            print(f"\nToken statistics for {id}:")
            print(f"  Average instruction + input tokens: {avg_input_tokens:.1f}")
            print(f"  Average output tokens: {avg_output_tokens:.1f}")
            print(f"  Average total tokens: {avg_total_tokens:.1f}")
            print(f"  Number of examples: {len(data)}")
            print(f"  Total tokens: {sum(total_tokens)}")
            print(f"  Max input tokens: {max(input_tokens) if input_tokens else 0}")
            print(f"  Max output tokens: {max(output_tokens) if output_tokens else 0}")
            print(
                f"  80th percentile input tokens: {pd.Series(input_tokens).quantile([0.5,0.75,0.9,0.99])}"
            )
            print(
                f"  Number >4096: {sum(
                [1 for x in total_tokens if x > 4096]
            )} (out of {len(data)})"
            )


def aggregate(model, ns, repos):
    # Create a list to store all DataFrames
    all_dfs = []

    # Iterate over each repo and read the CSV files
    for repo in repos:
        csv_file = f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # Add a new column to identify the repo
            df["repo"] = repo
            all_dfs.append(df)

    # Concatenate all DataFrames into one

    combined_df = pd.concat(all_dfs, ignore_index=True)

    calc_stats(repos, combined_df, ns)


if __name__ == "__main__":
    repos = ["MIL", "Compfiles"]
    # ns = [1] + list(range(5, 61, 5))
    ns = [32]  # + list(range(2, 33, 2))
    # ns = [64]
    if len(sys.argv) < 2:
        print("Usage: python eval.py <model1> <model2> ...")
        sys.exit(1)
    models = sys.argv[1:]
    for model in models:
        aggregate(model, ns, repos)
