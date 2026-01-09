#!/usr/bin/env python3
"""
Properly reconstruct inference data by recreating df2 from inference.py
and matching the row indices from the logfile.
"""

import re
import json
import pandas as pd
import duckdb
import os
from tqdm import tqdm
import argparse


def parse_completed_prompts(logfile_path):
    """Parse logfile to extract row indices and answers."""
    print(f"Parsing logfile...")

    with open(logfile_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    blocks = re.split(r'#{80,}\n', content)
    completed = {}

    for block in tqdm(blocks, desc="Parsing"):
        # Find "Prompt N completed."
        prompt_match = re.search(r'Prompt (\d+) completed\.', block)
        if not prompt_match:
            continue

        row_idx = int(prompt_match.group(1))

        # Extract answer
        answer_match = re.search(r'Answer:\s*<IMPROVED>(.*?)</IMPROVED>', block, re.DOTALL)
        if answer_match:
            completed[row_idx] = answer_match.group(1).strip()
        else:
            response_match = re.search(r"'content':\s*'<IMPROVED>(.*?)</IMPROVED>'", block, re.DOTALL)
            if response_match:
                improved = response_match.group(1).strip()
                improved = improved.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
                completed[row_idx] = improved

    print(f"Found {len(completed)} completed prompts")
    return completed


def construct_prompts(config_data, data, context, file_context, rag, annotation, informal, goal_state, examples):
    """Reconstruct prompts using same logic as inference.py"""
    items = []
    idx = 0

    for item in data:
        if item["id"]["isExtracted"] or len(item["id"]["errorMsgs"]) != 0:
            continue
        if item["id"]["kind"] != "theorem":
            continue

        name = item["id"]["name"]

        # Build prompt (simplified - just need the structure)
        prompt = config_data["prompts"]["system_prompt"] + "\n"
        if examples != 0:
            prompt += config_data["prompts"].get("example_prompt", "") + "\n"
        if context != 0:
            prompt += config_data["prompts"].get("context_prompt", "") + "\n"
        if file_context != 0:
            prompt += config_data["prompts"].get("file_context_prompt", "") + "\n"
        if rag != 0:
            prompt += config_data["prompts"].get("rag_prompt", "") + "\n"
        if annotation:
            prompt += config_data["prompts"].get("annotation_prompt", "") + "\n"
        if informal:
            prompt += config_data["prompts"].get("informal_prompt", "") + "\n"
        if goal_state:
            prompt += config_data["prompts"].get("goal_state_prompt", "") + "\n"

        # Add context, examples, etc. (simplified)
        prompt += f"<CURRENT>\n{item.get('content_sorry', item['id']['content'])}\n</CURRENT>\n"

        data_item = {
            "decl": name,
            "decl_idx": idx,
            "raw_prompt": prompt,
        }
        items.append(data_item)
        idx += 1

    return items


def recreate_df(dataset_path, split, metric, prompt_id, context, file_context, rag, annotation, informal, goal_state, examples):
    """Recreate df using same logic as inference.py main()"""
    print(f"Recreating dataframe from dataset...")

    with open(dataset_path, 'r') as f:
        all_data = json.load(f)
        dataset = all_data[split]

    files_to_process = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]

    prompt_root = os.path.join("prompts", prompt_id)
    metric_root = os.path.join("metrics", metric)
    config_path = os.path.join(metric_root, "config.json")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    df = pd.DataFrame(columns=["module", "decl", "decl_idx", "raw_prompt"])

    for file_info in tqdm(files_to_process, desc="Processing files"):
        file = file_info if type(file_info) is str else file_info["file"]

        file_path = os.path.join(prompt_root, "src", file.replace(".lean", ".json"))
        module = file.replace(".lean", "").replace("/", ".")

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data_raw = json.load(f)
                prompt_data = construct_prompts(config_data, data_raw, context, file_context,
                                               rag, annotation, informal, goal_state, examples)

            for item in prompt_data:
                df.loc[len(df)] = [
                    module,
                    item["decl"],
                    item["decl_idx"],
                    item["raw_prompt"],
                ]

    print(f"Created df with {len(df)} rows")
    return df


def reconstruct_data(df, n, completed_prompts):
    """Recreate df2 and filter to completed prompts."""
    print(f"Recreating df2 with n={n}...")

    # Recreate df2 using same logic as inference.py
    df2_parts = []
    for i in range(n):
        df_copy = df.copy()
        df_copy["prompt_idx"] = i
        df2_parts.append(df_copy)

    df2 = pd.concat(df2_parts, ignore_index=True)
    print(f"df2 has {len(df2)} rows (df: {len(df)} × n: {n})")

    # Filter to only completed rows and preserve the row index
    completed_indices = sorted(completed_prompts.keys())
    df_completed = df2.iloc[completed_indices].copy()

    # IMPORTANT: Save the df2 row index as a column before adding answers
    # This preserves the mapping when saved to parquet
    df_completed['df2_row_idx'] = df_completed.index

    # Add answers using the preserved index
    df_completed['answer'] = df_completed.index.map(completed_prompts)

    # Reset index so it's sequential for parquet, but keep df2_row_idx column
    df_completed = df_completed.reset_index(drop=True)

    print(f"Filtered to {len(df_completed)} completed rows")
    print(f"Row index range: {min(completed_indices)} to {max(completed_indices)}")

    return df_completed


def save_to_db(df, run_id):
    """Save dataframe to parquet and duckdb."""
    run_dir = os.path.join("evals", run_id)
    os.makedirs(run_dir, exist_ok=True)

    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save parquet
    parquet_path = os.path.join(data_dir, "part-00000.parquet")
    print(f"Writing parquet to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)

    # Create duckdb
    db_path = os.path.join(run_dir, "data.duckdb")
    print(f"Creating DuckDB at {db_path}...")

    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS run_data")
    con.execute(f"CREATE TABLE run_data AS SELECT * FROM read_parquet('{parquet_path}')")

    count = con.execute("SELECT COUNT(*) FROM run_data").fetchone()[0]
    print(f"Created run_data table with {count} rows")

    # Show sample
    sample = con.execute("SELECT module, decl, decl_idx FROM run_data LIMIT 10").fetchall()
    print("\nSample records:")
    for row in sample:
        print(f"  [{row[2]}] {row[0]}.{row[1]}")

    con.close()

    return db_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', help='Path to incomplete logfile')
    parser.add_argument('--run_id', required=True, help='Output run ID')
    parser.add_argument('--dataset_path', default='data/final_dataset_decontaminated.json')
    parser.add_argument('--split', default='train')
    parser.add_argument('--metric', default='declarativity2')
    parser.add_argument('--prompt_id', default='final_train')
    parser.add_argument('--n', type=int, default=1, help='n value (samples per prompt)')
    parser.add_argument('--context', type=int, default=5)
    parser.add_argument('--file_context', type=int, default=0)
    parser.add_argument('--rag', type=int, default=0)
    parser.add_argument('--annotation', action='store_true')
    parser.add_argument('--informal', action='store_true')
    parser.add_argument('--goal_state', action='store_true')
    parser.add_argument('--examples', type=int, default=4)

    args = parser.parse_args()

    # Step 1: Parse completed prompts from logfile
    completed_prompts = parse_completed_prompts(args.logfile)

    # Step 2: Recreate df using same logic as inference.py
    df = recreate_df(args.dataset_path, args.split, args.metric, args.prompt_id,
                     args.context, args.file_context, args.rag, args.annotation,
                     args.informal, args.goal_state, args.examples)

    # Step 3: Recreate df2 and filter to completed
    df_completed = reconstruct_data(df, args.n, completed_prompts)

    # Step 4: Save to database
    db_path = save_to_db(df_completed, args.run_id)

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Completed prompts: {len(completed_prompts)}")

    return 0


if __name__ == '__main__':
    exit(main())
