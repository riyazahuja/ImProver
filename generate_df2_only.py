#!/usr/bin/env python3
"""
Generate df2 using exact same logic as inference.py but without running inference.
Then map completed prompts from logfile to the row indices.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ImProver', 'basic'))

# Import from inference.py
from inference import construct_prompts
import pandas as pd
import json
import argparse
import re
from tqdm import tqdm
import duckdb


def generate_df2(args):
    """Generate df2 using same logic as inference.py main()"""

    # Load dataset
    with open(args.dataset_path, 'r') as f:
        all_data = json.load(f)
        dataset = all_data[args.split]

    # Get files to process
    files_to_process = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]

    prompt_root = os.path.join("prompts", args.prompt_id)
    metric_root = os.path.join("metrics", args.metric)
    config_path = os.path.join(metric_root, "config.json")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Build df (same as inference.py)
    df = pd.DataFrame(columns=["module", "decl", "decl_idx", "raw_prompt"])

    print(f"Processing {len(files_to_process)} files...")
    for file_info in tqdm(files_to_process):
        file = file_info if type(file_info) is str else file_info["file"]

        file_path = os.path.join(prompt_root, "src", file.replace(".lean", ".json"))
        module = file.replace(".lean", "").replace("/", ".")

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data_raw = json.load(f)
                prompt_data = construct_prompts(config_data, data_raw, args)

            for item in prompt_data:
                df.loc[len(df)] = [
                    module,
                    item["decl"],
                    item["decl_idx"],
                    item["raw_prompt"],
                ]

    print(f"Created df with {len(df)} rows")

    # Build df2 (same as inference_server.py)
    df2 = pd.concat([df.assign(prompt_idx=i) for i in range(args.n)], ignore_index=True)
    print(f"Created df2 with {len(df2)} rows (df: {len(df)} × n: {args.n})")

    return df2


def parse_completed_prompts(logfile_path):
    """Parse completed prompts from logfile."""
    print(f"Parsing logfile: {logfile_path}")

    with open(logfile_path, 'r') as f:
        content = f.read()

    blocks = re.split(r'#{80,}\n', content)
    completed = {}

    for block in tqdm(blocks, desc="Parsing"):
        prompt_match = re.search(r'Prompt (\d+) completed\.', block)
        if not prompt_match:
            continue

        row_idx = int(prompt_match.group(1))

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


def map_and_save(df2, completed_prompts, run_id):
    """Map completed prompts to df2 and save."""
    print("Mapping completed prompts to df2...")

    # Filter df2 to only completed rows
    completed_indices = sorted(completed_prompts.keys())
    df_completed = df2.iloc[completed_indices].copy()

    # Add df2_row_idx and answer columns
    df_completed['df2_row_idx'] = df_completed.index
    df_completed['answer'] = df_completed.index.map(completed_prompts)
    df_completed = df_completed.reset_index(drop=True)

    print(f"Matched {len(df_completed)} rows")
    print(f"Row index range: {min(completed_indices)} to {max(completed_indices)}")

    # Save to database
    run_dir = os.path.join("evals", run_id)
    os.makedirs(run_dir, exist_ok=True)

    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    parquet_path = os.path.join(data_dir, "part-00000.parquet")
    df_completed.to_parquet(parquet_path, index=False)

    db_path = os.path.join(run_dir, "data.duckdb")
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS run_data")
    con.execute(f"CREATE TABLE run_data AS SELECT * FROM read_parquet('{parquet_path}')")

    count = con.execute("SELECT COUNT(*) FROM run_data").fetchone()[0]
    print(f"\nCreated database with {count} rows at {db_path}")

    # Verify a few mappings
    print("\nSample mappings:")
    sample = con.execute("""
        SELECT df2_row_idx, module, decl, SUBSTRING(answer, 1, 60) as answer_preview
        FROM run_data
        ORDER BY df2_row_idx
        LIMIT 10
    """).fetchall()

    for row in sample:
        print(f"  [{row[0]}] {row[1]}.{row[2]}")
        print(f"       Answer: {row[3]}...")

    con.close()

    return db_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', help='Path to incomplete logfile')
    parser.add_argument('--run_id', required=True)

    # Parameters from train.sh line 62
    parser.add_argument('--dataset_path', default='data/final_dataset_decontaminated.json')
    parser.add_argument('--split', default='train')
    parser.add_argument('--metric', default='declarativity2')
    parser.add_argument('--prompt_id', default='final_train')
    parser.add_argument('--model', default='gpt-oss-120b')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--informal', action='store_true', default=False)
    parser.add_argument('--annotation', action='store_true', default=False)
    parser.add_argument('--context', type=int, default=5)
    parser.add_argument('--file_context', type=int, default=0)
    parser.add_argument('--rag', type=int, default=0)
    parser.add_argument('--examples', type=int, default=4)
    parser.add_argument('--goal_state', action='store_true', default=False)

    args = parser.parse_args()

    print("="*60)
    print("Generating df2 using inference.py logic...")
    print("="*60)

    # Step 1: Generate df2
    df2 = generate_df2(args)

    # Step 2: Parse completed prompts from logfile
    completed_prompts = parse_completed_prompts(args.logfile)

    # Step 3: Map and save
    db_path = map_and_save(df2, completed_prompts, args.run_id)

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Completed prompts: {len(completed_prompts)}")


if __name__ == '__main__':
    main()
