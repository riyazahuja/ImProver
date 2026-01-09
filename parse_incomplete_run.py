#!/usr/bin/env python3
"""
Parse incomplete inference logfile and reconstruct data.duckdb
for continuing the pipeline from eval stage.
"""

import re
import json
import pandas as pd
import duckdb
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def extract_improved_content(response_content):
    """Extract content between <IMPROVED> tags."""
    match = re.search(r'<IMPROVED>(.*?)</IMPROVED>', response_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_content.strip()

def parse_logfile(logfile_path):
    """Parse the logfile and extract completed prompts with their answers."""

    print(f"Reading logfile: {logfile_path}")
    with open(logfile_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Split into response blocks using the separator
    blocks = re.split(r'#{80,}\n', content)

    completed_prompts = []

    for block in tqdm(blocks, desc="Parsing blocks"):
        # Look for "Prompt X completed." pattern
        prompt_match = re.search(r'Prompt (\d+) completed\.', block)
        if not prompt_match:
            continue

        prompt_num = int(prompt_match.group(1))

        # Look for Answer: section with <IMPROVED> content
        answer_match = re.search(r'Answer:\s*<IMPROVED>(.*?)</IMPROVED>', block, re.DOTALL)

        if answer_match:
            improved = answer_match.group(1).strip()
            completed_prompts.append({
                'prompt_num': prompt_num,
                'answer': improved,
            })
        else:
            # Try alternate pattern - sometimes content is directly in the response body
            response_match = re.search(r"'content':\s*'<IMPROVED>(.*?)</IMPROVED>'", block, re.DOTALL)
            if response_match:
                improved = response_match.group(1).strip()
                # Unescape common escape sequences
                improved = improved.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
                completed_prompts.append({
                    'prompt_num': prompt_num,
                    'answer': improved,
                })

    print(f"Found {len(completed_prompts)} completed prompts")
    return completed_prompts

def reconstruct_dataframe(completed_prompts, original_data_path=None, config_path=None):
    """
    Reconstruct the dataframe by matching prompt numbers with original data.
    If original data is not available, create a minimal dataframe.
    """

    # Try to load original parquet if available
    if original_data_path and os.path.exists(original_data_path):
        print(f"Loading original data from: {original_data_path}")
        df_orig = pd.read_parquet(original_data_path)

        # Create a mapping of decl_idx to completed answers
        prompt_map = {p['prompt_num']: p['answer'] for p in completed_prompts}

        # Filter to only include completed prompts
        df_filtered = df_orig[df_orig['decl_idx'].isin(prompt_map.keys())].copy()

        # Add the answer column
        df_filtered['answer'] = df_filtered['decl_idx'].map(prompt_map)

        print(f"Matched {len(df_filtered)} rows from original data")
        return df_filtered

    else:
        # Create minimal dataframe from completed prompts only
        print("Creating minimal dataframe from completed prompts")
        records = []
        for p in completed_prompts:
            records.append({
                'decl_idx': p['prompt_num'],
                'prompt_idx': 0,  # Assuming single sample per prompt
                'decl': f"unknown_decl_{p['prompt_num']}",
                'module': "unknown",
                'raw_prompt': "",  # Not available from logfile
                'answer': p['answer']
            })

        return pd.DataFrame(records)

def create_duckdb(df, output_path, run_id):
    """Create DuckDB database from dataframe."""

    run_dir = os.path.join("evals", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save as parquet first
    data_dir = os.path.join(run_dir, "data_temp")
    os.makedirs(data_dir, exist_ok=True)

    parquet_path = os.path.join(data_dir, "part-00000.parquet")
    print(f"Writing parquet to: {parquet_path}")
    df.to_parquet(parquet_path, index=False)

    # Create DuckDB
    db_path = os.path.join(run_dir, "data_temp.duckdb")
    print(f"Creating DuckDB at: {db_path}")

    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS run_data")
    con.execute(f"CREATE TABLE run_data AS SELECT * FROM read_parquet('{parquet_path}')")

    # Verify
    count = con.execute("SELECT COUNT(*) FROM run_data").fetchone()[0]
    print(f"Created run_data table with {count} rows")

    con.close()

    return db_path, run_dir

def main():
    parser = argparse.ArgumentParser(description='Parse incomplete inference logfile')
    parser.add_argument('logfile', type=str, help='Path to the logfile')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID for output')
    parser.add_argument('--original_parquet', type=str, default=None,
                       help='Path to original parquet file (if available)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.json (if available)')

    args = parser.parse_args()

    # Parse logfile
    completed_prompts = parse_logfile(args.logfile)

    if not completed_prompts:
        print("ERROR: No completed prompts found in logfile!")
        return 1

    # Reconstruct dataframe
    df = reconstruct_dataframe(completed_prompts, args.original_parquet, args.config)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Create DuckDB
    db_path, run_dir = create_duckdb(df, args.run_id, args.run_id)

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"Created temporary database at: {db_path}")
    print(f"Run directory: {run_dir}")
    print(f"Total completed prompts: {len(completed_prompts)}")
    print(f"\nNext steps:")
    print(f"1. Copy config.json to {run_dir}/config.json if not already there")
    print(f"2. Run eval stage: python ImProver/basic/eval_improver.py {args.run_id}")
    print(f"3. Run analysis stage (if applicable)")

    return 0

if __name__ == '__main__':
    exit(main())
