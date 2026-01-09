#!/usr/bin/env python3
"""
Reconstruct data using an existing run as template for proper module/decl mapping.
"""

import re
import json
import pandas as pd
import duckdb
import os
from tqdm import tqdm
import argparse


def parse_completed_prompts(logfile_path):
    """Parse logfile to extract completed prompt indices and answers."""
    print(f"Parsing logfile...")

    with open(logfile_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    blocks = re.split(r'#{80,}\n', content)
    completed = {}

    for block in tqdm(blocks, desc="Parsing"):
        prompt_match = re.search(r'Prompt (\d+) completed\.', block)
        if not prompt_match:
            continue

        prompt_num = int(prompt_match.group(1))

        # Extract answer
        answer_match = re.search(r'Answer:\s*<IMPROVED>(.*?)</IMPROVED>', block, re.DOTALL)
        if answer_match:
            completed[prompt_num] = answer_match.group(1).strip()
        else:
            response_match = re.search(r"'content':\s*'<IMPROVED>(.*?)</IMPROVED>'", block, re.DOTALL)
            if response_match:
                improved = response_match.group(1).strip()
                improved = improved.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
                completed[prompt_num] = improved

    print(f"Found {len(completed)} completed prompts")
    return completed


def reconstruct_from_template(template_db_path, completed_prompts, run_id):
    """Use template database to get proper module/decl mapping."""
    print(f"Loading template from {template_db_path}...")

    con = duckdb.connect(template_db_path, read_only=True)

    # Get unique prompt templates (one per decl_idx)
    template_df = con.execute("""
        SELECT DISTINCT ON (decl_idx)
            module, decl, decl_idx, raw_prompt
        FROM run_data
        ORDER BY decl_idx, prompt_idx
    """).df()

    con.close()

    print(f"Template has {len(template_df)} unique prompts")

    # Build reconstructed dataframe
    records = []
    matched = 0

    for idx, row in tqdm(template_df.iterrows(), total=len(template_df), desc="Matching"):
        decl_idx = row['decl_idx']

        if decl_idx in completed_prompts:
            records.append({
                'module': row['module'],
                'decl': row['decl'],
                'decl_idx': decl_idx,
                'prompt_idx': 0,
                'raw_prompt': row['raw_prompt'] if 'raw_prompt' in row else "",
                'answer': completed_prompts[decl_idx]
            })
            matched += 1

    print(f"Matched {matched} out of {len(completed_prompts)} completed prompts")

    df = pd.DataFrame(records)

    # Create output database
    run_dir = os.path.join("evals", run_id)
    os.makedirs(run_dir, exist_ok=True)

    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    parquet_path = os.path.join(data_dir, "part-00000.parquet")
    df.to_parquet(parquet_path, index=False)

    db_path = os.path.join(run_dir, "data.duckdb")
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS run_data")
    con.execute(f"CREATE TABLE run_data AS SELECT * FROM read_parquet('{parquet_path}')")

    count = con.execute("SELECT COUNT(*) FROM run_data").fetchone()[0]
    print(f"\nCreated database with {count} rows")

    # Show sample
    sample = con.execute("SELECT module, decl, decl_idx FROM run_data LIMIT 10").fetchall()
    print("\nSample records:")
    for row in sample:
        print(f"  [{row[2]}] {row[0]}.{row[1]}")

    con.close()

    return db_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', help='Path to logfile')
    parser.add_argument('--template_db', required=True,
                       help='Path to template data.duckdb from similar run')
    parser.add_argument('--run_id', required=True, help='Output run ID')

    args = parser.parse_args()

    # Parse completed prompts
    completed = parse_completed_prompts(args.logfile)

    # Reconstruct using template
    db_path = reconstruct_from_template(args.template_db, completed, args.run_id)

    print(f"\n{'='*60}")
    print(f"SUCCESS: {db_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
