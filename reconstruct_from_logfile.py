#!/usr/bin/env python3
"""
Properly reconstruct the inference dataframe from incomplete logfile
by mapping global prompt indices to actual file/decl information.
"""

import re
import json
import pandas as pd
import duckdb
import os
from pathlib import Path
import argparse
from tqdm import tqdm


def parse_file_list_from_log(logfile_path):
    """Extract the file processing order and prompt counts from logfile."""
    print(f"Parsing file list from logfile...")

    with open(logfile_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Find the section with file processing info
    files_info = []
    pattern = r'Processing prompts/([^/]+)/src/(.+?)\.json with (\d+) prompts'

    for match in re.finditer(pattern, content):
        prompt_id = match.group(1)
        file_path = match.group(2) + '.lean'
        num_prompts = int(match.group(3))
        files_info.append((prompt_id, file_path, num_prompts))

    print(f"Found {len(files_info)} files with {sum(f[2] for f in files_info)} total prompts")
    return files_info


def build_prompt_index_mapping(files_info, prompt_root):
    """Build mapping from global prompt index to (module, decl, local_decl_idx)."""
    print(f"Building prompt index mapping...")

    mapping = {}
    global_idx = 0

    for prompt_id, file_path, num_prompts in tqdm(files_info, desc="Processing files"):
        if num_prompts == 0:
            continue

        module = file_path.replace('.lean', '').replace('/', '.')
        json_path = os.path.join(prompt_root, 'src', file_path.replace('.lean', '.json'))

        if not os.path.exists(json_path):
            print(f"Warning: File not found: {json_path}")
            # Skip these prompts in the global index
            global_idx += num_prompts
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Use the same filtering logic as inference.py
            local_idx = 0
            for item in data:
                if item["id"]["isExtracted"] or len(item["id"]["errorMsgs"]) != 0:
                    continue
                if item["id"]["kind"] != "theorem":
                    continue

                decl_name = item["id"]["name"]

                # Store mapping for this global index
                mapping[global_idx] = {
                    'module': module,
                    'decl': decl_name,
                    'decl_idx': local_idx,
                    'file_path': file_path
                }

                global_idx += 1
                local_idx += 1

                if local_idx >= num_prompts:
                    break

        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            global_idx += num_prompts
            continue

    print(f"Built mapping for {len(mapping)} prompts")
    return mapping


def parse_completed_prompts_from_log(logfile_path):
    """Parse logfile to extract completed prompt indices and their answers."""
    print(f"Parsing completed prompts from logfile...")

    with open(logfile_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    blocks = re.split(r'#{80,}\n', content)
    completed_prompts = {}

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
            completed_prompts[prompt_num] = improved
        else:
            # Try alternate pattern
            response_match = re.search(r"'content':\s*'<IMPROVED>(.*?)</IMPROVED>'", block, re.DOTALL)
            if response_match:
                improved = response_match.group(1).strip()
                improved = improved.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
                completed_prompts[prompt_num] = improved

    print(f"Found {len(completed_prompts)} completed prompts")
    return completed_prompts


def reconstruct_dataframe(prompt_index_mapping, completed_prompts):
    """Reconstruct the proper dataframe matching completed prompts to their metadata."""
    print(f"Reconstructing dataframe...")

    records = []
    matched = 0
    unmatched = 0

    for global_idx, answer in tqdm(completed_prompts.items(), desc="Building records"):
        if global_idx in prompt_index_mapping:
            info = prompt_index_mapping[global_idx]
            records.append({
                'module': info['module'],
                'decl': info['decl'],
                'decl_idx': info['decl_idx'],
                'prompt_idx': 0,  # Single sample
                'raw_prompt': "",  # Not available from logfile
                'answer': answer
            })
            matched += 1
        else:
            print(f"Warning: No mapping found for prompt {global_idx}")
            # Still include it with unknown info
            records.append({
                'module': f"unknown_module",
                'decl': f"unknown_decl_{global_idx}",
                'decl_idx': global_idx,
                'prompt_idx': 0,
                'raw_prompt': "",
                'answer': answer
            })
            unmatched += 1

    print(f"Matched: {matched}, Unmatched: {unmatched}")
    return pd.DataFrame(records)


def create_duckdb(df, output_path, run_id):
    """Create DuckDB database from dataframe."""

    run_dir = os.path.join("evals", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save as parquet
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    parquet_path = os.path.join(data_dir, "part-00000.parquet")
    print(f"Writing parquet to: {parquet_path}")
    df.to_parquet(parquet_path, index=False)

    # Create DuckDB
    db_path = os.path.join(run_dir, "data.duckdb")
    print(f"Creating DuckDB at: {db_path}")

    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS run_data")
    con.execute(f"CREATE TABLE run_data AS SELECT * FROM read_parquet('{parquet_path}')")

    # Verify
    count = con.execute("SELECT COUNT(*) FROM run_data").fetchone()[0]
    print(f"Created run_data table with {count} rows")

    # Show sample
    sample = con.execute("SELECT module, decl FROM run_data LIMIT 5").fetchall()
    print(f"\nSample records:")
    for row in sample:
        print(f"  {row[0]}.{row[1]}")

    con.close()

    return db_path, run_dir


def main():
    parser = argparse.ArgumentParser(description='Reconstruct inference data from incomplete logfile')
    parser.add_argument('logfile', type=str, help='Path to the incomplete logfile')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID for output')
    parser.add_argument('--prompt_root', type=str, default='prompts/final_train',
                       help='Root directory for prompt JSON files')

    args = parser.parse_args()

    # Step 1: Parse file list from logfile
    files_info = parse_file_list_from_log(args.logfile)

    # Step 2: Build mapping from global index to file/decl info
    prompt_index_mapping = build_prompt_index_mapping(files_info, args.prompt_root)

    # Step 3: Parse completed prompts from logfile
    completed_prompts = parse_completed_prompts_from_log(args.logfile)

    # Step 4: Reconstruct dataframe
    df = reconstruct_dataframe(prompt_index_mapping, completed_prompts)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df[['module', 'decl', 'decl_idx']].head(10))

    # Step 5: Create DuckDB
    db_path, run_dir = create_duckdb(df, args.run_id, args.run_id)

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Run directory: {run_dir}")
    print(f"Total prompts: {len(completed_prompts)}")
    print(f"\nNext steps:")
    print(f"1. Ensure config.json is in {run_dir}/config.json")
    print(f"2. Run evaluation: python ImProver/basic/eval_improver.py {args.run_id}")

    return 0


if __name__ == '__main__':
    exit(main())
