#!/usr/bin/env python3
"""
Reconstruct df/df2 using the EXACT file processing order from the logfile.
"""

import re
import json
import pandas as pd
import duckdb
import os
from tqdm import tqdm
import argparse


def extract_file_processing_order(logfile_path):
    """Extract the exact file processing order and prompt counts from logfile."""
    print("Extracting file processing order from logfile...")

    with open(logfile_path, 'r') as f:
        content = f.read()

    # Find all "Processing prompts/.../file.json with N prompts" lines
    pattern = r'Processing prompts/([^/]+)/src/(.+?)\.json with (\d+) prompts'
    matches = re.findall(pattern, content)

    files_info = []
    for prompt_id, file_path, num_prompts_str in matches:
        num_prompts = int(num_prompts_str)
        file_lean = file_path + '.lean'
        module = file_path.replace('/', '.')
        files_info.append({
            'prompt_id': prompt_id,
            'file': file_lean,
            'module': module,
            'num_prompts': num_prompts
        })

    total_prompts = sum(f['num_prompts'] for f in files_info)
    print(f"Found {len(files_info)} files with {total_prompts} total prompts")

    return files_info


def build_df_from_file_order(files_info, prompt_root):
    """Build df using exact file order from logfile."""
    print("Building df from file order...")

    records = []
    global_decl_idx = 0

    for file_info in tqdm(files_info, desc="Processing files"):
        if file_info['num_prompts'] == 0:
            continue

        json_path = os.path.join(prompt_root, 'src', file_info['file'].replace('.lean', '.json'))

        if not os.path.exists(json_path):
            # Skip this file but advance global index
            print(f"Warning: File not found: {json_path}, skipping {file_info['num_prompts']} prompts")
            global_decl_idx += file_info['num_prompts']
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Filter using same logic as inference.py
            local_idx = 0
            for item in data:
                if item["id"]["isExtracted"] or len(item["id"]["errorMsgs"]) != 0:
                    continue
                if item["id"]["kind"] != "theorem":
                    continue

                if local_idx >= file_info['num_prompts']:
                    break

                records.append({
                    'module': file_info['module'],
                    'decl': item["id"]["name"],
                    'decl_idx': global_decl_idx,
                    'raw_prompt': ""  # Would need full construction logic
                })

                global_decl_idx += 1
                local_idx += 1

            # Verify we got the expected number
            if local_idx != file_info['num_prompts']:
                print(f"Warning: {file_info['file']} expected {file_info['num_prompts']} but got {local_idx}")
                # Advance global index by the difference
                global_decl_idx += (file_info['num_prompts'] - local_idx)

        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            global_decl_idx += file_info['num_prompts']
            continue

    df = pd.DataFrame(records)
    print(f"Created df with {len(df)} rows, global_decl_idx reached {global_decl_idx}")
    return df


def parse_completed_prompts(logfile_path):
    """Parse completed prompts from logfile."""
    print("Parsing completed prompts...")

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


def create_df2_and_filter(df, n, completed_prompts):
    """Create df2 and filter to completed rows."""
    print(f"Creating df2 with n={n}...")

    # Create df2
    df2_parts = []
    for i in range(n):
        df_copy = df.copy()
        df_copy["prompt_idx"] = i
        df2_parts.append(df_copy)

    df2 = pd.concat(df2_parts, ignore_index=True)
    print(f"df2 has {len(df2)} rows")

    # Filter to completed
    completed_indices = sorted(completed_prompts.keys())
    df_completed = df2.iloc[completed_indices].copy()

    # Preserve row index
    df_completed['df2_row_idx'] = df_completed.index
    df_completed['answer'] = df_completed.index.map(completed_prompts)
    df_completed = df_completed.reset_index(drop=True)

    print(f"Filtered to {len(df_completed)} completed rows")
    print(f"Index range: {min(completed_indices)} to {max(completed_indices)}")

    return df_completed


def save_to_db(df, run_id):
    """Save to parquet and duckdb."""
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

    # Sample
    sample = con.execute("SELECT df2_row_idx, module, decl FROM run_data ORDER BY df2_row_idx LIMIT 10").fetchall()
    print("\nSample (by df2_row_idx):")
    for row in sample:
        print(f"  [{row[0]}] {row[1]}.{row[2]}")

    con.close()
    return db_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', help='Path to logfile')
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--prompt_root', default='prompts/final_train')
    parser.add_argument('--n', type=int, default=1)

    args = parser.parse_args()

    # Step 1: Extract file processing order FROM LOGFILE
    files_info = extract_file_processing_order(args.logfile)

    # Step 2: Build df using exact same order
    df = build_df_from_file_order(files_info, args.prompt_root)

    # Step 3: Parse completed prompts
    completed_prompts = parse_completed_prompts(args.logfile)

    # Step 4: Create df2 and filter
    df_completed = create_df2_and_filter(df, args.n, completed_prompts)

    # Step 5: Save
    db_path = save_to_db(df_completed, args.run_id)

    print(f"\n{'='*60}")
    print(f"SUCCESS: {db_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
