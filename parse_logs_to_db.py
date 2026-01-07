#!/usr/bin/env python3
"""
Parse inference logs and populate the readability database directly.
"""

import re
import os
import sys
import json
import argparse
import pandas as pd
import duckdb
from ImProver.basic.llm_metric import postprocess, randomize_order, calculate_prompt


def parse_log_file(log_path):
    """
    Parse the log file to extract prompt indices and answers.

    Returns:
        dict: {prompt_idx: answer_text}
    """
    with open(log_path, 'r') as f:
        content = f.read()

    # Split by the separator lines
    sections = content.split('#' * 80)

    results = {}
    for i in range(len(sections) - 1):
        section = sections[i]

        # Look for "Prompt X completed."
        prompt_match = re.search(r'Prompt (\d+) completed\.', section)
        if not prompt_match:
            continue

        prompt_idx = int(prompt_match.group(1))

        # Look for the answer in the next section
        next_section = sections[i + 1]
        answer_match = re.search(r'Answer:\n(.*?)(?=\n#|$)', next_section, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            results[prompt_idx] = answer

    return results


def reconstruct_dataframe(args, metric_config):
    """
    Reconstruct the same dataframe that would have been used for inference.
    """
    eval_db_path = os.path.join("evals", args.run_id, "eval.duckdb")
    try:
        eval_connection = duckdb.connect(eval_db_path)
        print(f"Successfully connected to {eval_db_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to open evaluation database at {eval_db_path}: {e}")

    # Initialize our dataframe to hold proofs for evaluation
    proof_data = []

    query = """SELECT module, decl, og_raw, og_annotated, new_trimmed, new_annotated, rowid
FROM evaluation_results
WHERE og_raw != '' AND new_trimmed != '' AND og_correct = TRUE AND new_correct = TRUE"""

    df_pairs = eval_connection.execute(query).fetchall()
    for (
        module,
        decl,
        og_raw,
        og_annotated,
        new_trimmed,
        new_annotated,
        rowid,
    ) in df_pairs:
        proof_data.append(
            {
                "module": str(module),
                "decl": str(decl),
                "proof1": str(og_raw),
                "proof1_annotated": str(og_annotated),
                "proof2": str(new_trimmed),
                "proof2_annotated": str(new_annotated),
                "rowid": int(rowid),
            }
        )

    # Set the same random seed if you want consistent ordering
    import random
    random.seed(42)  # Use the same seed as your original run
    randomize_order(proof_data)

    data = []
    for item in proof_data:
        prompts = calculate_prompt(
            item["proof1"],
            item["proof1_annotated"],
            item["proof2"],
            item["proof2_annotated"],
            item["original_first"],
            metric_config,
        )
        data.extend([{**prompt, **item} for prompt in prompts])

    proof_df = pd.DataFrame(data)

    # Duplicate rows for best-of-n behavior (preserve original semantics)
    df2_parts = []
    for i in range(args.judge_n):
        df_copy = proof_df.copy()
        df_copy["prompt_idx"] = i
        df2_parts.append(df_copy)
    df2 = pd.concat(df2_parts, ignore_index=True)

    return df2


def main():
    parser = argparse.ArgumentParser(
        description="Parse inference logs and populate readability database"
    )
    parser.add_argument("run_id", type=str, help="Run ID to use for evaluation")
    parser.add_argument("log_file", type=str, help="Path to the log file to parse")
    parser.add_argument(
        "--judge_n", type=int, default=3, help="Best-of-n value (default: 3)"
    )

    args = parser.parse_args()

    # Load run config to get metric information
    run_config_path = os.path.join("evals", args.run_id, "config.json")
    try:
        with open(run_config_path, "r") as f:
            run_config = json.load(f)
        metric = run_config["metric"]
        print(f"Loaded run config: metric={metric}")
    except Exception as e:
        raise RuntimeError(f"Failed to load run config from {run_config_path}: {e}")

    # Load metric config
    metric_config_path = os.path.join("metrics", metric, "config.json")
    try:
        with open(metric_config_path, "r") as f:
            metric_config = json.load(f)
        print(f"Loaded metric config from {metric_config_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load metric config from {metric_config_path}: {e}"
        )

    # Parse the log file
    print(f"Parsing log file: {args.log_file}")
    log_results = parse_log_file(args.log_file)
    print(f"Found {len(log_results)} completed prompts in logs")

    # Reconstruct the original dataframe
    print("Reconstructing original dataframe...")
    df = reconstruct_dataframe(args, metric_config)
    print(f"Reconstructed dataframe with {len(df)} rows")

    # Match log results with dataframe rows
    outputs = []
    matched = 0
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        row_dict = row._asdict()

        if idx in log_results:
            row_dict["generated_text"] = log_results[idx]
            matched += 1
        else:
            # No answer found in logs
            row_dict["generated_text"] = ""
            print(f"Warning: No answer found for prompt {idx}")

        # Apply postprocessing
        from ImProver.basic.llm_metric import postprocess as postprocess_func

        # We need to define postprocess inline since it's nested in the original
        def postprocess(row):
            if row["original"].strip() == row["improved"].strip():
                return dict(answer=0, **row)

            # CUSTOM HEURISTICS (same as in llm_metric.py)
            import difflib
            if row["improved"].strip().startswith(row["original"].strip()):
                return dict(answer=(-5 if row["original_first"] else 5), **row)
            if "".join(
                [
                    li[2]
                    for li in difflib.ndiff(
                        row["original"].replace("\n", "").replace(" ", ""),
                        row["improved"].replace("\n", "").replace(" ", ""),
                    )
                    if li[0] != " "
                ]
            ) in ["by", "byapply", "byexact"]:
                return dict(answer=0, **row)

            text = row["generated_text"]

            # Search for <SCORE>...</SCORE> in the text
            match = re.search(r"<SCORE>(.*?)</SCORE>", text, re.DOTALL)
            if match:
                score_str = match.group(1).strip()
            else:
                # If no <SCORE> tag, try to find <SCORE> and go to end
                match_start = re.search(r"<SCORE>(.*)", text, re.DOTALL)
                if match_start:
                    score_str = match_start.group(1).strip()
                else:
                    # If no <SCORE> tag at all, try from beginning
                    score_str = text.strip()
            if not score_str:
                return dict(answer=None, **row)

            # Now, try to parse score_str as an integer between -5 and 5 inclusive
            try:
                score = int(score_str)
                if score < -5 or score > 5:
                    score = None
            except Exception:
                score = None
            return dict(answer=score, **row)

        result = postprocess(row_dict)
        outputs.append(result)

    print(f"Matched {matched}/{len(df)} prompts from logs")

    # Create output dataframe
    df_out = pd.DataFrame(outputs)

    # Write to parquet and database
    run_output_dir = os.path.join("evals", args.run_id, "readability")
    os.makedirs(run_output_dir, exist_ok=True)

    parquet_file = os.path.join(run_output_dir, "part-00000.parquet")
    try:
        df_out.to_parquet(parquet_file, index=False)
        print(f"Wrote parquet file to {parquet_file}")
    except Exception:
        # Fallback via DuckDB COPY
        con_tmp = duckdb.connect()
        con_tmp.register("df_out", df_out)
        con_tmp.execute(f"COPY df_out TO '{parquet_file}' (FORMAT PARQUET)")
        con_tmp.unregister("df_out")
        con_tmp.close()
        print(f"Wrote parquet file to {parquet_file} (via DuckDB)")

    con = duckdb.connect(os.path.join("evals", args.run_id, "readability.duckdb"))

    con.execute("DROP TABLE IF EXISTS scores;")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS scores AS
        SELECT * FROM read_parquet('{run_output_dir}/*.parquet');
    """
    )
    con.close()

    print(f"Successfully populated readability.duckdb")
    print(f"Output directory: {run_output_dir}")


if __name__ == "__main__":
    main()
