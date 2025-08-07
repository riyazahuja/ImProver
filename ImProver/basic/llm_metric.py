import ray
import re
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
import os
import torch
import pandas as pd
import json
import datetime
import multiprocessing
import argparse
import duckdb
import random



def run_inference(df, args, metric_config):
    # assuming gpus sit behind different PCIe host bridges on separate
    # NUMA sockets (i.e. nvidia-smi topo -m shows SYS between gpus)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    ray.init(
        num_cpus=args.cpus, num_gpus=args.gpus
    )  # , _temp_dir='/home/riyaza/ray_tmp')
    DataContext.get_current().wait_for_min_actors_s = 1800
    ctx = DataContext.get_current()
    # ctx.progress_bar = True
    # ctx.execution_options.verbose_progress = True

    assert Version(ray.__version__) >= Version(
        "2.44.1"
    ), "Ray version must be at least 2.44.1"

    # ds = ray.data.from_pandas(df)
    # Create a new dataframe with duplicated rows, each with a unique prompt_idx
    df2_parts = []
    for i in range(args.n):
        df_copy = df.copy()
        df_copy["prompt_idx"] = i
        df2_parts.append(df_copy)

    df2 = pd.concat(df2_parts, ignore_index=True)
    # Use df2 instead of df for the Ray dataset
    ds = ray.data.from_pandas(df2).repartition(args.gpus * 12)
    # ds = ray.data.from_pandas(df).repartition(args.gpus * 4)
    # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
    print(ds.schema())

    size = ds.count()
    print(f"Size of dataset: {size} prompts")

    # ctx.execution_options = ExecutionOptions(task_extra_resources={"CPU": 0.25})

    config = vLLMEngineProcessorConfig(
        model_source=args.model if args.model else metric_config['llm']['metric_model'],
        engine_resources={"CPU": args.cpus // args.gpus, "GPU": 1},
        concurrency=args.gpus,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "enable_chunked_prefill": True,
            "max_model_len": 8192,
            "max_num_batched_tokens": 65536,
            # "max_num_batched_tokens": 4096,
            # "max_model_len": 16384,
        },
        max_concurrent_batches=32,
        batch_size=32,
    )

    def postprocess(row):
        text = row["generated_text"]
        
        import re

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

        # scaled_score = score / 5 # now its between -1 and 1


        return dict(answer=score, **row)
        
        # IGNORE vvvv (OLD CODE)
        # match = re.search(r"(\d+)$", text)
        # if match:
        #     score = int(match.group()[-1]) # Extract the last number from the text
        # else:
        #     print(f"Warning: No score found in generated text: {text}. Might need to increase number of generated tokens. ")
        #     score = 0
        # # Will score everything between 0 (first is better) and 10 (second is better), so we need to scale it to the rubric points
        # if row["original_first"]:
        #     score = (score - 5) / 10 * row["points"]
        # else:
        #     score = (5 - score) / 10 * row["points"]
        # return dict(answer=score, **row)

    
    
        
    system = """You are an expert formal mathematician and quality evaluator of formal proofs. You will be given two proofs of the same theorem, and you must determine which proof is better based on how the user tells you to evaluate them.
Think and reason carefully about your answer, listen to exactly what the user tells you to do, and output your final score wrapped in<SCORE>...</SCORE> tags as an integer between -5 and 5 inclusive, 
where -5 means the first proof is much better, 5 means the second proof is much better, and 0 means they are of the same quality -- according to whatever the user's instructions are for how to evaluate the proofs."""
        
    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": row["raw_prompt"]}
                ],
            sampling_params=dict(
                # n=args.n,
                truncate_prompt_tokens=8192 - 512,
                # temperature=0.3,
                max_tokens=512,
            ),
        ),
        postprocess=postprocess,
    )
    ds = vllm_processor(ds).materialize()

    run_output_dir = os.path.join("evals", args.run_id, "readability")
    os.makedirs(run_output_dir, exist_ok=True)

    ds.repartition(16).write_parquet(f"local://{run_output_dir}")

    con = duckdb.connect(os.path.join("evals", args.run_id, "readability.duckdb"))
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS scores AS
        SELECT * FROM read_parquet('{run_output_dir}/*.parquet');
    """
    )

    return run_output_dir

def strip_lean_comments(src: str) -> str:
    """
    Remove *all* Lean-4 comments **and** leading attribute tags.

    • Line comments:             -- … (to end-of-line)
    • Block & doc comments:      /- … -/   and   /-- … -/
      ⋄ Nesting is handled.
    • Attribute tags:            @[ … ]   (e.g. @[simp], @[simp, reducible]).
      ⋄ Only stripped when found outside comments/strings.

    String literals (\" … \") are respected, including \"escaped quotes\".
    All new-lines are preserved so original line numbering is unchanged.
    """
    i, n = 0, len(src)
    out = []
    in_string = False
    in_line   = False           # after "--"
    depth     = 0               # nesting of /- … -/

    while i < n:
        c = src[i]

        # ── inside string literal ────────────────────────────────────────────
        if in_string:
            if c == '\\' and i + 1 < n:                 # keep escape + char
                out.extend(src[i:i+2]); i += 2; continue
            if c == '"':   in_string = False
            out.append(c); i += 1; continue

        # ── inside single-line comment ───────────────────────────────────────
        if in_line:
            if c == '\n':   in_line = False; out.append('\n')
            i += 1; continue

        # ── inside (possibly nested) block/doc comment ───────────────────────
        if depth:
            if src.startswith('-/', i):                # close one level
                depth -= 1; i += 2; continue
            if src.startswith('/--', i):               # nested doc
                depth += 1; i += 3; continue
            if src.startswith('/-', i):                # nested block
                depth += 1; i += 2; continue
            if c == '\n':   out.append('\n')           # keep new-lines
            i += 1; continue

        # ── normal code region ───────────────────────────────────────────────
        # 1) attribute tags  @[ … ]
        if src.startswith('@[', i):
            i += 2
            # skip until corresponding ]
            while i < n and src[i] != ']':
                i += 1
            if i < n: i += 1            # skip closing ]
            # remove trailing spaces/tabs (leave new-line)
            while i < n and src[i] in ' \t':
                i += 1
            continue

        # 2) open/close comment regions
        if src.startswith('/--', i):     depth = 1; i += 3; continue
        if src.startswith('/-',  i):     depth = 1; i += 2; continue
        if src.startswith('--',  i):     in_line = True; i += 2; continue

        # 3) open string
        if c == '"':   in_string = True; out.append(c); i += 1; continue

        # 4) ordinary code char
        out.append(c); i += 1

    return ''.join(out).strip()



def calculate_prompt(proof1, proof2, original_first, metric_config):
    ret = []
    for i, rubric in enumerate(metric_config["llm"]["rubric"]):
        if rubric.get("comments", True):
            proof1, proof2 = strip_lean_comments(proof1), strip_lean_comments(proof2)
        prompt = (
            rubric["text"]
            + f"""Here are the two proofs:

<FIRST_PROOF>
{proof1}
</FIRST_PROOF>

<SECOND_PROOF>
{proof2}
</SECOND_PROOF>
"""
        )
        if original_first:
            ret.append({"raw_prompt": prompt, "points": rubric["points"], "category": i, "original": proof1, "improved": proof2})
        else:
            ret.append({"raw_prompt": prompt, "points": rubric["points"], "category": i, "original": proof2, "improved": proof1})
    return ret



first_time = True

def aggregate_scores(rows, metric_config):
    #given a list of rows with the same rowid, module, and decl, get the final score
    #first, group the rows by prompt_idx, and for each group, do the following:
    # for a group, ensure that we have all of the len(READABILITY_RUBRIC_PROMPTS) categories,
    # and for each item in the group corresponds to a distinct category. Then the answer column (actually the min(answer, points)) is 
    # the final score for that item/category. Then you must simply sum up the scores for each category to get the final score for the group.
    # normalize this group score by the total points available, and then take the mean of these values across the groups. this is the final final score.
    global first_time
    if first_time:
        first_time = False

        # print(f">>> Aggregating scores for {len(rows)} rows [{rows[0]['module']}, {rows[0]['decl']}]")
        # for row in rows:
        #     print(f"  {row['prompt_idx']}: {row['answer']} / {row['points']}\t| {row['prompt']}")
        
    # Group rows by prompt_idx
    prompt_groups = {}
    for row in rows:
        idx = row['prompt_idx']
        if idx not in prompt_groups:
            prompt_groups[idx] = []
        prompt_groups[idx].append(row)

    
    # Calculate total available points
    total_available_points = sum(rubric['points'] for rubric in metric_config["llm"]["rubric"])

    # Process each prompt group
    normalized_scores = []
    for idx, group in prompt_groups.items():
        # Check if we have all categories
        if len(group) != len(metric_config["llm"]["rubric"]):
            print(f"Warning: Group for prompt_idx {idx} has {len(group)} categories instead of {len(metric_config['llm']['rubric'])}")
            continue
            
        # Ensure each category is distinct
        categories = [item['category'] for item in group]
        if len(set(categories)) != len(categories):
            print(f"Warning: Duplicate categories found in group for prompt_idx {idx}")
            continue
            
        # Calculate score for each category
        total_score = 0
        for category in group:
            s_i = category['answer']
            if s_i is None:
                continue
            flip = 1 if category['original_first'] else -1
            total_score += flip * s_i * category['points'] / 5
        normalized_score = total_score / total_available_points
                
        normalized_scores.append(normalized_score)
    
    # Return the mean of normalized scores across all groups
    if not normalized_scores:
        return 0  # Return 0 if no valid groups were found

    # length_diff_penalty_factor = 0.25

    total = sum(normalized_scores)
    # if len(rows[0]["original"].split("\n")) / len(rows[0]["improved"].split("\n")) > 2 or len(rows[0]["improved"].split("\n")) / len(rows[0]["original"].split("\n")) > 2:
    #     total -= length_diff_penalty_factor * total_available_points

    return total / len(normalized_scores)

def get_readability_scores(readability_connection,args, metric_config):
    # if prompts:
    #     condition = "WHERE is_og = TRUE"
    # else:
    #     condition = "WHERE is_og = FALSE"
    prompts=True
    
    
    if readability_connection:
        try:
            
            query = f"SELECT * FROM scores"
            result = readability_connection.execute(query).fetchdf()
            
            if not result.empty:
                # Group by module and decl
                grouped_data = {}
                for _, row in result.iterrows():
                    key = (row['module'], row['decl'], int(row['rowid']))
                    if key not in grouped_data:
                        grouped_data[key] = []
                    grouped_data[key].append(row.to_dict())
                
                # Calculate scores for each group
                final_scores = []
                for (module, decl, rowid), rows in grouped_data.items():
                    score = aggregate_scores(rows, metric_config)
                    data = {
                        'module': module,
                        'decl': decl,
                        'score': score,
                        'rowid': rowid
                    }
                    # if not prompts:
                    #     data['rowid'] = rowid
                        
                    final_scores.append(data)
                
                # Convert to DataFrame
                scores_df = pd.DataFrame(final_scores)
                print(f"Calculated readability scores for {len(scores_df)} original proofs")
                                
                return scores_df
            else:
                print("No original proofs found in the readability database")
                return None
        except Exception as e:
            print(f"Error processing readability scores: {e}")
            return None
    else:
        print("No readability database connection")
        return None

def parse_readabilityDB(args, metric_config):
    readabilityDB_path = os.path.join("evals", args.run_id, "readability.duckdb")
    if readabilityDB_path:
        try:
            readability_connection = duckdb.connect(readabilityDB_path)
            print(f"Connected to existing database at {readabilityDB_path}")
        except Exception as e:
            print(f"Error connecting to existing database: {e}")
            readability_connection = None
            
    # # Query the database to get scores for original/new proof pairs
    # readability_scores_data = get_readability_scores(readability_connection,args)
    # if readability_scores_data is not None:
    #     # Create the directory if it doesn't exist
    #     os.makedirs("prompts", exist_ok=True)

    #     # Connect to or create the database
    #     prompt_db_path = os.path.join("prompts", "readability.duckdb")
    #     try:
    #         prompt_connection = duckdb.connect(prompt_db_path)
    #         print(f"Connected to prompt database at {prompt_db_path}")
            
    #         # Create the table if it doesn't exist
    #         prompt_connection.execute("""
    #             CREATE TABLE IF NOT EXISTS readability_scores (
    #                 module VARCHAR,
    #                 decl VARCHAR,
    #                 score FLOAT,
    #                 PRIMARY KEY (module, decl)
    #             )
    #         """)
            
    #         # Register the DataFrame as a view
    #         prompt_connection.register('temp_scores', readability_scores_data)
            
    #         # Clear existing scores and insert new ones in a transaction
    #         prompt_connection.execute("BEGIN TRANSACTION")
    #         prompt_connection.execute("DELETE FROM readability_scores")
    #         prompt_connection.execute("INSERT INTO readability_scores SELECT module, decl, score FROM temp_scores")
    #         prompt_connection.execute("COMMIT")
            
    #         print(f"Successfully stored {len(readability_scores_data)} readability scores in {prompt_db_path}")
    #         prompt_connection.close()
            
    #     except Exception as e:
    #         print(f"Error storing readability scores: {e}")
    
    model_scores_data = get_readability_scores(readability_connection,args, metric_config)#,prompts=False)
    print(f"Model scores data: {model_scores_data}")
    if model_scores_data is not None:
        # Open connection to eval database
        eval_db_path = os.path.join("evals", args.run_id, "eval.duckdb")
        # prompt_db_path = os.path.join("prompts", "readability.duckdb")

        try:
            eval_connection = duckdb.connect(eval_db_path)
            # prompt_connection = duckdb.connect(prompt_db_path)
    
            print(f"Connected to evaluation database at {eval_db_path}")
            # print(f"Connected to prompt database at {prompt_db_path}")
            # First duplicate the evaluation_results to make a evaluation_results_legacy table
            try:
                # Check if the legacy table already exists
                table_exists = eval_connection.execute("""
                    SELECT count(*) FROM information_schema.tables 
                    WHERE table_name = 'evaluation_results_legacy'
                """).fetchone()[0]
                
                if table_exists == 0:
                    # Create the legacy table
                    eval_connection.execute("""
                        CREATE TABLE evaluation_results_legacy AS 
                        SELECT * FROM evaluation_results
                    """)
                    print("Created evaluation_results_legacy backup table")
                else:
                    print("evaluation_results_legacy table already exists, skipping backup creation")
            except Exception as e:
                print(f"Error creating backup table: {e}")
            # Process each row in model_scores_data
            for _, row in model_scores_data.iterrows():
                rowid = row['rowid']
                module = row['module']
                decl = row['decl']
                score = row['score']
                
                # Get the original score from the prompt database
                # result = prompt_connection.execute(
                #     "SELECT score FROM readability_scores WHERE module = ? AND decl = ?",
                #     [module, decl]
                # ).fetchone()
                
                # if result:
                # delta = float(result[0]) / 10
                
                # Update the row in the eval database
                eval_connection.execute(
                    """
                    UPDATE evaluation_results 
                    SET 
                        new_score = ?,
                        og_score = ?,
                        delta = ?
                    WHERE 
                        rowid = ?
                    """,
                    [0, 0, score, int(rowid)]
                )
                    
                print(f"Updated scores for rowid {rowid}, module {module}, decl {decl}: delta={score}")
                # else:
                #     print(f"Warning: No original score found for module {module}, decl {decl}")
            
            # Commit the changes
            eval_connection.commit()
            print(f"Successfully updated {len(model_scores_data)} rows in the evaluation database")
            
            # Close connections
            eval_connection.close()

            
        except Exception as e:
            print(f"Error updating scores in evaluation database: {e}")
            
    
def randomize_order(proof_data):
    for p in proof_data:
        if random.random() < 0.5:
            p['proof1'], p['proof2'] = p['proof2'], p['proof1']
            p['original_first'] = False
        else:
            p['original_first'] = True


def main(args):
    
    
    # Try to open the eval.duckdb file
    eval_db_path = os.path.join("evals", args.run_id, "eval.duckdb")
    try:
        eval_connection = duckdb.connect(eval_db_path)
        print(f"Successfully connected to {eval_db_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to open evaluation database at {eval_db_path}: {e}")

    # Initialize our dataframe to hold proofs for evaluation
    proof_data = []

    # # Get all improved proofs that are marked as correct
    # improved_proofs = eval_connection.execute("""
    #     SELECT 
    #         rowid, 
    #         module, 
    #         decl, 
    #         new_raw 
    #     FROM 
    #         evaluation_results 
    #     WHERE 
    #         new_correct = TRUE
    # """).fetchall()

    # # Add improved proofs to our data
    # for rowid, module, decl, new_raw in improved_proofs:
    #     if new_raw:  # Ensure we have a valid proof
    #         proof_data.append({
    #             'module': module,
    #             'decl': decl,
    #             'proof': new_raw,
    #             'rowid': int(rowid),
    #             'is_og': False
    #         })

    # Get all pairs of original and improved proofs that are marked as correct and have non-empty proofs
#     query = """
# SELECT a.decl,
#        struct_pack(a.*) AS row_a,
#        struct_pack(b.*) AS row_b
# FROM   evaluation_results AS a
# JOIN   evaluation_results AS b
#        ON  a.decl = b.decl
#        AND a.module = b.module
#        AND a.og_raw != ''
#        AND b.new_raw != ''
#        AND a.rowid != b.rowid
#        AND a.is_og = TRUE
#        AND b.is_og = FALSE
#        AND b.new_correct = TRUE;"""
    query = """SELECT module, decl, og_raw, new_trimmed, rowid 
FROM evaluation_results
WHERE og_raw != '' AND new_trimmed != '' AND og_correct = TRUE AND new_correct = TRUE"""

    df_pairs = eval_connection.execute(query).fetchall()
    # Couldn't be bothered to use pandas here
    for module, decl, og_raw, new_trimmed, rowid in df_pairs:
        proof_data.append({
            'module': module,
            'decl': decl,
            'proof1': og_raw,
            'proof2': new_trimmed,
            'rowid': int(rowid),
        })
    
    randomize_order(proof_data)

    # Load run config to get metric information
    run_config_path = os.path.join("evals", args.run_id, "config.json")
    try:
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
        metric = run_config['metric']
        print(f"Loaded run config: metric={metric}")
    except Exception as e:
        raise RuntimeError(f"Failed to load run config from {run_config_path}: {e}")

    # Load metric config
    metric_config_path = os.path.join("metrics", metric, "config.json")
    try:
        with open(metric_config_path, 'r') as f:
            metric_config = json.load(f)
        print(f"Loaded metric config from {metric_config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load metric config from {metric_config_path}: {e}")

    data = []
    for item in proof_data:
        prompts = calculate_prompt(item["proof1"], item["proof2"], item["original_first"], metric_config)
        data.extend([{**prompt, **item} for prompt in prompts])
    
    proof_df = pd.DataFrame(data)

    

    # returns the path to the directory containing run metadata and the parquet lake

    output_path = run_inference(proof_df, args, metric_config)
    
    
    parse_readabilityDB(args, metric_config)
    
    ray.shutdown()
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates and infers the LLM-based readability metric on a collection of proofs"
    )
    parser.add_argument("run_id", type=str, help="Run ID to use for evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    try:
        available_gpus = torch.cuda.device_count()
    except (ImportError, AttributeError):
        available_gpus = 0

    parser.add_argument(
        "--gpus",
        type=int,
        default=available_gpus,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument("--n", type=int, default=3, help="Best-of-n value (default: 3)")

    args = parser.parse_args()

    main(args)
