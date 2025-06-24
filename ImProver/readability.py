import ray
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


READABILITY_RUBRIC_PROMPTS = [
    {
        "text": """You are an expert at evaluating mathematical proofs in the Lean 4 language. You will be provided a proof, and you must score them with an integer number of points. You should award points as follows:

**Clarity and organization: 2 points**
The proof should receive 2 points in this category if is easy to understand the mathematical argument it is making, and if intermediate "have" statements are clear and placed appropriately. The proof should receive 0 points in this category if it makes use of an overly convoluted proof term, or if it is difficult to interpret the proof informally.

You will think about this criterion, and score the proof based on its description. You will the print out only the score you gave this proof (a single number).""",
        "points": 2,
    },
    {
        "text": """You are an expert at evaluating mathematical proofs in the Lean 4 language. You will be provided a proof, and you must score them with an integer number of points. You should award points as follows:

**Using outside theorems effectively: 2 points**
The proof should receive 2 points in this category if it uses results from Mathlib, etc. to logically progress the proof, and if it is clear why such results are relevant to the proof. The proof should receive 0 points in this category if it attempts to re-prove trivial statements that have already been proven in Mathlib, or previously in the same proof.

You will think about this criterion, and score the proof based on its description. You will the print out only the score you gave this proof (a single number).""",
        "points": 2,
    },
    {
        "text": """You are an expert at evaluating mathematical proofs in the Lean 4 language. You will be provided a proof, and you must score them with an integer number of points. You should award points as follows:

**Clean layout: 2 points**
The proof should receive 2 points in this category if each line is generally 100 characters or less, if "·", indentations, and newlines are used to break up proofs with multiple goals, and if longer tactic proofs are placed on the line following the "by" keyword. The proof should receive 0 points in this category if any of the above style conventions are violated.

You will think about this criterion, and score the proof based on its description. You will the print out only the score you gave this proof (a single number).""",
        "points": 2,
    },
    {
        "text": """You are an expert at evaluating mathematical proofs in the Lean 4 language. You will be provided a proof, and you must score them with an integer number of points. You should award points as follows:

**Comments: 1 points**
The proof should receive 1 points in this category if complex or important points in the proof are commented ("/- ... -/" or "-- ...") with a description of the step in question, including what it symbolizes in informal mathematics. The proof should receive 0 points in this category if its comments are too long or too frequent, or if they are irrelevant to the steps of the proof nearby.

You will think about this criterion, and score the proof based on its description. You will the print out only the score you gave this proof (a single number).""",
        "points": 1,
    },
    {
        "text": """You are an expert at evaluating mathematical proofs in the Lean 4 language. You will be provided a proof, and you must score them with an integer number of points. You should award points as follows:

**Variable conventions: 1 point**
The proof should receive 1 point in this category if "α", "β", "γ" are used as names for general types, "h", "h₁", etc. are used for hypotheses, "m", "n", "k" are used for natural numbers, "i", "j", "k" are used for integers, and uppercase letters are used for types with some mathematical definition ("G" for a group, "R" for a ring, etc.). The proof should receive 0 points in this category if any of the above conventions are violated.

You will think about this criterion, and score the proof based on its description. You will the print out only the score you gave this proof (a single number).""",
        "points": 1,
    },
    {
        "text": """You are an expert at evaluating mathematical proofs in the Lean 4 language. You will be provided a proof, and you must score them with an integer number of points. You should award points as follows:

**Automation tactics: 1 point**
The proof should receive 1 point in this category if powerful automation tactics (such as "simp", "linarith", "ring", "aesop", etc.) are used where appropriate in effective places, and if they replace steps that would be considered straightforward, purely computational/technical, or trivial in an ordinary mathematical argument. The proof should receive 0 points in this category if the proof contains long sequences of tactics that could be replaced by one of the automation tactics mentioned above, or if it overuses these tactics in ineffective places.

You will think about this criterion, and score the proof based on its description. You will the print out only the score you gave this proof (a single number).""",
        "points": 1,
    },
]


def run_inference(df, args):
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
        model_source=args.model,
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
        score = 0
        for token in text.split():
            if token in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                score = int(token)
                break
        return dict(answer=score, **row)

    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["raw_prompt"]}],
            sampling_params=dict(
                # n=args.n,
                truncate_prompt_tokens=8192 - 128,
                # temperature=0.3,
                max_tokens=128,
            ),
        ),
        postprocess=postprocess,
    )
    ds = vllm_processor(ds).materialize()

    run_output_dir = os.path.join(args.output_dir, args.runID, "readability")
    os.makedirs(run_output_dir, exist_ok=True)

    ds.repartition(16).write_parquet(f"local://{run_output_dir}")

    con = duckdb.connect(os.path.join(args.output_dir, args.runID, "readability.duckdb"))
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS scores AS
        SELECT * FROM read_parquet('{run_output_dir}/*.parquet');
    """
    )

    return run_output_dir


def calculate_prompt(proof):
    ret = []
    for i, rubric in enumerate(READABILITY_RUBRIC_PROMPTS):
        prompt = (
            rubric["text"]
            + f"""The proof you will score is the following:
        ```lean
        {proof}
        ```
        
        Remember to output ONLY the final score, without anything else."""
        )
        ret.append({"raw_prompt": prompt, "points": rubric["points"], "category": i})
    return ret


from pathlib import Path


def get_custom_stem(file_path: str) -> str:
    p = Path(file_path)
    parts = p.parts

    if len(parts) >= 4 and parts[2] == "Mathlib":
        return str(Path(*parts[:4]))
    else:
        return str(Path(*parts[:3]))


def aggregate_scores(rows):
    #given a list of rows with the same rowid, module, and decl, get the final score
    #first, group the rows by prompt_idx, and for each group, do the following:
    # for a group, ensure that we have all of the len(READABILITY_RUBRIC_PROMPTS) categories,
    # and for each item in the group corresponds to a distinct category. Then the answer column (actually the min(answer, points)) is 
    # the final score for that item/category. Then you must simply sum up the scores for each category to get the final score for the group.
    # normalize this group score by the total points available, and then take the mean of these values across the groups. this is the final final score.
        
        
    # Group rows by prompt_idx
    prompt_groups = {}
    for row in rows:
        idx = row['prompt_idx']
        if idx not in prompt_groups:
            prompt_groups[idx] = []
        prompt_groups[idx].append(row)
    
    # Calculate total available points
    total_available_points = sum(rubric['points'] for rubric in READABILITY_RUBRIC_PROMPTS)
    
    # Process each prompt group
    normalized_scores = []
    for idx, group in prompt_groups.items():
        # Check if we have all categories
        if len(group) != len(READABILITY_RUBRIC_PROMPTS):
            print(f"Warning: Group for prompt_idx {idx} has {len(group)} categories instead of {len(READABILITY_RUBRIC_PROMPTS)}")
            continue
            
        # Ensure each category is distinct
        categories = [item['category'] for item in group]
        if len(set(categories)) != len(categories):
            print(f"Warning: Duplicate categories found in group for prompt_idx {idx}")
            continue
            
        # Calculate score for each category (min of answer and points)
        group_score = sum(min(item['answer'], item['points']) for item in group)
        
        # Normalize by dividing by total points
        normalized_score = group_score
        normalized_scores.append(normalized_score)
    
    # Return the mean of normalized scores across all groups
    if not normalized_scores:
        return 0  # Return 0 if no valid groups were found
    
    return sum(normalized_scores) / len(normalized_scores)

def get_readability_scores(readability_connection,args,prompts=True):
    if prompts:
        condition = "WHERE is_og = TRUE"
    else:
        condition = "WHERE is_og = FALSE"
        
    
    if readability_connection:
        try:
            
            query = f"SELECT * FROM scores {condition}"
            result = readability_connection.execute(query).fetchdf()
            
            if not result.empty:
                # Group by module and decl
                grouped_data = {}
                for _, row in result.iterrows():
                    key = (row['module'], row['decl'], None if prompts else row['rowid'])
                    if key not in grouped_data:
                        grouped_data[key] = []
                    grouped_data[key].append(row.to_dict())
                
                # Calculate scores for each group
                final_scores = []
                for (module, decl, rowid), rows in grouped_data.items():
                    score = aggregate_scores(rows)
                    data = {
                        'module': module,
                        'decl': decl,
                        'score': score
                    }
                    if not prompts:
                        data['rowid'] = rowid
                        
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

def parse_readabilityDB(args):
    readabilityDB_path = os.path.join(args.output_dir, args.runID, "readability.duckdb")
    if readabilityDB_path:
        try:
            readability_connection = duckdb.connect(readabilityDB_path)
            print(f"Connected to existing database at {readabilityDB_path}")
        except Exception as e:
            print(f"Error connecting to existing database: {e}")
            readability_connection = None
            
    # Query the database to get scores for original proofs
    readability_scores_data = get_readability_scores(readability_connection,args,prompts=True)
    if readability_scores_data is not None:
        # Create the directory if it doesn't exist
        os.makedirs(args.prompts_dir, exist_ok=True)

        # Connect to or create the database
        prompt_db_path = os.path.join(args.prompts_dir, "readability.duckdb")
        try:
            prompt_connection = duckdb.connect(prompt_db_path)
            print(f"Connected to prompt database at {prompt_db_path}")
            
            # Create the table if it doesn't exist
            prompt_connection.execute("""
                CREATE TABLE IF NOT EXISTS readability_scores (
                    module VARCHAR,
                    decl VARCHAR,
                    score FLOAT,
                    PRIMARY KEY (module, decl)
                )
            """)
            
            # Register the DataFrame as a view
            prompt_connection.register('temp_scores', readability_scores_data)
            
            # Clear existing scores and insert new ones in a transaction
            prompt_connection.execute("BEGIN TRANSACTION")
            prompt_connection.execute("DELETE FROM readability_scores")
            prompt_connection.execute("INSERT INTO readability_scores SELECT module, decl, score FROM temp_scores")
            prompt_connection.execute("COMMIT")
            
            print(f"Successfully stored {len(readability_scores_data)} readability scores in {prompt_db_path}")
            prompt_connection.close()
            
        except Exception as e:
            print(f"Error storing readability scores: {e}")
    
    model_scores_data = get_readability_scores(readability_connection,args,prompts=False)
    print(f"Model scores data: {model_scores_data}")
    if model_scores_data is not None:
        # Open connection to eval database
        eval_db_path = os.path.join(args.output_dir, args.runID, "eval.duckdb")
        prompt_db_path = os.path.join(args.prompts_dir, "readability.duckdb")
        
        try:
            eval_connection = duckdb.connect(eval_db_path)
            prompt_connection = duckdb.connect(prompt_db_path)
            
            print(f"Connected to evaluation database at {eval_db_path}")
            print(f"Connected to prompt database at {prompt_db_path}")
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
                new_score = row['score']
                
                # Get the original score from the prompt database
                result = prompt_connection.execute(
                    "SELECT score FROM readability_scores WHERE module = ? AND decl = ?",
                    [module, decl]
                ).fetchone()
                
                if result:
                    og_score = result[0]
                    
                    # Calculate percent change
                    if og_score != 0:
                        delta = (new_score - og_score) / og_score
                    else:
                        delta = None
                    
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
                        [new_score, og_score, delta, int(rowid)]
                    )
                    
                    print(f"Updated scores for rowid {rowid}, module {module}, decl {decl}: og={og_score}, new={new_score}, delta={delta}")
                else:
                    print(f"Warning: No original score found for module {module}, decl {decl}")
            
            # Commit the changes
            eval_connection.commit()
            print(f"Successfully updated {len(model_scores_data)} rows in the evaluation database")
            
            # Close connections
            eval_connection.close()
            prompt_connection.close()
            
        except Exception as e:
            print(f"Error updating scores in evaluation database: {e}")
            
    
    
    
    
    

def main(args):
    if args.inference:    
        promptDB_path = os.path.join(args.prompts_dir, args.prompts_id, "readability.duckdb")
        promptDB_connection = None
        if os.path.exists(promptDB_path):
            try:
                promptDB_connection = duckdb.connect(promptDB_path)
                print(f"Connected to existing database at {promptDB_path}")
            except Exception as e:
                print(f"Error connecting to existing database: {e}")
                promptDB_connection = None
        
        
        # Try to open the eval.duckdb file
        eval_db_path = os.path.join(args.output_dir, args.runID, "eval.duckdb")
        try:
            eval_connection = duckdb.connect(eval_db_path)
            print(f"Successfully connected to {eval_db_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to open evaluation database at {eval_db_path}: {e}")

        # Initialize our dataframe to hold proofs for evaluation
        proof_data = []

        # If we don't have a prompt database connection
        if promptDB_connection is None:
            print("No prompt database connection. Will fetch original proofs from eval database.")
            # Get one row per module+decl with original proofs
            original_proofs = eval_connection.execute("""
                SELECT 
                    MIN(rowid) as rowid, 
                    module, 
                    decl, 
                    ANY_VALUE(og_raw) as og_raw
                FROM 
                    evaluation_results 
                GROUP BY 
                    module, decl
            """).fetchall()
            
            # Add original proofs to our data
            for _, module, decl, old_raw in original_proofs:
                if old_raw:  # Ensure we have a valid proof
                    proof_data.append({
                        'module': module,
                        'decl': decl,
                        'proof': old_raw,
                        'rowid': None,
                        'is_og': True
                    })

        # Get all improved proofs that are marked as correct
        improved_proofs = eval_connection.execute("""
            SELECT 
                rowid, 
                module, 
                decl, 
                new_raw 
            FROM 
                evaluation_results 
            WHERE 
                new_correct = TRUE
        """).fetchall()

        # Add improved proofs to our data
        for rowid, module, decl, new_raw in improved_proofs:
            if new_raw:  # Ensure we have a valid proof
                proof_data.append({
                    'module': module,
                    'decl': decl,
                    'proof': new_raw,
                    'rowid': int(rowid),
                    'is_og': False
                })

        data = []
        for item in proof_data:
            prompts = calculate_prompt(item["proof"])
            data.extend([{**prompt, **item} for prompt in prompts])
        
        proof_df = pd.DataFrame(data)

        

        # returns the path to the directory containing run metadata and the parquet lake
    
        output_path = run_inference(proof_df, args)
    
    
    parse_readabilityDB(args)
    




if __name__ == "__main__":




    parser = argparse.ArgumentParser(
        description="Generates and infers the LLM-based readability metric on a collection of proofs"
    )
    parser.add_argument("runID", type=str, help="Run ID to use for evaluation")
    parser.add_argument("prompts_id", type=str, help="Prompt ID to use for evaluation")
    parser.add_argument(
        "--inference",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run inference (default: True)",
    )    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".evals/",
        help="Directory of runs (default: .evals/)",
    )    
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="Model to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default=".prompts/",
        help="Directory of prompt data (default: .prompts/)",
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
