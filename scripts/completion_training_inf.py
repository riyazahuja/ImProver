import os
import pandas as pd
import json
import datetime
import multiprocessing
import argparse


def get_prompt(data, inf_stmt, inf_pf):
    
    instruction = f"""You are an expert Lean4 theorem proving assistant and formal mathematician. Prove the current theorem (wrapped in <CURRENT>...</CURRENT>) with a correct, formal, and complete (sorry-free) Lean4 proof. 
Feel free to first reason and think about the proof of the theorem informally, but for the final output, be sure to output your final response as a complete and correct Lean4 theorem statement and proof wrapped in <IMPROVED>...</IMPROVED> tags.

<CURRENT>
{data.get('current_sorry', '')}
</CURRENT>"""    
    
    
    output = "<IMPROVED>\n"+ data.get('current','') + "\n</IMPROVED>"

    think = f"""It seems that we need to prove the current Lean4 theorem. Looking informally, the theorem statement says that: 
{inf_stmt}

With this in mind, we can see that the informal proof of the theorem is as follows:
{inf_pf}

With this in mind, I will now output my final response as a correct and formal Lean4 Lean4 theorem conjecture wrapped in <IMPROVED>...</IMPROVED> tags.
"""
    if inf_stmt is not None or inf_pf is not None:
        output = "<think>\n" + think + "\n</think>\n" + output
    
    return {
        "instruction": instruction,
        "output": output
    }

def construct_prompts(config_data, data, module, conn, args):
    
    items = []
    for name, decl_data in data.items():
        
        # Query the database for informal statements and proofs matching the declaration
        query = f"""
        SELECT informal_statement, informal_proof 
        FROM informal_data 
        WHERE name = '{name.replace("'", "''")}' AND module = '{module.replace("'", "''")}'
        """
        results = conn.execute(query).fetchall()

        # Extract informal statement and proof if available
        inf_stmt = None
        inf_pf = None
        for result in results:
            if result[0] is not None and result[0].strip():
                inf_stmt = result[0]
            if result[1] is not None and result[1].strip():
                inf_pf = result[1]

        items.append(get_prompt(decl_data, inf_stmt, inf_pf))
    return items
                
        


from pathlib import Path
import duckdb

def get_custom_stem(file_path: str) -> str:
    p = Path(file_path)
    parts = p.parts

    if len(parts) >= 4 and parts[2] == "Mathlib":
        return str(Path(*parts[:4]))
    else:
        return str(Path(*parts[:3]))

def main(args):    
    with open(args.dataset_path, "r") as f:
        all = json.load(f)
        dataset = all[args.split]
    files_to_process = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]

    prompt_root = os.path.join(args.prompts_dir, args.prompts_id)
    config_path = os.path.join(prompt_root, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # df = pd.DataFrame(columns=["file_path", "decl", "decl_idx", "raw_prompt", "output"])
    df = []


    conn = duckdb.connect(os.path.join(args.prompts_dir, args.prompts_id, "informal_data.duckdb"))
        
    for file in files_to_process:
        file_path = os.path.join(prompt_root, file.replace(".lean", ".json"))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data_raw = json.load(f)
                module = file.replace(".lean", "").replace("/", ".")
                prompt_data = construct_prompts(config_data, data_raw, module, conn, args)
                df.extend(prompt_data)
            # print(f"Processing {file_path} with {len(prompt_data)} prompts")
            
            print(f"Processed {module} with {len(prompt_data)} prompts")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create output filename with timestamp

    output_file = os.path.join(args.output_dir, f"informal_completion.jsonl")
    
    # Write DataFrame to JSONL file
    with open(output_file, 'w') as f:
        for item in df:
            f.write(json.dumps(item) + '\n')
    
    print(f"Successfully wrote {len(df)} records to {output_file}")
    
    # # Print statistics about data distribution
    # print(f"Data distribution across {len(data)} files:")
    # for stem, count in sorted(data.items(), key=lambda x: x[1], reverse=True):
    #     print(f"  {stem}: {count} prompts")
                
    
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("prompts_id", type=str, help="Path to dataset JSON file")

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
        "--output_dir",
        type=str,
        default="data/",
        help="Directory to output data (default: data/)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )


    parser.add_argument(
        "--annotation", type=bool, default=False, help="Annotation? (default: False)"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Number of context retrievals (default: 0)",
    )
    parser.add_argument(
        "--rag", type=int, default=0, help="Number of RAG retrievals (default: 0)"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of few-shot example retrievals (default: 0)",
    )

    args = parser.parse_args()

    main(args)
