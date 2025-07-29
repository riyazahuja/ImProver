import pandas as pd
import json
import tqdm
import os
from pathlib import Path
import re
import subprocess
import multiprocessing
import asyncio
import sys
import time
from datetime import datetime
import argparse
from multiprocessing import cpu_count
import torch
import json
    
def main(args):


    from ..online.prompting.rag import add_to_db, get_rag_string
    import duckdb
    print(f"[IMPROVER: Adding to database {args.prompts_id} with k={args.k}...]")
    add_to_db(args.prompts_id, k=args.k)
    print(f"[IMPROVER: Getting RAG strings for {args.prompts_id} with k={args.k}...]")
    informal_conn = duckdb.connect(os.path.join("prompts", args.prompts_id, "informal_data.duckdb"))
    
    all_prompt_files = []
    for a, b, c in os.walk(os.path.join("prompts", args.prompts_id, "src")):
        for file in c:
            if file.endswith(".json"):
                all_prompt_files.append(os.path.join(a, file))
                contents = json.load(open(os.path.join(a, file), "r"))
                for prompt in contents:
                    name = prompt["id"]["name"]
                    module = prompt["id"]["module"]
                    rag = get_rag_string(informal_conn, name, module, k=args.k)
                    prompt["rag"] = [rag] if rag else []
                json.dump(contents, open(os.path.join(a, file), "w"), indent=4)
    informal_conn.close()
    print(f"[IMPROVER: RAG setup complete for prompts {args.prompts_id} with {len(all_prompt_files)} files.]")




    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("--prompts_id", type=str, default="prompts_" + datetime.now().strftime("%Y%m%d_%H%M%S")),

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )


    parser.add_argument("--k", type=int, default=10)

    
    
    
    args = parser.parse_args()

    main(args)


