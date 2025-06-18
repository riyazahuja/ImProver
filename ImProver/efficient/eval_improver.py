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
import duckdb
from glob import glob


async def eval_file(file, args, config):
    st = time.time()
    # output at inference_dir/runID/evals/[file_path].json
    output_path = os.path.join(
        args.inference_dir, args.runID, "evals", file.replace(".lean", ".json")
    )
    # print(file)
    #['lake', 'exe', 'eval_improver', 'Compfiles.Usa2008P1', 'length', 'runs/RUN_20250515_031905', 'runs/RUN_20250515_031905/evals/Compfiles/Usa2008P1.json']['lake', 'exe', 'eval_improver', 'Compfiles.Usa2008P1', 'length', 'runs/RUN_20250515_031905', 'runs/RUN_20250515_031905/evals/Compfiles/Usa2008P1.json']
    cmd = [
        "lake",
        "exe",
        "eval_improver",
        file.replace("/", ".").replace(".lean", ""),
        config["metric"],
        os.path.join(args.inference_dir, args.runID),
        output_path
    ]
    print(" ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
            )
    
    try:
        await asyncio.wait_for(proc.wait(),20*60)

        # stdout, stderr = await proc.communicate()
        # if proc.returncode != 0:
            # print(f">>> Error evaluating {file}: \n\tSTDOUT: {stdout.decode()}\n\tSTDERR: {stderr.decode()}\n")
        # else:
        print(f">>> success on {file}! (took {time.time()-st}s)\n")
        return
    except asyncio.TimeoutError:
        proc.kill()
        print(f">>> [TIME-OUT] {file} (> {20*60}s)")
        return
    except Exception as e:
        print(f">>> Exception running improver on {file}: {str(e)}")
        return


async def main_async(args):
    
    with open(os.path.join(args.inference_dir, args.runID, "config.json"), "r") as f:
        config = json.load(f)
        
    with open(config["dataset"], "r") as f:
        all = json.load(f)
        dataset = all[config["split"]]
        
    files_to_process = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]
    
    semaphore = asyncio.Semaphore(args.cpus)
    progress_bar = tqdm.tqdm(total=len(files_to_process), desc="Processing files")

    async def worker(f):
        async with semaphore:
            ok = await eval_file(f, args, config)
            progress_bar.update(1)
            return ok
    tasks = [asyncio.create_task(worker(file_info)) for file_info in files_to_process]
    await asyncio.gather(*tasks)
    progress_bar.close()
    
    
    
   

    evals_dir = os.path.join(args.inference_dir, args.runID, "evals")
    db_path = os.path.join(args.inference_dir, args.runID, "eval.duckdb")
    con = duckdb.connect(db_path)
    
    #   SAFE MODE
    # con.execute(f"CREATE TABLE IF NOT EXISTS evaluation_results AS SELECT * FROM '{evals_dir}/**/*.json';")
    
    #   UN-SAFE MODE, (but probably better lol)
    con.execute("DROP TABLE IF EXISTS evaluation_results;")
    con.execute(f"CREATE TABLE evaluation_results AS SELECT * FROM '{evals_dir}/**/*.json';")

    
    con.close()
    





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("runID", type=str, help="Run ID to use for evaluation")
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="runs/",
        help="Directory of runs (default: runs/)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )


    args = parser.parse_args()

    asyncio.run(main_async(args))

