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


async def calculate_prompt(file, args):
    st = time.time()
    cmd = [
        "lake",
        "exe",
        "getPfTree",
        file.replace("/", ".").replace(".lean", ""),
        args.output_dir,
    ]
    print(" ".join(cmd))
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            print(f">>> Error extracting prompts on {file}: {stderr.decode()}\n")
        else:
            print(f">>> success on {file}! (took {time.time()-st}s)\n")
        return
    except Exception as e:
        print(f">>> Exception running improver on {file}: {str(e)}")
        return


async def main_async(args):

    with open(args.dataset_path, "r") as f:
        all = json.load(f)
        dataset = all[args.split]
    files_to_process = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]

    semaphore = asyncio.Semaphore(args.cpus)

    async def run_with_semaphore(file_info):
        async with semaphore:
            return await calculate_prompt(file_info, args)

    tasks = [run_with_semaphore(file_info) for file_info in files_to_process]

    progress_bar = tqdm.tqdm(total=len(tasks), desc="Processing files")

    async def run_with_progress(task):
        result = await task
        progress_bar.update(1)
        return result

    progress_tasks = [run_with_progress(task) for task in tasks]

    await asyncio.gather(*progress_tasks)
    progress_bar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="KG2.75",
        help="Directory to output KG (default: KG2.75)",
    )
    # parser.add_argument(
    #     "--example_dir",
    #     type=str,
    #     default="prompt_examples",
    #     help="Directory to output prompts (default: prompt_examples)",
    # )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    args = parser.parse_args()

    asyncio.run(main_async(args))
