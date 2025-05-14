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
        "get_prompts",
        file.replace("/", ".").replace(".lean", ""),
        args.metric,
        args.output_dir,
    ]

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
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
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
        default="prompts/",
        help="Directory to output prompts (default: prompts/)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    args = parser.parse_args()

    asyncio.run(main_async(args))


"""


Game plan: replace this with cpu_count()-many aysnc (semaphore bounded) tasks calling out to a lean script to save all prompts

prompt_save format: for each file, keep a json file with:

{
    name (ci) : 
    {
        system : str
        
        example_prompt : str
        examples : [
            {
                context : [
                    {
                        name : str
                        context_item_type : str
                        content : str
                    }
                ]
                retrieved : [
                    {
                        src : str
                        content : str
                    }
                ]
                annotation : str
                current : str
                improved : str
            }
        ]
        
        rag_prompt : str
        rag: [
            {
                src: str
                content: str
            }
        ]
        
        context_prompt : str
        context: [
            {
                name : str
                context_item_type : str
                content : str
            }
        ]
        
        annotation_prompt : str
        annotation: str
        
        current: str
    }
    
}

where we have a prompts directory, with dir structure:
prompts / metric/ repo / [file-path] / name.json



We assign a task for each file in the eval set (test/train), and save it as above.



Once everything is saved, we have a simple inference script that for each file in the eval set 
(and each thm in each file), we load its respective prompt and run it via vllm (all simple python) n times (i.e. collect all T= n*|{# thms in f | f in dataset}|). save the responses to a json file
corresponding to the file name:
model_responses / repo / [file-path] / [id] / name.json  := 
{
    decl : [
        "content" : str,
    ]
}


Now we have an eval script, that takes in an id and test set and extracts out all T model responses (T/n per thm) and splits them into a task for each file.
we run these via a processPoolExecutor (cpu_count() workers). In each process, we first compile our module, saving each CompilationStep and its corresponding environment
to evaluate each of the n responses of our model, as well as the original from the prompt. we save each output (instance) to a json as currently done, and once returned to the python,
run the collection to combined json, convert to csv, and run metric analysis.



"""
