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

SYSTEM_PROMPTS = {
    "length" : "You are an expert Lean4 theorem rewriting assistant. Shorten the current Lean4 theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem. Be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags, as shown in the example. Namely, only return the statment and proof of the current theorem in Lean4 code, wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n",
    "declarativity" : "You are an expert Lean4 theorem rewriting assistant. Rewrite the current Lean4 theorem (wrapped in <CURRENT>...</CURRENT>) to be as declarative in style as possible. We define and measure declarativity as the number of explicitly typed \"have\" statements, which you will aim to maximize insofar as to construct a more readable, structured, and forward-reasoning approach to the proof as possible - while also ensuring that the output is still a correct proof of the theorem. Be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags, as shown in the example. Namely, only return the statment and proof of the current theorem in Lean4 code, wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n",
    "dependency" : "You are an expert Lean4 theorem rewriting assistant. Rewrite the current Lean4 theorem (wrapped in <CURRENT>...</CURRENT>) to be as independent of external theorems and lemmas as possible. Namely, you aim to rewrite the proof to minimize the number of external dependencies - while also ensuring that the output is still a correct proof of the theorem. Be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags, as shown in the example. Namely, only return the statment and proof of the current theorem in Lean4 code, wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n",
    "completion" : "You are an expert Lean4 theorem proving assistant and formal mathematician. Prove the current theorem (wrapped in <CURRENT>...</CURRENT>) with a correct, formal, and complete (sorry-free) Lean4 proof. Be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags, as shown in the example. Namely, only return the statment and proof of the current theorem in Lean4 code, wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n",
    "readability" : """You are an expert Lean4 theorem rewriting assistant. Rewrite the current Lean4 theorem (wrapped in <CURRENT>...</CURRENT>) to be as readable as possible. Namely, you aim to maximize readability as measured by the following rubric: 
1. Clarity and organization - 2 points: The proof should be easy to understand in the mathematical argument it is making, and intermediate "have" statements are clear and placed appropriately.
2. Using outside theorems effectively - 2 points: The proof should use results from Mathlib, etc. to logically progress the proof, and it is clear why such results are relevant to the proof.
3. Clean layout - 2 points: Each line is generally 100 characters or less, uses "·" for casing and proper indentations/newlines to break up proofs with multiple goals.
4. Comments - 1 point: Complex or important points in the proof are commented with a description of the step in question, including what it symbolizes in informal mathematics.
5. Variable conventions - 1 point: Standard, concise, and descriptive variable names are utilized throughout.
6. Automation tactics - 1 point: Powerful automation tactics are used where appropriate in effective places, and replace steps that would be considered straightforward, purely computational/technical, or trivial in an ordinary mathematical argument.
    
    Accordinging to this definition of readability, you will aim to rewrite the proof to maximize the number of points it receives in this rubric - while also ensuring that the output is still a correct proof of the theorem. Be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags, as shown in the example. Namely, only return the statment and proof of the current theorem in Lean4 code, wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n""",
    "conjecturer" : "You are a Lean4 library builder and (formal) mathematician. Given a Lean4 theorem and proof (referred to as the seed theorem) conjecture a formal theorem statement. More explicitly, given a seed theorem, come up with a conjecture that builds off of and expands upon that theorem that may be correct, and is novel, interesting, and useful. This conjecture should be a formal lean4 theorem statement (you can leave the proof as \":= by sorry\"). Feel free to first explore related ideas and concepts at a high level in informal mathematics, but for the final output, be sure to output your final response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags.\n\n"
}

ANNOTATION_PROMPT = " A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response."

CONTEXT_PROMPT = " The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>."

RAG_PROMPT = " The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>."

EXAMPLE_PROMPT = "Here are some examples of such optimization, as wrapped in <EXAMPLES>...</EXAMPLES>. Note that these examples are for illustrative purposes only and should not be copied directly. Instead, use them to understand the kind of optimization expected and apply similar techniques to the current theorem."



def make_config(args):
    config = {
        "example_dir": args.example_dir,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "system_prompt": SYSTEM_PROMPTS,
        "annotation_prompt": ANNOTATION_PROMPT,
        "context_prompt": CONTEXT_PROMPT,
        "rag_prompt": RAG_PROMPT,
        "example_prompt": EXAMPLE_PROMPT,
    }

    config_file = os.path.join(args.prompts_dir, args.prompt_id, "config.json")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {config_file}")


async def calculate_prompt(file_info, args):
    if type(file_info) is str:
        file = file_info
        theorems = []
    elif type(file_info) is dict:
        file = file_info["file"]
        theorems = file_info.get("theorems", [])
    else:
        print(f">>> Invalid file info type: {type(file_info)}")
        return

    
    
    
    st = time.time()
    if len(theorems)!=0:
        cmd = [
            "lake",
            "exe",
            "get_prompts",
            file.replace("/", ".").replace(".lean", ""),
            os.path.join(args.prompts_dir, args.prompt_id, "src"),
            args.python_cmd,
            "--theorems",
            ",".join(theorems) if theorems else ""
        ]
    else:
        cmd = [
            "lake",
            "exe",
            "get_prompts",
            file.replace("/", ".").replace(".lean", ""),
            os.path.join(args.prompts_dir, args.prompt_id, "src"),
            args.python_cmd,
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


def get_parser() -> argparse.ArgumentParser:
    """Return the ``argparse`` parser used by this module."""
    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("--prompt_id", type=str, default="prompts_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default=".prompts",
        help="Directory to output prompts (default: .prompts)",
    )
    parser.add_argument(
        "--example_dir",
        type=str,
        default=".prompts/.prompt_examples",
        help="Directory to prompt examples (default: .prompts/.prompt_examples)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )
    parser.add_argument(
        "--python_cmd",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current)",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    make_config(args)
    asyncio.run(main_async(args))

