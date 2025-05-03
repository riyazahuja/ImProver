import pandas as pd
import json
import tqdm
import os
from pathlib import Path
import re
import subprocess
import multiprocessing
import concurrent
from concurrent.futures import ProcessPoolExecutor
import sys
import concurrent
import time
from datetime import datetime
import pytz


def run_improver_parallel(file_info, args):
    """Run improver on a single file"""

    st = time.time()

    repo, module = file_info
    n = args[0]
    model = args[1]
    port = args[2]
    dataset = args[3]
    annotation = args[4]
    context = args[5]
    rag = args[6]
    id = args[7]

    est = pytz.timezone("US/Eastern")
    current_time = datetime.now(est)
    print(
        f"Running {repo}/{module} [{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}]"
    )

    # Convert decls list to comma-separated string
    # decls_str = ",".join(decls)
    module = module.replace("/", ".").replace(".lean", "")
    # Create output JSON path
    output_json = (
        f"improver_outputs_new/{repo}/{model}{id}/{module.replace('.', '_')}.json"
    )

    # Construct the lake command
    cmd = [
        "lake",
        "exe",
        "improver",
        # "--decls",
        # decls_str,
        "--best_of_n",
        f"{n}",  # You can make this configurable
        "--proofAsSorry",
        "false",
        "--model",
        model,
        "--json_path",
        output_json,
        "--endpoint",
        f"http://0.0.0.0:{port}/v1/chat/completions",
        "--annotation",
        annotation,
        "--context",
        context,
        "--rag",
        f"{rag}",
        module,
    ]

    print(" ".join(cmd))

    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f">>> Error running improver on {module}: {result.stderr}\n")
        else:
            print(f">>> success on {module}! (took {time.time()-st}s)\n")
        return (repo, output_json)
    except Exception as e:
        print(f">>> Exception running improver on {module}: {str(e)}")
        return (repo, None)


def main_parallel(repos, *args):
    n = args[0]
    model = args[1]
    port = args[2]
    test_set = args[3]
    annotation = args[4]
    context = args[5]
    rag = args[6]
    id = args[7]

    # Read the test set
    with open(test_set, "r") as f:
        all = json.load(f)
        test_set = all['train']

    combined_results = {}

    for repo in repos:
        os.makedirs(f"improver_outputs_new/{repo}", exist_ok=True)
        os.makedirs(f"improver_outputs_new/{repo}/{model}{id}", exist_ok=True)
        files_to_process = []
        for file in test_set[repo]:
            files_to_process.append((repo, file))

        output_jsons = []
        with ProcessPoolExecutor(max_workers=min(10,multiprocessing.cpu_count())) as executor:
            results = executor.map(
                run_improver_parallel,
                files_to_process,
                [args] * len(files_to_process),
            )
            output_jsons.extend([r for r in results if r is not None])

        combined_results[repo] = []
        for repo_out, json_file in output_jsons:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_results[repo].extend(data)
            except Exception as e:
                print(f"Error reading {json_file}: {str(e)}")

        json_path = (
            f"improver_outputs_new/{repo}/{model}{id}/improver_combined_results.json"
        )
        with open(json_path, "w") as f:
            json.dump(combined_results[repo], f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 8 and len(sys.argv) != 9:
        print(
            "Usage: python eval.py <n> <model> <port> <dataset> <annotation?> <context> <rag> <?id>"
        )
        sys.exit(1)

    test_set = sys.argv[4]
    with open(test_set, "r") as f:
        all = json.load(f)
        test_set = all['train']
    repos = list(test_set.keys())

    id = ""
    if len(sys.argv) == 9:
        id = f"_{sys.argv[8]}"

    main_parallel(repos, *sys.argv[1:8], id)

    # for repo in repos:

    #     main(repo, *sys.argv[1:8])
