import pandas as pd
import json
import os
from pathlib import Path
import re
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def run_improver(file_info, repo, n):
    """Run improver on a single file"""
    module, decls = file_info

    # Convert decls list to comma-separated string
    decls_str = ",".join(decls)
    module = module.replace("/", ".").replace(".lean", "")
    # Create output JSON path
    output_json = f"improver_outputs/{repo}/{module.replace('.', '_')}.json"

    # Construct the lake command
    cmd = [
        "lake",
        "exe",
        "ImProver",
        "--decls",
        decls_str,
        "--best_of_n",
        f"{n}",  # You can make this configurable
        "--json_path",
        output_json,
        module,
    ]

    print(" ".join(cmd))

    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running improver on {module}: {result.stderr}")
        return output_json
    except Exception as e:
        print(f"Exception running improver on {module}: {str(e)}")
        return None


def main(test_set, repo, n):

    # Create output directory if it doesn't exist
    os.makedirs(f"improver_outputs/{repo}", exist_ok=True)

    # Read the test set
    with open(test_set, "r") as f:
        test_set = json.load(f)

    # Create a list of (module, decls) tuples to process
    files_to_process = []
    for file, decls in test_set[repo].items():
        files_to_process.append((file, decls))

    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()

    # Run improver in parallel
    output_jsons = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(
            run_improver,
            files_to_process,
            [repo] * len(files_to_process),
            [n] * len(files_to_process),
        )
        output_jsons.extend([r for r in results if r is not None])

    # Optionally combine all output JSONs into one
    combined_results = []
    for json_file in output_jsons:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_results.extend(data)
                else:
                    combined_results.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {str(e)}")

    # Save combined results
    with open(f"improver_outputs/{repo}/improver_combined_results.json", "w") as f:
        json.dump(combined_results, f, indent=2)


if __name__ == "__main__":
    main("scripts/test_set_no_inst.json", "Mathlib", 5)
