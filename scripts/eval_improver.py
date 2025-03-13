import pandas as pd
import json
import os
from pathlib import Path
import re
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys


def run_improver(file_info, args, repo):
    """Run improver on a single file"""
    module, decls = file_info
    n = args[0]
    model = args[1]
    port = args[2]
    dataset=args[3]
    annotation=args[4]
    context = args[5]

    # Convert decls list to comma-separated string
    decls_str = ",".join(decls)
    module = module.replace("/", ".").replace(".lean", "")
    # Create output JSON path
    output_json = f"improver_outputs_new/{repo}/{model}/{module.replace('.', '_')}.json"

    # Construct the lake command
    cmd = [
        "lake",
        "exe",
        "improver",
        "--decls",
        decls_str,
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


def parse_json_to_csv(input_file, output_file):
    # Read the JSON data from file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Extract the relevant fields
    csv_data = []
    for item in data:
        row = {
            "decl": item.get("decl", ""),
            "module": item.get("module", ""),
            "model": item.get("model", ""),
            "method": item.get("method", ""),
            "metric": item.get("metric", ""),
            "n": item.get("n", 0),
            "og_raw": item.get("og_raw", ""),
            "og_score": item.get("og_score", 0),
            "og_errors": item.get("og_errors", ""),
            "og_correct": item.get("og_correct", False),
            "new_raw": item.get("new_raw", ""),
            "new_score": item.get("new_score", 0),
            "new_errors": item.get("new_errors", ""),
            "new_correct": item.get("new_correct", False),
            "delta": item.get("delta", 0),
            "time": item.get("time", -1),
            "syntax_search": item.get("syntax_search", False),
            "mathlib_search": item.get("mathlib_search", False),
            "examples": item.get("examples", 0),
            "annotation": item.get("annotation", False),
        }
        csv_data.append(row)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")



def main(repo, *args):
    n = args[0]
    model = args[1]
    port = args[2]
    test_set=args[3]
    annotation=args[4]
    context = args[5]

    # Create output directory if it doesn't exist
    os.makedirs(f"improver_outputs_new/{repo}", exist_ok=True)
    os.makedirs(f"improver_outputs_new/{repo}/{model}", exist_ok=True)

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
            [args] * len(files_to_process),
            [repo] * len(files_to_process),
            # [n] * len(files_to_process),
            # [model] * len(files_to_process),
            # [port] * len(files_to_process),
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
    json_path = f"improver_outputs_new/{repo}/{model}/improver_combined_results.json"
    csv_path = f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
    
    with open(
        json_path, "w"
    ) as f:
        json.dump(combined_results, f, indent=2)
        
    parse_json_to_csv(json_path,csv_path)
        
    


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python eval.py <n> <model> <port> <dataset> <annotation?> <context>")
        sys.exit(1)
    
    test_set = sys.argv[4]
    with open(test_set, "r") as f:
        test_set = json.load(f)
    repos = list(test_set.keys())
    
    for repo in repos:
        
    
    # n = sys.argv[1]
    # model = sys.argv[2]
    # port = sys.argv[3]
    # repo = sys.argv[4]
    # dataset=sys.argv[5]
    # annotation=sys.argv[6]
    # context = sys.argv[7]
    
    
        main(repo, *sys.argv[1:7])
