import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import re
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path


def calculate_metrics(filtered_df: pd.DataFrame, minimax="MIN"):
    # Calculate accuracy as ratio of new_correct=True rows to total
    accuracy = (filtered_df["new_correct"] == True).mean()

    # Calculate nonzero_accuracy as ratio of (new_correct=True AND delta<0) rows to total
    nonzero_accuracy = len(
        filtered_df[(filtered_df["new_correct"] == True) & (filtered_df["delta"] > 0)]
    ) / len(filtered_df)

    # For improvement: set delta to -1 for rows with delta=-1, then take mean
    improvement_df = filtered_df.copy()
    improvement_df.loc[improvement_df["delta"] == -1, "delta"] = 0
    improvement = improvement_df["delta"].mean()

    # For nonzero_improvement: mean of delta where new_correct=True and delta>0
    nonzero_rows = filtered_df[
        (filtered_df["new_correct"] == True) & (filtered_df["delta"] > 0)
    ]
    nonzero_improvement = nonzero_rows["delta"].mean() if len(nonzero_rows) > 0 else 0

    return {
        "accuracy": accuracy,
        "nonzero_accuracy": nonzero_accuracy,
        "improvement": improvement,
        "nonzero_improvement": nonzero_improvement,
    }


def run_improver(file_info, repo, n):
    """Run improver on a single file"""
    module, decls = file_info

    # Convert decls list to comma-separated string
    decls_str = ",".join(decls)
    module = module.replace("/", ".").replace(".lean", "")
    # Create output JSON path
    output_json = f"improver_outputs_new/{repo}/{module.replace('.', '_')}.json"

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


def get_data(test_set, repo, n):

    # Create output directory if it doesn't exist
    os.makedirs(f"improver_outputs_new/{repo}", exist_ok=True)

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
    with open(f"improver_outputs_new/{repo}/improver_combined_results.json", "w") as f:
        json.dump(combined_results, f, indent=2)


def to_csv(repo, model):
    # Read the combined results
    with open(
        f"improver_outputs_new/{repo}/{model}/improver_combined_results.json", "r"
    ) as f:
        combined_results = json.load(f)

    # Create a DataFrame from the results
    df = pd.DataFrame(combined_results)

    # Save the DataFrame to a CSV file
    output_csv = f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def main(model, ns,repo):
    n = max(ns)

    # First run ImProver to get data
    # get_data(test_set, repo, n)

    # Convert results to CSV
    to_csv(repo, model)

    # Import extract functions
    sys.path.append("benchmark")

    # Load and analyze the data
    df = pd.read_csv(f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv")
    # For each n in ns, take first n trajectories from the max n run
    metrics_by_n = {}
    for n_prime in ns:
        # Create a copy of the dataframe
        filtered_df = df.copy()

        # Group by module and decl to handle each declaration separately
        filtered_df["row_num"] = filtered_df.groupby(["module", "decl"]).cumcount()

        # Keep only first n_prime rows for each decl
        filtered_df = filtered_df[filtered_df["row_num"] < n_prime]

        # For each decl, keep only the best row based on criteria
        def select_best_row(group):
            # If any row has new_correct = True, select from those
            correct_rows = group[group["new_correct"] == True]
            if len(correct_rows) > 0:
                # Among correct rows, return the one with minimal new_score
                return correct_rows.nsmallest(1, "new_score")
            # If no correct rows, return the first row
            return group.iloc[:1]

        # Apply the selection process to each group
        filtered_df = (
            filtered_df.groupby(["module", "decl"])
            .apply(select_best_row)
            .reset_index(drop=True)
        )

        # Save the current filtered dataframe
        # filtered_df.to_csv(
        #     f"improver_outputs_new/{repo}/{model}/improver_n{n_prime}_filtered.csv",
        #     index=False,
        # )

        # Load the specific file
        # filtered_df = pd.read_csv("improver_outputs_new/MIL/filtered-1shot.csv")

        # filtered_df.to_csv(
        #     f"improver_outputs_new/{repo}/{model}/improver_n{n_prime}_filtered.csv",
        #     index=False,
        # )

        # print(filtered_df)
        # Calculate metrics with the filtered dataframe
        metrics = calculate_metrics(filtered_df)
        metrics_by_n[n_prime] = metrics
        
        print(f"\nMetrics for n={n_prime}:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # Create lists to store metrics for plotting
        ns_list = list(metrics_by_n.keys())
        accuracy = [metrics_by_n[n]["accuracy"] for n in ns_list]
        nonzero_accuracy = [metrics_by_n[n]["nonzero_accuracy"] for n in ns_list]
        mean_improvement = [metrics_by_n[n]["improvement"] for n in ns_list]
        mean_nonzero_improvement = [
            metrics_by_n[n]["nonzero_improvement"] for n in ns_list
        ]
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(ns_list, accuracy, "b-", label="Accuracy")
        plt.plot(ns_list, nonzero_accuracy, "r-", label="Nonzero Accuracy")
        plt.plot(ns_list, mean_improvement, "g-", label="Mean Improvement")
        plt.plot(
            ns_list, mean_nonzero_improvement, "y-", label="Mean Nonzero Improvement"
        )

        plt.xlabel("Number of Attempts (n)")
        plt.ylabel("Metric Value")
        plt.title("ImProver Metrics vs Number of Attempts")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
        plt.savefig(f"improver_outputs_new/{repo}/{model}/metrics_plot.png")
        plt.close()

    # get_data(test_set,repo,n)


if __name__ == "__main__":
    repos = ['MIL',"Mathlib","Compfiles"]
    # ns = [1] + list(range(5, 61, 5))
    ns= range(1,11)
    if len(sys.argv) < 2:
        print("Usage: python eval.py <model1> <model2> ...")
        sys.exit(1)
    models = sys.argv[1:]
    for model in models:
        for repo in repos:
            main(model, ns,repo)
