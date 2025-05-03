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
import numpy as np
from scipy import stats


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

    # For rows where new_raw is whitespace, set new_score to -1 and new_correct to False
    df.loc[df["new_raw"].str.isspace().fillna(True), ["new_score", "new_correct"]] = [
        -1,
        False,
    ]

    # Save the DataFrame to a CSV file
    output_csv = f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def plot_delta_histogram(df, output_path, title="Delta vs Original Score"):
    # Filter to keep only valid data
    filtered_df = df[df["new_correct"] == True].copy()
    filtered_df = filtered_df[filtered_df["delta"] > 0]

    if len(filtered_df) == 0:
        print("No valid data for histogram")
        return

    # Calculate statistics
    mean_delta = filtered_df["delta"].mean()
    median_delta = filtered_df["delta"].median()
    std_delta = filtered_df["delta"].std()

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    plt.scatter(filtered_df["og_score"], filtered_df["delta"], alpha=0.5, color="blue")

    # Add trend line using polynomial fit
    z = np.polyfit(filtered_df["og_score"], filtered_df["delta"], 2)
    p = np.poly1d(z)
    x_range = np.linspace(
        filtered_df["og_score"].min(), filtered_df["og_score"].max(), 100
    )
    plt.plot(x_range, p(x_range), "r-", linewidth=2)

    # Add labels and stats
    plt.xlabel("Original Score")
    plt.ylabel("Delta (Improvement)")
    plt.title(
        f"{title}\nMean: {mean_delta:.4f}, Median: {median_delta:.4f}, Std Dev: {std_delta:.4f}"
    )
    plt.grid(True, alpha=0.3)

    # Add statistical info as text
    stats_text = (
        f"Statistics:\n"
        f"Number of samples: {len(filtered_df)}\n"
        f"Mean delta: {mean_delta:.4f}\n"
        f"Median delta: {median_delta:.4f}\n"
        f"Std deviation: {std_delta:.4f}\n"
        f"Min delta: {filtered_df['delta'].min():.4f}\n"
        f"Max delta: {filtered_df['delta'].max():.4f}\n"
        f"Polynomial fit: {z[0]:.6f}x² + {z[1]:.6f}x + {z[2]:.6f}"
    )
    plt.figtext(0.15, 0.75, stats_text, bbox=dict(facecolor="white", alpha=0.8))

    # Add histogram as inset
    ax_inset = plt.axes([0.6, 0.6, 0.3, 0.25])
    ax_inset.hist(filtered_df["delta"], bins=20, color="green", alpha=0.7)
    ax_inset.set_title("Delta Distribution")
    ax_inset.set_xlabel("Delta")
    ax_inset.set_ylabel("Frequency")

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Histogram saved to {output_path}")


def calc_stats(df, ns, model):
    n = max(ns)

    # Create a copy of the dataframe
    filtered_df = df.copy()

    # Group by module and decl to handle each declaration separately
    filtered_df["row_num"] = filtered_df.groupby(["module", "decl"]).cumcount()

    # Keep only first n rows for each decl
    filtered_df = filtered_df[filtered_df["row_num"] < n]

    # For each decl, keep only the best row based on criteria
    def select_best_row(group):
        # If any row has new_correct = True, select from those
        correct_rows = group[(group["new_correct"] == True) & (group["new_raw"] != "")]
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

    # Calculate metrics
    metrics = calculate_metrics(filtered_df)
    print(f"\nMetrics for {model} on all repos (n={n}):")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Plot delta histogram
    plot_delta_histogram(
        filtered_df,
        f"improver_outputs_new/{model}_delta_histogram.png",
        f"Delta vs Original Score - {model} (All Repos)",
    )


def aggregate(model, ns, repos):
    # Create a list to store all DataFrames
    all_dfs = []

    # Iterate over each repo and read the CSV files
    for repo in repos:
        csv_file = f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # Add a new column to identify the repo
            df["repo"] = repo
            all_dfs.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_dfs, ignore_index=True)

    calc_stats(combined_df, ns, model)


def main(model, ns, repo):
    n = max(ns)

    # Convert results to CSV
    to_csv(repo, model)

    # Import extract functions
    sys.path.append("benchmark")

    # Load and analyze the data
    df = pd.read_csv(
        f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
    )

    # Create a copy of the dataframe
    filtered_df = df.copy()

    # Group by module and decl to handle each declaration separately
    filtered_df["row_num"] = filtered_df.groupby(["module", "decl"]).cumcount()

    # Keep only first n rows for each decl
    filtered_df = filtered_df[filtered_df["row_num"] < n]

    # For each decl, keep only the best row based on criteria
    def select_best_row(group):
        # If any row has new_correct = True, select from those
        correct_rows = group[(group["new_correct"] == True) & (group["new_raw"] != "")]
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

    # Calculate metrics
    metrics = calculate_metrics(filtered_df)
    print(f"\nMetrics for {model} on {repo} (n={n}):")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Plot delta histogram
    output_path = f"improver_outputs_new/{repo}/{model}/delta_histogram.png"
    plot_delta_histogram(
        filtered_df, output_path, f"Delta vs Original Score - {model} ({repo})"
    )


if __name__ == "__main__":
    repos = ["MIL", "Mathlib", "Compfiles"]
    ns = [64]  # Using maximum n for the analysis
    if len(sys.argv) < 2:
        print("Usage: python histo.py <model1> <model2> ...")
        sys.exit(1)
    models = sys.argv[1:]
    for model in models:
        for repo in repos:
            main(model, ns, repo)
        aggregate(model, ns, repos)
