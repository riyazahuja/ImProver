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
    try:
        df.loc[
            df["new_raw"].str.isspace().fillna(True), ["new_score", "new_correct"]
        ] = [
            -1,
            False,
        ]
    except:
        print(repo)
        print(model)
        print(df)
        raise KeyError()

    # Save the DataFrame to a CSV file
    output_csv = f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def get_metrics_for_model(model, n, repo):
    """Process data for a single model and return metrics"""
    # Convert results to CSV if needed
    to_csv(repo, model)

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

    # Calculate metrics with the filtered dataframe
    metrics = calculate_metrics(filtered_df)
    return metrics


def compare_models(models, n, repos):
    """Create bar charts comparing models across different metrics"""
    all_metrics = {}

    # Process each repo
    for repo in repos:
        repo_metrics = {}
        for model in models:
            metrics = get_metrics_for_model(model, n, repo)
            repo_metrics[model] = metrics
        all_metrics[repo] = repo_metrics

    # Create a combined metrics dict for all repos
    combined_metrics = {
        model: {
            metric: 0
            for metric in [
                "accuracy",
                "nonzero_accuracy",
                "improvement",
                "nonzero_improvement",
            ]
        }
        for model in models
    }

    # Calculate average metrics across all repos
    for model in models:
        for metric in [
            "accuracy",
            "nonzero_accuracy",
            "improvement",
            "nonzero_improvement",
        ]:
            values = [all_metrics[repo][model][metric] for repo in repos]
            combined_metrics[model][metric] = sum(values) / len(values)

    # Create bar charts for each repo and the combined results
    create_bar_chart(all_metrics, n, models, repos)

    # Print metrics
    for repo in repos + ["Combined"]:
        print(f"\nMetrics for {repo} at n={n}:")
        for model in models:
            metrics_dict = (
                combined_metrics if repo == "Combined" else all_metrics[repo][model]
            )
            metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics_dict.items()])
            print(f"{model}: {metrics_str}")


def create_bar_chart(all_metrics, n, models, repos):
    """Create bar charts comparing models"""
    # Define metrics to plot
    metrics_to_plot = [
        "accuracy",
        "nonzero_accuracy",
        "improvement",
        "nonzero_improvement",
    ]
    metrics_labels = [
        "Accuracy",
        "Nonzero Accuracy",
        "Improvement",
        "Nonzero Improvement",
    ]

    # Define colors for each metric
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Calculate combined metrics for all repos
    combined_metrics = {}
    for model in models:
        combined_metrics[model] = {}
        for metric in metrics_to_plot:
            values = [all_metrics[repo][model][metric] for repo in repos]
            combined_metrics[model][metric] = sum(values) / len(values)

    # Add combined metrics to all_metrics
    all_metrics["Combined"] = combined_metrics

    # Create plots for each repo + combined
    for repo in repos + ["Combined"]:
        plt.figure(figsize=(12, 7))

        # Set up the bar positions
        x = np.arange(len(models))
        width = 0.2  # Width of the bars

        # Create bars for each metric
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metrics_labels)):
            # Extract values for this metric for all models
            values = [all_metrics[repo][model][metric] for model in models]
            plt.bar(x + (i - 1.5) * width, values, width, label=label, color=colors[i])

        # Add labels, title, and legend
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.title(f"Performance Comparison for {repo} at n={n}")
        plt.xticks(x, models, rotation=45)
        plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
        plt.legend()
        plt.tight_layout()

        # Save the figure
        os.makedirs("improver_outputs_new/comparison_charts", exist_ok=True)
        plt.savefig(
            f"improver_outputs_new/comparison_charts/{repo}_n{n}_comparison.png"
        )
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python multi_rag.py <n_value> <model1> <model2> ...")
        sys.exit(1)

    n = int(sys.argv[1])
    models = sys.argv[2:]
    repos = ["MIL", "Mathlib", "Compfiles"]

    compare_models(models, n, repos)
