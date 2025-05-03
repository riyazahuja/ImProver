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


def find_missing_decls(repo, model, expected_n):
    """
    Find declarations that don't have the expected number of rows.

    Args:
        repo (str): Repository name
        model (str): Model name
        expected_n (int): Expected number of rows for each declaration

    Returns:
        dict: Dictionary with total decl count and list of missing declarations
    """
    try:
        # Load the CSV file
        csv_file = f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            return {"total_decls": 0, "missing_decls": []}

        df = pd.read_csv(csv_file)

        # Count the number of rows for each declaration
        decl_counts = df.groupby(["module", "decl"]).size().reset_index(name="count")

        # Find declarations with fewer than expected_n rows
        missing_decls = decl_counts[decl_counts["count"] < expected_n]

        # Get the total number of unique declarations
        total_decls = len(decl_counts)

        missing_list = missing_decls[["module", "decl", "count"]].to_dict("records")

        return {"total_decls": total_decls, "missing_decls": missing_list}
    except Exception as e:
        print(f"Error processing {repo}/{model}: {str(e)}")
        return {"total_decls": 0, "missing_decls": []}


def check_all_models_repos(models, repos, expected_n):
    """
    Check all models and repositories for missing declarations.

    Args:
        models (list): List of model names
        repos (list): List of repository names
        expected_n (int): Expected number of rows for each declaration
    """
    summary = {}

    for model in models:
        model_summary = {}
        for repo in repos:
            result = find_missing_decls(repo, model, expected_n)
            model_summary[repo] = result
        summary[model] = model_summary

    # Print summary
    print(f"\n===== Summary for expected n={expected_n} =====")
    for model, repos_data in summary.items():
        print(f"\nModel: {model}")
        for repo, data in repos_data.items():
            total = data["total_decls"]
            missing = len(data["missing_decls"])
            if total > 0:
                print(f"  {repo}: {missing}/{total} missing ({missing/total*100:.2f}%)")
            else:
                print(f"  {repo}: No data")

    # Print detailed missing declarations
    print("\n===== Detailed Missing Declarations =====")
    for model, repos_data in summary.items():
        has_missing = any(
            len(data["missing_decls"]) > 0 for data in repos_data.values()
        )
        if has_missing:
            print(f"\nModel: {model}")
            for repo, data in repos_data.items():
                missing_decls = data["missing_decls"]
                if missing_decls:
                    print(f"  {repo}:")
                    for decl in missing_decls:
                        print(
                            f"    {decl['module']}.{decl['decl']} (has {decl['count']} rows)"
                        )

    return summary


if __name__ == "__main__":
    repos = ["MIL", "Mathlib", "Compfiles"]
    # ns = [1] + list(range(4, 65, 4))

    if len(sys.argv) < 3:
        print("Usage: python find_anomalies.py <expected_n> <model1> <model2> ...")
        sys.exit(1)

    expected_n = int(sys.argv[1])
    models = sys.argv[2:]

    # Check for missing declarations
    anomalies = check_all_models_repos(models, repos, expected_n)

    # # Optionally run the original analysis
    # for model in models:
    #     for repo in repos:
    #         main(model, ns, repo)
    #     aggregate(model, ns, repos)
