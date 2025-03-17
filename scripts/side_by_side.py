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


def main(model_base, model_test, ns, repo):

    # Import extract functions
    sys.path.append("benchmark")

    # Load and analyze the data
    df_test = pd.read_csv(
        f"improver_outputs_new/{repo}/{model_test}/improver_combined_results.csv"
    )
    df_base = pd.read_csv(
        f"improver_outputs_new/{repo}/{model_base}/improver_combined_results.csv"
    )
    # For each n in ns, take first n trajectories from the max n run
    metrics_by_n = {}
    for n_prime in ns:
        # Create a copy of the dataframe
        filtered_df_test = df_test.copy()
        filtered_df_base = df_base.copy()

        # Group by module and decl to handle each declaration separately
        filtered_df_test["row_num"] = filtered_df_test.groupby(
            ["module", "decl"]
        ).cumcount()

        # Keep only first n_prime rows for each decl
        filtered_df_test = filtered_df_test[filtered_df_test["row_num"] < n_prime]

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
        filtered_df_test = (
            filtered_df_test.groupby(["module", "decl"])
            .apply(select_best_row)
            .reset_index(drop=True)
        )

        filtered_df_base["row_num"] = filtered_df_base.groupby(
            ["module", "decl"]
        ).cumcount()

        # Keep only first n_prime rows for each decl
        filtered_df_base = filtered_df_base[filtered_df_base["row_num"] < n_prime]
        # Apply the selection process to each group
        filtered_df_base = (
            filtered_df_base.groupby(["module", "decl"])
            .apply(select_best_row)
            .reset_index(drop=True)
        )
        # Merge the test and base dataframes on module and decl
        merged_df = pd.merge(
            filtered_df_test,
            filtered_df_base,
            on=["module", "decl"],
            suffixes=("_test", "_base"),
        )

        # Filter for rows where base got it wrong, test got it right, and test showed improvement
        filtered_merged_df = merged_df[
            (
                (merged_df["new_correct_base"] == False)
                & (merged_df["new_correct_test"] == True)
                & (merged_df["delta_test"] > 0)
            )
            | (
                (merged_df["new_correct_base"] == True)
                & (merged_df["new_correct_test"] == True)
                & (merged_df["delta_test"] > 0)
                & (merged_df["delta_base"] < merged_df["delta_test"])
            )
        ]

        if len(filtered_merged_df) > 0:
            # Create a new dataframe with the required columns
            result_df = pd.DataFrame(
                {
                    "original_prompt": filtered_merged_df["original_prompt"],
                    "og_raw": filtered_merged_df["og_raw_test"],
                    "og_score": filtered_merged_df["og_score_test"],
                    "new_raw_base": filtered_merged_df["new_raw_base"],
                    "new_score_base": filtered_merged_df["new_score_base"],
                    "new_raw_test": filtered_merged_df["new_raw_test"],
                    "new_score_test": filtered_merged_df["new_score_test"],
                    "module": filtered_merged_df["module"],
                    "model": [model_test] * len(filtered_merged_df),
                    "decl": filtered_merged_df["decl"],
                }
            )
            # result_df = filtered_merged_df

            # Save the resulting dataframe to a CSV file
            output_dir = f"improver_outputs_new/{repo}/comparison"
            os.makedirs(output_dir, exist_ok=True)
            output_csv = f"{output_dir}/{model_base}_vs_{model_test}_n{n_prime}.csv"
            result_df.to_csv(output_csv, index=False)
            print(
                f"Side-by-side comparison saved to {output_csv} with {len(result_df)} rows"
            )
        else:
            print(f"No matching rows found for comparison with n={n_prime}")


if __name__ == "__main__":
    repos = ["MIL", "Mathlib", "Compfiles"]
    # ns = [1] + list(range(5, 61, 5))
    # ns = [1] + list(range(4, 65, 4))
    ns = [64]
    if len(sys.argv) < 2:
        print("Usage: python eval.py <model_base> <model_test1> ...")
        sys.exit(1)

    model_base = sys.argv[1]
    models_test = sys.argv[2:]
    for model_test in models_test:
        for repo in repos:
            main(model_base, model_test, ns, repo)
