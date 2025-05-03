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


def calc_stats(name, df, ns):
    n = max(ns)

    # First run ImProver to get data
    # get_data(test_set, repo, n)

    # Convert results to CSV

    # Import extract functions
    sys.path.append("benchmark")

    # Load and analyze the data
    # For each n in ns, take first n trajectories from the max n run
    metrics_by_n = {}
    print(f"Data for {model} on all repos:")
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

        # Calculate utilization for each declaration
        filtered_df["utilization_score"] = 0.0
        filtered_df["used_names"] = None
        filtered_df["total_names"] = 0

        positive_util_indices = []  # Store indices of rows with positive utilization

        for idx, row in filtered_df.iterrows():
            # Extract original prompt and new raw code
            original_prompt = row.get("original_prompt", "")
            new_raw = row.get("new_raw", "")

            # Skip if either field is missing
            if not isinstance(original_prompt, str) or not isinstance(new_raw, str):
                continue

            # Extract content between <RETRIEVED> tags - get the second instance
            retrieved_matches = re.findall(
                r"<RETRIEVED>(.*?)</RETRIEVED>",
                original_prompt,
                re.DOTALL | re.MULTILINE,
            )
            if len(retrieved_matches) < 2:  # Check if we have at least 2 matches
                continue

            retrieved_content = retrieved_matches[1]  # Take the second match
            # Extract all <DOC> blocks
            doc_blocks = re.findall(r"<DOC>(.*?)</DOC>", retrieved_content, re.DOTALL)

            # Extract names from theorem name, lemma name, or def name patterns
            names = []
            for doc in doc_blocks:
                # Match "theorem name", "lemma name", or "def name" patterns
                matches = re.findall(
                    r"(theorem|lemma|def)\s+([\w.\'_]+)", doc, re.IGNORECASE
                )
                names.extend([match[1] for match in matches])

            # Count how many names appear in the new_raw
            if names:  # Only calculate if we found names
                used_names = [name for name in names if name in new_raw]
                utilization = len(used_names) / len(names) if len(names) > 0 else 0

            # Store utilization data in the dataframe
            filtered_df.at[idx, "utilization_score"] = utilization
            filtered_df.at[idx, "used_names"] = used_names
            filtered_df.at[idx, "total_names"] = len(names)

            if utilization > 0:
                positive_util_indices.append(idx)

        # Create dataframe of rows with positive utilization
        positive_util_df = (
            filtered_df.loc[positive_util_indices].copy()
            if positive_util_indices
            else pd.DataFrame()
        )

        # Calculate average utilization
        utilization_scores = filtered_df["utilization_score"].tolist()
        avg_utilization = (
            sum(utilization_scores) / len(utilization_scores)
            if utilization_scores
            else 0
        )
        print(f"Average utilization score for n={n_prime}: {avg_utilization:.2f}")

        # Create a directory for utilization plots if it doesn't exist
        os.makedirs("utilization_plots", exist_ok=True)

        # Plot distribution of utilization scores
        plt.figure(figsize=(10, 6))
        plt.hist(utilization_scores, bins=10, range=(0, 1), edgecolor="black")
        plt.title(
            f"Distribution of Utilization Scores for n={n_prime} (avg: {avg_utilization:.2f})"
        )
        plt.xlabel("Utilization Score")
        plt.ylabel("Number of Declarations")
        plt.grid(alpha=0.3)
        # Create directory for specific model plots if it doesn't exist
        os.makedirs(f"utilization_plots/{name}", exist_ok=True)
        plt.savefig(f"utilization_plots/{name}/utilization_distribution_n{n_prime}.png")
        plt.close()

        # Save rows with positive utilization to file
        if not positive_util_df.empty:
            print(
                f"Found {len(positive_util_df)} rows with positive utilization for n={n_prime}"
            )
            positive_util_df.to_csv(
                f"utilization_plots/positive_utilization_n{n_prime}.csv", index=False
            )

            # Also print top 5 rows with highest utilization
            print("\nTop 5 rows with highest utilization:")
            top_rows = positive_util_df.nlargest(5, "utilization_score")
            for i, (_, row) in enumerate(top_rows.iterrows(), 1):
                print(
                    f"{i}. Module: {row['module']}, Decl: {row['decl']}, Score: {row['utilization_score']:.2f}, Used: {len(row['used_names'])}/{row['total_names']}"
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

    calc_stats(model, combined_df, ns)

    # get_data(test_set,repo,n)


if __name__ == "__main__":
    repos = ["MIL", "Mathlib", "Compfiles"]
    # ns = [1] + list(range(5, 61, 5))
    # ns = [1] + list(range(4, 65, 4))
    ns = [32]
    if len(sys.argv) < 2:
        print("Usage: python eval.py <model1> <model2> ...")
        sys.exit(1)
    models = sys.argv[1:]
    for model in models:
        aggregate(model, ns, repos)
