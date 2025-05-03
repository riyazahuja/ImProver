import pandas as pd
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
        filtered_df_test = filtered_df_test[filtered_df_test["row_num"] < 1]

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
            filtered_df_test.copy(),
            filtered_df_base.copy(),
            on=["module", "decl"],
            suffixes=("_test", "_base"),
        )

        # Filter for rows where base got it wrong, test got it right, and test showed improvement
        filtered_merged_df = merged_df[
            ((merged_df["new_correct_base"] == True) & (merged_df["delta_base"] > 0))
        ]

        if len(filtered_merged_df) > 0:
            # Create a new dataframe with the required columns
            print(filtered_merged_df)

            prompt_all = filtered_merged_df["original_prompt_test"].values

            prompt_all = [
                re.sub(
                    r"Shorten the current theorem.*?tag\.", "", prompt, flags=re.DOTALL
                )
                for prompt in prompt_all
            ]

            prompt_base = filtered_merged_df["original_prompt_base"].values

            prompt_base = [
                re.sub(
                    r"Shorten the current theorem.*?tag\.", "", prompt, flags=re.DOTALL
                )
                for prompt in prompt_base
            ]
            # Create prompt_rag by removing specified sections from prompt_all
            prompt_rag = []
            prompt_norag = []

            for prompt in prompt_all:
                # Remove the annotation explanation text
                rag_prompt = prompt.replace(
                    "A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response. The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>. ",
                    "",
                )
                norag_prompt = prompt.replace(
                    "The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>. ",
                    "",
                )

                rag_prompt = prompt
                norag_prompt = prompt
                # Remove everything between <CONTEXT> and </CONTEXT>
                rag_prompt = re.sub(
                    r"<CONTEXT>.*?</CONTEXT>", "", rag_prompt, flags=re.DOTALL
                )

                # Remove everything between <ANNOTATION> and </ANNOTATION>
                rag_prompt = re.sub(
                    r"<ANNOTATION>.*?</ANNOTATION>", "", rag_prompt, flags=re.DOTALL
                )

                # Remove everything between <RETRIEVED> and </RETRIEVED>
                norag_prompt = re.sub(
                    r"<RETRIEVED>.*?</RETRIEVED>", "", norag_prompt, flags=re.DOTALL
                )

                prompt_rag.append(rag_prompt)
                prompt_norag.append(norag_prompt)

            result_df = pd.DataFrame(
                {
                    "base_prompt": prompt_base,
                    "norag_prompt": prompt_norag,
                    "rag_prompt": prompt_rag,
                    "all_prompt": prompt_all,
                    "og_raw": filtered_merged_df["og_raw_test"],
                    "og_score": filtered_merged_df["og_score_test"],
                    "new_raw": filtered_merged_df["new_raw_base"],
                    "new_score": filtered_merged_df["new_score_base"],
                    "delta": filtered_merged_df["delta_base"],
                    # "new_raw_test": filtered_merged_df["new_raw_test"],
                    # "new_score_test": filtered_merged_df["new_score_test"],
                    "module": filtered_merged_df["module"],
                    # "model": [model_test] * len(filtered_merged_df),
                    "decl": filtered_merged_df["decl"],
                }
            )
            # result_df = filtered_merged_df

            # Save the resulting dataframe to a CSV file
            output_dir = f"improver_outputs_new/{repo}_examples/"
            os.makedirs(output_dir, exist_ok=True)
            output_csv = f"{output_dir}/examples_all.csv"
            result_df.to_csv(output_csv, index=False)
            print(f"Examples saved to {output_csv} with {len(result_df)} rows")

            return output_csv

        else:
            print(f"No matching rows found for comparison with n={n_prime}")

            return None


if __name__ == "__main__":
    repos = ["MIL", "Compfiles"]
    # ns = [1] + list(range(5, 61, 5))
    # ns = [1] + list(range(4, 65, 4))
    ns = [32]
    if len(sys.argv) < 2:
        print("Usage: python eval.py <model_base> <model_test1> ...")
        sys.exit(1)

    model_base = sys.argv[1]
    models_test = sys.argv[2:]

    for model_test in models_test:
        outputs = []
        for repo in repos:
            csv = main(model_base, model_test, ns, repo)
            outputs.append(csv)

        # Combine all output CSVs
        valid_outputs = [output for output in outputs if output is not None]
        if valid_outputs:
            # Read all valid CSV files
            dfs = [pd.read_csv(csv_path) for csv_path in valid_outputs]

            # Merge all dataframes
            merged_df = pd.concat(dfs, ignore_index=True)

            # Filter by specific declarations
            fixed_decls = [
                "UpperLowerContinuous.upper_basis",
                # "Imo2009P5.imo2009_p5",
                "Bulgaria1998P6.lemma_1",
                # "Bulgaria1998P11.Thue's_lemma",
                # "Usa2023P4.lemma3",
                # "Imo1981P3.ProblemPredicate.m_le_n",
                # "Imo2006P3.imo2006_p3",
                # "C09_S04_3",
                "C03S04.C03_S04_1",
                "C04_S02_24",
                # "C04_S02_20",
            ]

            # Filter the merged dataframe
            filtered_merged_df = merged_df[merged_df["decl"].isin(fixed_decls)]

            # Save merged and filtered results
            output_dir = f"improver_outputs_new/combined_examples"
            os.makedirs(output_dir, exist_ok=True)

            # Save the complete merged dataframe
            merged_output_csv = (
                f"{output_dir}/all_examples_{model_base}_vs_{model_test}.csv"
            )
            filtered_merged_df.to_csv(merged_output_csv, index=False)
            print(
                f"Combined examples saved to {merged_output_csv} with {len(filtered_merged_df)} rows"
            )

            # Create separate .txt files for each prompt type
            prompt_types = ["base", "norag", "rag", "all"]
            for prompt_type in prompt_types:
                output_file = f"{output_dir}/{prompt_type}_examples.txt"
                with open(output_file, "w") as f:
                    for _, row in filtered_merged_df.iterrows():
                        prompt = row[f"{prompt_type}_prompt"]
                        improved = row["new_raw"]
                        f.write(
                            f"<EXAMPLE>\n{prompt}\n{improved}</IMPROVED>\n</EXAMPLE>\n\n"
                        )
                print(f"Created {prompt_type} examples file: {output_file}")
