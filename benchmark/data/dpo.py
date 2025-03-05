import pandas as pd
import json
import argparse


def process_dpo_preferences(data_csv, traj_csv, output_jsonl):
    # Read the data CSV
    df_data = pd.read_csv(data_csv)
    # Filter rows with new_correct==True and delta < 0
    df_data_filtered = df_data[
        (df_data["new_correct"] == True) & (df_data["delta"] < 0)
    ]

    print(f"Loaded {len(df_data_filtered)} rows from {data_csv}")

    # Create a mapping from decl to og_raw (assuming decl is unique or taking the first occurrence)
    decl_to_prompt = (
        df_data_filtered.drop_duplicates(subset=["decl"])[["decl", "og_raw"]]
        .set_index("decl")["og_raw"]
        .to_dict()
    )
    valid_decls = set([s.strip() for s in decl_to_prompt.keys()])
    print(f"Number of valid decls: {len(valid_decls)}")
    # Read the trajectory CSV
    df_traj = pd.read_csv(traj_csv)
    # Filter trajectories to only include rows with decl in valid_decls
    df_traj["decl"] = df_traj["decl"].str.strip()

    df_traj_filtered = df_traj[df_traj["decl"].isin(valid_decls)].copy()
    print(f"Loaded {len(df_traj_filtered)} rows from {traj_csv}")
    print(f"Number of distinct decls: {len(df_traj_filtered['decl'].unique())}")
    output_examples = []

    print(f"num groups: {len(df_traj_filtered.groupby('decl'))}")
    all_false = 0
    lt2 = 0
    # Group by decl
    for decl, group in df_traj_filtered.groupby("decl"):
        # If all responses in the group are incorrect, skip this decl
        if group["correct"].eq(False).all():
            all_false += 1
            # print(f"Skipping decl {decl} because all responses are incorrect")
            continue

        # Remove duplicate responses: trim the "raw" value and drop duplicates
        group["raw_trim"] = group["raw"].astype(str).str.strip()
        group_unique = group.drop_duplicates(subset=["raw_trim"])

        # If there's only one unique response, ignore this decl
        if len(group_unique) < 2:
            # print(
            #     f"Skipping decl {decl} because there are less than 2 unique responses"
            # )
            # print(group_unique)
            lt2 += 1
            continue

        # For each row, assign rank:
        # For incorrect responses, assign rank 99.
        # For correct responses, rank them by ascending score (lower score means better).
        correct_mask = group_unique["correct"] == True
        # If there are correct responses, rank them:
        if correct_mask.sum() > 0:
            # Sort correct rows by score (lowest first) and assign ranks 1,2,...
            correct_rows = group_unique[correct_mask].sort_values(
                by="score", ascending=True
            )
            correct_rows = correct_rows.assign(rank=lambda df: range(1, len(df) + 1))
        else:
            correct_rows = pd.DataFrame(columns=group_unique.columns)

        # For rows that are incorrect, assign rank 99
        incorrect_rows = group_unique[~correct_mask].copy()
        if not incorrect_rows.empty:
            incorrect_rows["rank"] = 99

        # Combine the two
        combined = pd.concat([correct_rows, incorrect_rows])
        # Optionally, sort by rank (this is just for clarity)
        combined = combined.sort_values(by="rank")

        # Create the zephyr.nectar style JSON object
        # Use the og_raw (prompt) from the data CSV corresponding to this decl.
        og_raw = decl_to_prompt.get(decl, "")
        instruction = "Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem. Include the output in the <IMPROVED>...</IMPROVED> tag."
        input_text = f"<CURRENT>\n{og_raw}</CURRENT>\n\n<IMPROVED>\n"

        prompt = instruction + "\n\n" + input_text

        answers = []
        for _, row in combined.iterrows():
            answers.append(
                {"answer": f'{row["raw_trim"]}</IMPROVED>', "rank": int(row["rank"])}
            )

        # Only add the example if there is more than one distinct answer
        if len(answers) < 2:
            continue

        example = {"prompt": prompt, "answers": answers}
        output_examples.append(example)

    # Write out the results as a JSONL file
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for example in output_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Processed {len(output_examples)} examples and saved to {output_jsonl}")
    print(f"Number of decls with all responses incorrect: {all_false}")
    print(f"Number of decls with less than 2 unique responses: {lt2}")
    print(f"Total analyzed: {len(output_examples) + all_false + lt2}")
    print(f"Number of answers: {sum(len(e["answers"]) for e in output_examples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert data and trajectory CSVs to zephyr.nectar style preference pairs for DPO."
    )
    parser.add_argument("data_csv", help="Path to the data CSV file")
    parser.add_argument("traj_csv", help="Path to the trajectory CSV file")
    parser.add_argument("output_jsonl", help="Path to the output JSONL file")

    args = parser.parse_args()
    process_dpo_preferences(args.data_csv, args.traj_csv, args.output_jsonl)
