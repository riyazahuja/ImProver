# import pandas as pd
# import json
# import argparse


# def process_dpo_preferences(data_csv, traj_csv, output_jsonl):
#     # Read the data CSV
#     df_data = pd.read_csv(data_csv)
#     # Filter rows with new_correct==True and delta < 0
#     df_data_filtered = df_data[
#         (df_data["new_correct"] == True) & (df_data["delta"] < 0)
#     ]

#     print(f"Loaded {len(df_data_filtered)} rows from {data_csv}")

#     # Create a mapping from decl to og_raw (assuming decl is unique or taking the first occurrence)
#     decl_to_prompt = (
#         df_data_filtered.drop_duplicates(subset=["decl"])[["decl", "og_raw"]]
#         .set_index("decl")["og_raw"]
#         .to_dict()
#     )
#     valid_decls = set([s.strip() for s in decl_to_prompt.keys()])
#     print(f"Number of valid decls: {len(valid_decls)}")
#     # Read the trajectory CSV
#     df_traj = pd.read_csv(traj_csv)
#     # Filter trajectories to only include rows with decl in valid_decls
#     df_traj["decl"] = df_traj["decl"].str.strip()

#     df_traj_filtered = df_traj[df_traj["decl"].isin(valid_decls)].copy()
#     print(f"Loaded {len(df_traj_filtered)} rows from {traj_csv}")
#     print(f"Number of distinct decls: {len(df_traj_filtered['decl'].unique())}")
#     output_examples = []

#     print(f"num groups: {len(df_traj_filtered.groupby('decl'))}")
#     all_false = 0
#     lt2 = 0
#     # Group by decl
#     for decl, group in df_traj_filtered.groupby("decl"):
#         # Prepare data
#         group["raw_trim"] = group["raw"].astype(str).str.strip()
#         group_unique = group.drop_duplicates(subset=["raw_trim"])

#         # Find correct responses
#         correct_responses = group_unique[group_unique["correct"] == True]

#         # Skip if no correct responses
#         if len(correct_responses) == 0:
#             all_false += 1
#             continue

#         # Skip if less than 2 unique responses
#         if len(group_unique) < 2:
#             lt2 += 1
#             continue

#         # Find the best correct response (lowest score)
#         chosen = correct_responses.sort_values("score").iloc[0]

#         # Find the rejected response
#         incorrect_responses = group_unique[group_unique["correct"] == False]
#         if len(incorrect_responses) > 0:
#             # If we have incorrect responses, use the first one
#             rejected = incorrect_responses.iloc[0]
#         else:
#             # If all responses are correct, use the one with highest score
#             rejected = correct_responses.sort_values("score", ascending=False).iloc[0]

#         # Skip if chosen and rejected are the same
#         if chosen["raw_trim"] == rejected["raw_trim"]:
#             continue

#         # Get prompt from the mapping
#         og_raw = decl_to_prompt.get(decl, "")
#         instruction = "Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem. Include the output in the <IMPROVED>...</IMPROVED> tag."
#         curr = f"<CURRENT>\n{og_raw}</CURRENT>\n\n<IMPROVED>\n"

#         # Create example in ChatML Argilla format
#         example = {
#             "prompt": instruction + "\n\n" + curr,
#             "chosen": f'{chosen["raw_trim"]}</IMPROVED>',
#             "rejected": f'{rejected["raw_trim"]}</IMPROVED>',
#         }

#         output_examples.append(example)

#     # Write out the results as a JSONL file
#     with open(output_jsonl, "w", encoding="utf-8") as f:
#         for example in output_examples:
#             f.write(json.dumps(example) + "\n")

#     print(f"Processed {len(output_examples)} examples and saved to {output_jsonl}")
#     print(f"Number of decls with all responses incorrect: {all_false}")
#     print(f"Number of decls with less than 2 unique responses: {lt2}")
#     print(f"Total analyzed: {len(output_examples) + all_false + lt2}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Convert data and trajectory CSVs to ChatML Argilla format preference pairs for DPO."
#     )
#     parser.add_argument("data_csv", help="Path to the data CSV file")
#     parser.add_argument("traj_csv", help="Path to the trajectory CSV file")
#     parser.add_argument("output_jsonl", help="Path to the output JSONL file")

#     args = parser.parse_args()
#     process_dpo_preferences(args.data_csv, args.traj_csv, args.output_jsonl)


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
        # Prepare data
        group["raw_trim"] = group["raw"].astype(str).str.strip()
        group_unique = group.drop_duplicates(subset=["raw_trim"])

        # Find correct responses
        correct_responses = group_unique[group_unique["correct"] == True]

        # Skip if no correct responses
        if len(correct_responses) == 0:
            all_false += 1
            continue

        # Skip if less than 2 unique responses
        if len(group_unique) < 2:
            lt2 += 1
            continue

        # Find the best correct response (lowest score)
        chosen = correct_responses.sort_values("score").iloc[0]

        # Find the rejected response
        incorrect_responses = group_unique[group_unique["correct"] == False]
        if len(incorrect_responses) > 0:
            # If we have incorrect responses, use the first one
            rejected = incorrect_responses.iloc[0]
        else:
            # If all responses are correct, use the one with highest score
            rejected = correct_responses.sort_values("score", ascending=False).iloc[0]

        # Skip if chosen and rejected are the same
        if chosen["raw_trim"] == rejected["raw_trim"]:
            continue

        # Get prompt from the mapping
        og_raw = decl_to_prompt.get(decl, "")
        instruction = "Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem. Include the output in the <IMPROVED>...</IMPROVED> tag."
        curr = f"<CURRENT>\n{og_raw}</CURRENT>\n\n<IMPROVED>\n"

        # Create example in ChatML Argilla format
        example = {
            "prompt": instruction + "\n\n" + curr,
            "chosen": f'{chosen["raw_trim"]}</IMPROVED>',
            "rejected": f'{rejected["raw_trim"]}</IMPROVED>',
        }

        output_examples.append(example)

    # Write out the results as a JSONL file
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for example in output_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Processed {len(output_examples)} examples and saved to {output_jsonl}")
    print(f"Number of decls with all responses incorrect: {all_false}")
    print(f"Number of decls with less than 2 unique responses: {lt2}")
    print(f"Total analyzed: {len(output_examples) + all_false + lt2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert data and trajectory CSVs to ChatML Argilla format preference pairs for DPO."
    )
    parser.add_argument("data_csv", help="Path to the data CSV file")
    parser.add_argument("traj_csv", help="Path to the trajectory CSV file")
    parser.add_argument("output_jsonl", help="Path to the output JSONL file")

    args = parser.parse_args()
    process_dpo_preferences(args.data_csv, args.traj_csv, args.output_jsonl)
