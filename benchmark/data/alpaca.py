import pandas as pd
import json
import argparse


def create_alpaca_jsonl(input_csv, output_jsonl):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Filter rows where new_correct is True and delta < 0
    filtered_df = df[(df["new_correct"] == True) & (df["delta"] < 0)]

    # Open output file in write mode
    with open(output_jsonl, "w", encoding="utf-8") as out_file:
        for _, row in filtered_df.iterrows():

            current = row["og_raw"]  # Original code snippet
            improved = row["new_raw"]  # Improved code snippet

            instruction = "Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem. Include the output in the <IMPROVED>...</IMPROVED> tag."
            input_text = f"<CURRENT>\n{current}</CURRENT>\n\n<IMPROVED>\n"
            output_text = f"{improved}</IMPROVED>"

            # Create the Alpaca-style JSON object
            alpaca_example = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }

            # Write the JSON object as a single line
            out_file.write(json.dumps(alpaca_example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CSV to an Alpaca-style JSONL file (filtering new_correct==True and delta<0)"
    )
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_jsonl", help="Path to the output JSONL file")
    args = parser.parse_args()

    create_alpaca_jsonl(args.input_csv, args.output_jsonl)
