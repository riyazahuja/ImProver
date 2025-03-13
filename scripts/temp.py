import json
import csv
import pandas as pd
import sys


# Function to parse JSON data and extract key information
def parse_json_to_csv(input_file, output_file):
    # Read the JSON data from file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Extract the relevant fields
    csv_data = []
    for item in data:
        row = {
            "decl": item.get("decl", ""),
            "module": item.get("module", ""),
            "model": item.get("model", ""),
            "method": item.get("method", ""),
            "metric": item.get("metric", ""),
            "n": item.get("n", 0),
            "og_raw": item.get("og_raw", ""),
            "og_score": item.get("og_score", 0),
            "og_errors": item.get("og_errors", ""),
            "og_correct": item.get("og_correct", False),
            "new_raw": item.get("new_raw", ""),
            "new_score": item.get("new_score", 0),
            "new_errors": item.get("new_errors", ""),
            "new_correct": item.get("new_correct", False),
            "delta": item.get("delta", 0),
            "time": item.get("time", -1),
            "syntax_search": item.get("syntax_search", False),
            "mathlib_search": item.get("mathlib_search", False),
            "examples": item.get("examples", 0),
            "annotation": item.get("annotation", False),
        }
        csv_data.append(row)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    # Display basic statistics
    print(f"\nTotal entries: {len(df)}")
    

if __name__ == "__main__":

    
    
    for repo in ['MIL','Compfiles','Mathlib']:
        model = "Llama-8B"
        json_path = f"improver_outputs_new/{repo}/{model}/improver_combined_results.json"
        csv_path = f"improver_outputs_new/{repo}/{model}/improver_combined_results.csv"
        

        parse_json_to_csv(json_path,csv_path)
