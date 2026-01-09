import json
import os

# Load the dataset
with open(
    "/home/riyaza/eval_improver/improver/data/final_dataset_decontaminated.json", "r"
) as f:
    data = json.load(f)

# Initialize a dictionary to store all training data
train_data = {}

# Iterate through each project in the train section
for project_name, file_list in data["train"].items():
    train_data[project_name] = {}

    # For each file in the project
    for file_path in file_list:
        # Convert file path to corresponding JSON path
        json_file_path = f"/home/riyaza/eval_improver/improver/prompts/final_train/src/{file_path}.json"

        # Check if the JSON file exists
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r") as json_file:
                    file_data = json.load(json_file)
                    train_data[project_name][file_path] = file_data
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file_path}: {e}")
        else:
            print(f"File not found: {json_file_path}")

print(f"Loaded training data for {len(train_data)} projects")
for project, files in train_data.items():
    print(f"{project}: {len(files)} files")
