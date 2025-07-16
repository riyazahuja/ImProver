import os
import glob
import json

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")
# Count .json files in the same directory and all subdirectories
json_files = glob.glob(os.path.join(script_dir, "**/*.json"), recursive=True)
count = len(json_files)

print(f"Number of .json files: {count}")

# Open and read the final_dataset.json file
dataset_path = "/home/riyaza/eval_improver/improver/train/data/train/final_dataset.json"
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Recursively find all file paths in the "file" subattr from "test" attribute
def extract_file_paths(data):
    file_paths = []
    if isinstance(data, dict):
        if "file" in data:
            file_paths.append(data["file"])
        for value in data.values():
            file_paths.extend(extract_file_paths(value))
    elif isinstance(data, list):
        for item in data:
            file_paths.extend(extract_file_paths(item))
    return file_paths

def extract_file_paths_rich(data):
    file_paths = []
    if isinstance(data, dict):
        if "file" in data:
            file_paths.append((data["file"],data['theorems']))
        for value in data.values():
            file_paths.extend(extract_file_paths(value))
    elif isinstance(data, list):
        for item in data:
            file_paths.extend(extract_file_paths(item))
    return file_paths

file_paths = extract_file_paths(dataset.get("test", {}))
print(f"Extracted {len(file_paths)} file paths from the dataset.")
print(f"Extracted unique file paths: {len(set(file_paths))}")

rich_file_paths = extract_file_paths_rich(dataset.get("test", {}))
print(f"Extracted {len(set(rich_file_paths))} rich file paths from the dataset.")

# Replace .lean with .json
modified_file_paths = [path.replace('.lean', '.json') for path in file_paths]
# print(modified_file_paths)
# Convert json_files to relative paths to script_dir
relative_json_files = [os.path.relpath(path, script_dir) for path in json_files]
# print(relative_json_files)
# Find items in modified_file_paths that are not in relative_json_files
missing_files = [path for path in modified_file_paths if path not in relative_json_files]
other_missing_files = [path for path in relative_json_files if path not in modified_file_paths]
print(f"Modified file paths count: {len(modified_file_paths)}")
print(f"Unique modified files: {len(set(modified_file_paths))}")

print(f"Missing files count: {len(missing_files)}")
print(f"Missing files: {missing_files}")
