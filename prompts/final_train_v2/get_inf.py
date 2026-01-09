#!/usr/bin/env python3

import os
import json
import glob
from pathlib import Path


def find_all_json_files(directory):
    """
    Recursively find all JSON files in the given directory.
    Returns a list of file paths.
    """
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def get_corresponding_non_v2_path(v2_path):
    """
    Convert a v2 path to its corresponding non-v2 path.
    Replace 'final_train_v2' with 'final_train' in the path.
    """
    return v2_path.replace("final_train_v2", "final_train")


def load_json_file(file_path):
    """
    Load JSON content from file. Returns None if file doesn't exist or has invalid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def save_json_file(file_path, data):
    """
    Save JSON data to file with proper formatting.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


def get_item_name(json_item):
    """
    Extract the name from a JSON item (a dictionary).
    The structure is a dictionary with an 'id' field containing a 'name' field.
    """
    if isinstance(json_item, dict) and "id" in json_item:
        id_obj = json_item.get("id", {})
        if isinstance(id_obj, dict) and "name" in id_obj:
            return id_obj["name"]
    return None


def update_v2_with_informal_fields(v2_data, non_v2_data):
    """
    Update v2 data with informal_proof and informal_statement from non_v2 data.
    Returns True if any updates were made.
    """
    # Both should be lists containing dict objects
    if not isinstance(v2_data, list) or not isinstance(non_v2_data, list):
        print("Data is not in expected list format")
        return False

    if len(v2_data) == 0 or len(non_v2_data) == 0:
        print("One of the data lists is empty")
        return False

    # Create a mapping from name to item for non-v2 data
    non_v2_name_map = {}
    for item in non_v2_data:
        if isinstance(item, dict):
            name = get_item_name(item)
            if name:
                non_v2_name_map[name] = item

    updates_made = False

    # Update each v2 item with matching non-v2 data
    for v2_item in v2_data:
        if not isinstance(v2_item, dict):
            continue

        v2_name = get_item_name(v2_item)
        if v2_name is None:
            continue

        # Find matching non-v2 item
        if v2_name in non_v2_name_map:
            non_v2_item = non_v2_name_map[v2_name]

            # Update informal_proof if it exists in non-v2 and is empty/missing in v2
            if "informal_proof" in non_v2_item:
                if "informal_proof" not in v2_item or not v2_item["informal_proof"]:
                    v2_item["informal_proof"] = non_v2_item["informal_proof"]
                    updates_made = True

            # Update informal_statement if it exists in non-v2 and is empty/missing in v2
            if "informal_statement" in non_v2_item:
                if (
                    "informal_statement" not in v2_item
                    or not v2_item["informal_statement"]
                ):
                    v2_item["informal_statement"] = non_v2_item["informal_statement"]
                    updates_made = True

    return updates_made


def process_json_files():
    """
    Main function to process all JSON files and update v2 versions with informal fields.
    """
    # Define the base directories
    v2_src_dir = "/home/riyaza/eval_improver/improver/prompts/final_train_v2/src"
    non_v2_src_dir = "/home/riyaza/eval_improver/improver/prompts/final_train/src"

    # Find all JSON files in v2 directory
    print(f"Finding JSON files in {v2_src_dir}...")
    v2_json_files = find_all_json_files(v2_src_dir)
    print(f"Found {len(v2_json_files)} JSON files in v2 directory")

    # Process each v2 file
    processed_count = 0
    updated_count = 0

    for v2_file_path in v2_json_files:
        # Calculate corresponding non-v2 path
        non_v2_file_path = get_corresponding_non_v2_path(v2_file_path)

        # Check if corresponding non-v2 file exists
        if not os.path.exists(non_v2_file_path):
            print(f"No corresponding non-v2 file for: {v2_file_path}")
            continue

        # Load both files
        v2_data = load_json_file(v2_file_path)
        non_v2_data = load_json_file(non_v2_file_path)

        if v2_data is None or non_v2_data is None:
            continue

        # Update v2 data with informal fields from non-v2 data
        if update_v2_with_informal_fields(v2_data, non_v2_data):
            # Save the updated v2 file
            if save_json_file(v2_file_path, v2_data):
                updated_count += 1
                print(f"Updated: {v2_file_path}")
            else:
                print(f"Failed to save: {v2_file_path}")
        else:
            print(f"No updates for: {v2_file_path}")

        processed_count += 1

        # Progress indicator
        if processed_count % 50 == 0:
            print(
                f"Processed {processed_count}/{len(v2_json_files)} files, updated {updated_count} files"
            )

    print(f"\nProcessing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Total files updated: {updated_count}")


if __name__ == "__main__":
    process_json_files()
