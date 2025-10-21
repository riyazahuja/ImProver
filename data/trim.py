import json
import os
from pathlib import Path


def collect_theorems_from_file(file_path):
    """Extract theorem names from a JSON file where isExtracted=False and kind=theorem"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        theorems = []
        for item in data:

            if item.get("id", {}).get("isExtracted") == False and (
                item.get("id", {}).get("kind")
                == "theorem"
                # or item.get("id", {}).get("kind") == "instance"
                # True
            ):
                theorem_name = item.get("id", {}).get("name")
                # print(item.get("id", {}).get("content")[:100])
                if theorem_name:
                    theorems.append(theorem_name)

        return theorems
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def main():
    # Load the original dataset
    dataset_path = (
        "/home/riyaza/eval_improver/improver/data/final_dataset_decontaminated.json"
    )
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Create new dataset structure
    new_dataset = {"train": {}, "test": dataset["test"]}  # Keep test section unchanged

    # Process each repository in the train section
    for repo_name, file_list in dataset["train"].items():
        print(f"Processing repository: {repo_name}")
        new_dataset["train"][repo_name] = []

        for file_path in file_list:
            file_path = file_path.replace(".lean", "")
            # Construct the path to the corresponding JSON file
            json_file_path = f"/home/riyaza/eval_improver/improver/prompts/final_train_v2/src/{file_path}.json"

            # Check if the JSON file exists

            if os.path.exists(json_file_path):
                theorems = collect_theorems_from_file(json_file_path)
                # theorems_trimmed = [thm for i, thm in enumerate(theorems) if i % 5 == 0]

                # Add to new dataset with file and theorems structure
                new_dataset["train"][repo_name].append(
                    {"file": file_path, "theorems": theorems}
                )

                # print(f"  {file_path}: {len(theorems)} theorems")
            else:
                print(f"  Warning: JSON file not found for {json_file_path}")
                # Still add the file but with empty theorems list
                new_dataset["train"][repo_name].append(
                    {"file": file_path, "theorems": []}
                )

    # Save the new dataset
    output_path = "/home/riyaza/eval_improver/improver/data/final_dataset_decontaminated_with_theorems.json"
    with open(output_path, "w") as f:
        json.dump(new_dataset, f, indent=2)

    print(f"\nNew dataset saved to: {output_path}")

    # Print summary statistics
    total_files = 0
    total_theorems = 0
    for repo_name, repo_data in new_dataset["train"].items():
        repo_files = len(repo_data)
        repo_theorems = sum(len(item["theorems"]) for item in repo_data)
        total_files += repo_files
        total_theorems += repo_theorems
        print(f"{repo_name}: {repo_files} files, {repo_theorems} theorems")

    print(f"\nTotal: {total_files} files, {total_theorems} theorems")


if __name__ == "__main__":
    main()
