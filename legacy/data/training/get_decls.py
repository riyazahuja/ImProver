import csv
import json
import re
from collections import defaultdict

# File to ID mappings
MIL = [
    (
        "Tests/MIL/C03_Logic/solutions/Solutions_S02_The_Existential_Quantifier.lean",
        "MIL_C3S2",
    ),
    (
        "Tests/MIL/C05_Elementary_Number_Theory/solutions/Solutions_S03_Infinitely_Many_Primes.lean",
        "MIL_C5S3",
    ),
    (
        "Tests/MIL/C09_Topology/solutions/Solutions_S03_Topological_Spaces.lean",
        "MIL_C9S3",
    ),
    ("Tests/MIL/C03_Logic/solutions/Solutions_S05_Disjunction.lean", "MIL_C3S5"),
    (
        "Tests/MIL/C03_Logic/solutions/Solutions_S04_Conjunction_and_Iff.lean",
        "MIL_C3S4",
    ),
    (
        "Tests/MIL/C04_Sets_and_Functions/solutions/Solutions_S02_Functions.lean",
        "MIL_C4S2",
    ),
    ("Tests/MIL/C09_Topology/solutions/Solutions_S02_Metric_Spaces.lean", "MIL_C9S2"),
]
HTPI = [("HTPILib/Chap3.lean", "HTPI_C3"), ("HTPILib/Chap5.lean", "HTPI_C5")]

# Combine all file to ID mappings
FILE_TO_ID = dict(MIL + HTPI)


def get_declarations_by_repo_and_file(csv_path):
    # Create a nested defaultdict for storing data in the structure repo:file:[decls...]
    repo_file_decls = defaultdict(lambda: defaultdict(set))

    # Track example counter per file
    example_counters = defaultdict(int)

    # Regex pattern to match lemma/theorem names
    pattern = re.compile(r"(lemma|theorem|problem|def)\s+(\S+)")
    example_pattern = re.compile(r"example\s+.*")
    negative_pattern = re.compile(r"(abbrev|instance)")

    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            repo = row["repo"]
            file = row["file"]
            decl = row["decl"]

            # Get file ID if available
            file_id = FILE_TO_ID.get(file, "unknown")

            # Check if it's an example declaration
            example_match = example_pattern.match(decl)
            if example_match:
                # Increment counter for this file
                example_counters[file] += 1
                # Create the new theorem name with format <id>_<i>
                new_theorem_name = f"{file_id}_{example_counters[file]}"
                # Replace with the new theorem name
                processed_decl = new_theorem_name
                # Store the full declaration for potential file modification later
                repo_file_decls[repo][file].add(processed_decl)
                continue

            # Apply the regular pattern matching
            match = pattern.search(decl)
            if match:
                # Replace the declaration with just the name part (group 2)
                processed_decl = match.group(2)
            elif negative_pattern.search(decl):
                # Skip the declaration if it matches the negative pattern
                continue
            else:
                # Keep the original declaration if no match
                processed_decl = decl

            # Add the processed declaration to the set of declarations for this file in this repo
            repo_file_decls[repo][file].add(processed_decl)

    # Convert the defaultdict to a regular dict and sets to lists for JSON serialization
    result = {}
    for repo, files in repo_file_decls.items():
        result[repo] = {file: list(decls) for file, decls in files.items()}

    return result, example_counters


def main():
    # Specify the path to your CSV file
    csv_path = "FINAL/data.csv"

    # Get the declarations organized by repo and file
    repo_file_decls, example_counters = get_declarations_by_repo_and_file(csv_path)

    # Write the result to a JSON file
    output_path = "repo_file_declarations_3.json"
    with open(output_path, "w") as json_file:
        json.dump(repo_file_decls, json_file, indent=2)

    print(f"Declarations organized by repo and file have been written to {output_path}")
    print(f"Example count by file: {dict(example_counters)}")


if __name__ == "__main__":
    main()
