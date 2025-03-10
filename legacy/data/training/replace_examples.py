import os
import re
import sys


def replace_examples_with_theorems(file_paths_with_ids):
    """
    Replaces instances of "example ___" with "theorem <id><i>" in Lean4 files.
    Each file gets a user-provided identifier, and examples within each file are numbered sequentially.
    """
    for file_path, file_id in file_paths_with_ids:
        try:
            # Read the file's content
            with open(file_path, "r") as file:
                content = file.read()

            # Find all instances of "example" followed by space
            example_pattern = re.compile(r"example\s")
            matches = list(example_pattern.finditer(content))

            # Replace each match with "theorem <id><i>"
            new_content = content
            offset = 0  # Track offset due to length changes

            for i, match in enumerate(matches, start=1):
                start = match.start() + offset
                end = match.end() + offset

                replacement = f"theorem {file_id}_{i} "
                new_content = new_content[:start] + replacement + new_content[end:]

                # Update offset based on length difference
                offset += len(replacement) - (end - start)

            # Write the modified content back to the file
            with open(file_path, "w") as file:
                file.write(new_content)

            print(
                f"Processed {file_path}: Replaced {len(matches)} examples with prefix {file_id}"
            )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":

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
        (
            "Tests/MIL/C09_Topology/solutions/Solutions_S02_Metric_Spaces.lean",
            "MIL_C9S2",
        ),
    ]

    HTPI = [("HTPILib/Chap3.lean", "HTPI_C3"), ("HTPILib/Chap5.lean", "HTPI_C5")]

    active = "MIL"
    if active == "MIL":
        file_paths_with_ids = MIL
    elif active == "HTPI":
        file_paths_with_ids = HTPI
    else:
        print("Invalid active dataset")
        sys.exit(1)
    # Extract file paths and ids from command line arguments
    replace_examples_with_theorems(file_paths_with_ids)
