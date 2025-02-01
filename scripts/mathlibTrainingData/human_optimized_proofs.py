# Data extraction script for mathlib4 repository
# Searches for theorems that have been manually optimized for length by human mathematicians, and returns them in plaintext without context

# Before using, set the REPO_PATH variable to the path of the mathlib repository on your local machine (or run without it to automatically clone mathlib)

# Usage: get_improved_theorems() returns a generator of tuples (original theorem, improved theorem)
#        save_to_csv() saves all significantly improved theorems to a CSV file

# Made by Tate Rowney

import os, subprocess, sys

REPO_PATH = ""

if not REPO_PATH:
    if input("OK to clone mathlib repository to current directory? Select \"N\" to specify existing local mathlib directory (y/N) ").lower().strip() == "y":
        REPO_PATH = os.getcwd()+"/mathlib4"
        print("Cloning mathlib repository to", REPO_PATH)
        res, error = subprocess.Popen(["git", "clone", "https://github.com/leanprover-community/mathlib4.git"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        if error:
            print("Error cloning repository:", error)
            sys.exit()
    else:
        REPO_PATH = input("Enter the path to the mathlib repository: ")


# Requires: hash of a commit, path to a file in the repository
# Returns: a list of pairs of valid theorems (original, improved) without context that have been changed during this commit
def get_improvement_candidates(commithash : str, filepath : str):
    original, error = subprocess.Popen(["git", "show", commithash + "^:" + filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_PATH).communicate()
    if error:
        # print("!!!!!!!!!!!!! Error in getting original file:", error, "!!!!!!!!!!!!!")
        return []
    original = get_theorems(original.decode("utf-8"))

    improved, error = subprocess.Popen(["git", "show", commithash + ":" + filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_PATH).communicate()
    if error:
        # print("!!!!!!!!!!!!! Error in getting improved file:", error, "!!!!!!!!!!!")
        return []
    improved = get_theorems(improved.decode("utf-8"))

    diff_raw, error = subprocess.Popen(["git", "diff", commithash + "^", commithash, "-U0", filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_PATH).communicate()
    if error:
        # print("!!!!!!!!!!!!! Error in getting diff:", error, "!!!!!!!!!!!!!")
        return []
    diff_raw = diff_raw.decode("utf-8")

    # Get the lines of stuff that has been changed in the commit, and store it in a list of tuples (original line number, improved line number)
    diff_lines = []
    for line in diff_raw.split("\n"):
        if line.startswith("@@"):
            diff_loc = line.split("@@")[1].split("@@")[0]
            # want stuff where they're revising, not just adding or removing
            # TODO: maybe change this to work with removing unnecessary tactics, etc.
            if "-" in diff_loc and "+" in diff_loc:
                original_line = int(diff_loc.split("+")[0].split("-")[1].split(",")[0].strip())
                improved_line = int(diff_loc.split("+")[1].split(",")[0].strip())
                diff_lines.append((original_line, improved_line))

    candidates = []

    # Filter out some candidate modified theorems
    for theorem_name in original.keys():
        # Is a theorem with the same name present in
        if theorem_name in improved:
            # check if these theorems were modified at all
            original_lines = original[theorem_name][1]
            improved_lines = improved[theorem_name][1]
            for original_line, improved_line in diff_lines:
                if original_line in original_lines and improved_line in improved_lines:
                    # if so, return them
                    candidates.append((original[theorem_name][0], improved[theorem_name][0]))

    return candidates


# Requires: plaintext of a lean file with default syntax, etc.
# Returns: A dictionary of {theorem name: (theorem text, range of lines covered)}
def get_theorems(leanfile):
    segments = {}

    keywords = ["theorem", "example", "problem", "def", "variable", "namespace", "open", "section", "noncomputable", "end", "import", "section", "axiom", "lemma", "universe", "instance", "structure", "attribute", "class", "alias"]
    leanfile = leanfile.split("\n")

    block_start, block_end = 0, 0
    current_keyword, current_name = "", ""
    in_block_comment = False

    # Scan through the file, separating into syntactic blocks (stuff marked by "theorem", "def", etc.); keep the theorems
    while block_end < len(leanfile):
        # skip over theorems, etc. that are commented out
        if "/-" in leanfile[block_end]:
            in_block_comment = True

        if in_block_comment:
            if "-/" in leanfile[block_end]:
                in_block_comment = False
            block_end += 1
            continue

        else:
            first_word = leanfile[block_end].split(" ")[0]
            if first_word in keywords:
                start_next_search_loc = block_end

                # include comments and annotations at the top
                in_block_comment_inner = False
                for step_backward in range(block_end-1, 0, -1):
                    if leanfile[step_backward].strip().startswith("@") or leanfile[step_backward].strip().startswith("--") or leanfile[step_backward].strip() == "":
                        block_end = step_backward
                    elif "-/" in leanfile[step_backward]:
                        if "/-" not in leanfile[step_backward]:
                            in_block_comment_inner = True
                        block_end = step_backward
                    elif in_block_comment_inner:
                        if "/-" in leanfile[step_backward]:
                            in_block_comment_inner = False
                            block_end = step_backward
                    else:
                        break

                # If we're looking at a theorem, include it; otherwise, skip over to the next block
                if current_keyword in ["theorem", "problem", "lemma"] and current_name != "":
                    if leanfile[block_end-1].strip().startswith("#"):
                        block_end -= 1
                    segments[current_name] = ("\n".join(leanfile[block_start:block_end]).strip(), range(block_start, block_end))

                if len(leanfile[start_next_search_loc].split(" ")) > 1:
                    current_name = leanfile[start_next_search_loc].split(" ")[1]
                else:
                    current_name = ""
                current_keyword = first_word

                block_start = block_end
                block_end = start_next_search_loc
            block_end += 1

    # If the last block in the file is a theorem, include that too
    if current_keyword in ["theorem", "problem", "lemma"] and current_name != "":
        if leanfile[block_end-1].strip().startswith("#"):
            block_end -= 1
        segments[current_name] = ("\n".join(leanfile[block_start:block_end]).strip(), range(block_start, block_end))

    return segments


# Requires: a syntactically-correct Lean theorem in plaintext
# Returns: the length of a theorem (number of tactics), excluding comments and annotations, and accounting for multiple tactics on a single line of text
def get_length(theorem):
    total = 0
    in_block_comment = False
    has_started_proof = False
    for line in theorem.split("\n"):
        if ":=" in line:
            has_started_proof = True
        if "/-" in line:
            in_block_comment = True
        if in_block_comment:
            if "-/" in line:
                in_block_comment = False
            continue
        elif (not (line.strip().startswith("--") or line.strip().startswith("@") or line.strip() == "")) and has_started_proof:
            total += len(line.split(";"))
    return total


# Requires: a list of pairs of valid theorems (original, improved) without context
# Returns: the theorems that have been significantly improved in length
# TODO: any other metrics to ensure good quality?
def filter_theorems(theorems):
    ret = []
    for original, improved in theorems:
        if get_length(improved) < get_length(original):
            ret.append((original, improved))
    return ret


# Requires: a keyword to search commit messages in the Mathlib repository for
# Returns: a generator of tuples (commit hash, commit message) that contain the keyword
def get_mathlib_commits(search_keyword="golf"):
    commit_log, error = subprocess.Popen(["git", "log", "--all", "-i", f"--grep={search_keyword}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_PATH).communicate()
    if error:
        print("Error in getting commit log:", error)
        return
    hashes = []
    for line in commit_log.decode("utf-8").split("\n"):
        if line.startswith("commit"):
            hashes.append(line.split(" ")[1])
    for hash in hashes:
        changes, error = subprocess.Popen(["git", "show", hash + "^", hash, "--name-only"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_PATH).communicate()
        if error:
            print("Error in getting commit changes TATE PLS FIX")
            continue
        for line in changes.decode("utf-8").split("\n"):
            if line.startswith("Mathlib"):
                yield hash, line


# Requires: True
# Returns: a generator of tuples (original theorem, improved theorem) that have been significantly length-optimized by human mathematicians
def get_improved_theorems():
    for hash, file in get_mathlib_commits("golf"):
        for original, improved in filter_theorems(get_improvement_candidates(hash, file)):
            yield original, improved


# Requires: True
# Returns: None
# Saves all significantly improved theorems to a CSV file
def save_to_csv():
    with open("human_optimized_theorems.csv", "w") as f:
        f.write("Original,Improved\n")
        for original, improved in get_improved_theorems():
            f.write("\"" + original.replace("\"", "\"\"") + "\",\"" + improved.replace("\"", "\"\"") + "\"\n")

if __name__ == '__main__':
    save_to_csv()
    # for original, improved in get_improved_theorems():
    #     print("Original:\n\n", original)
    #     print("\n\nImproved:\n\n", improved)
    #     print("\n\n-------------------\n\n")
    #     input()
