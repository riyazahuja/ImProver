import re
import pandas as pd


def remetrize_mod(text: str):
    text = text.strip()
    pattern = r":=\s*by"
    match = re.search(pattern, text)
    start = match.end()
    text = text[start:]

    split = text.split("\n")
    tactics = []
    pattern = r"(?<!<);(?!=>)"
    for line in split:
        semicolon_splits = re.split(pattern, line)
        tactics.extend(semicolon_splits)
    # tactics = split
    tactics = [t for t in tactics if t != ""]

    indent = "  "
    have_idxs = [
        i
        for i, tac in enumerate(tactics)
        if tac.startswith(indent) and tac.strip().startswith("have")
    ]
    have_groups = []
    for i in have_idxs:
        end_decl = [k for k in range(i, len(tactics)) if ":" in tactics[k]]
        if len(end_decl) == 0:
            continue
        end_decl = end_decl[0]

        end_end_decl = [k for k in range(i, len(tactics)) if ":=" in tactics[k]]
        if len(end_end_decl) == 0:
            continue
        end_end_decl = end_end_decl[0]

        have_groups.append((i, end_end_decl))  # inclusive

    if len(have_groups) == 0:
        return 0  # len(tactics)

    # def get_semis(end_decl, end):
    #     return sum(tactics[i].count(";") for i in range(end_decl + 1, end)) - sum(
    #         tactics[i].count("<;>") for i in range(end_decl + 1, end)
    #     )

    # have_grp_max = max([b - end + get_semis(end, b) for a, end, b in have_groups])
    # print(have_groups)
    have_grp_total = sum([b - a + 1 for a, b in have_groups])
    rest = have_grp_total / len(tactics)

    score = rest
    return score


def remetrize_mod(text: str):
    text = text.strip()
    pattern = r":=\s*by"
    match = re.search(pattern, text)
    start = match.end()
    text = text[start:]

    split = text.split("\n")
    tactics = []
    pattern = r"(?<!<);(?!=>)"
    for line in split:
        semicolon_splits = re.split(pattern, line)
        tactics.extend(semicolon_splits)
    # tactics = split
    tactics = [t for t in tactics if t != ""]

    indent = "  "
    have_idxs = [
        i
        for i, tac in enumerate(tactics)
        if tac.startswith(indent) and tac.strip().startswith("have")
    ]
    have_groups = []
    for i in have_idxs:
        end_decl = [
            k
            for k in range(i, len(tactics))
            if re.search(r":=\s*by", tactics[k]) is not None
        ]
        if len(end_decl) == 0:
            continue
        end_decl = end_decl[0]

        have_groups.append((i, end_decl))  # inclusive

    if len(have_groups) == 0:
        return 0  # len(tactics)

    # def get_semis(end_decl, end):
    #     return sum(tactics[i].count(";") for i in range(end_decl + 1, end)) - sum(
    #         tactics[i].count("<;>") for i in range(end_decl + 1, end)
    #     )

    # have_grp_max = max([b - end + get_semis(end, b) for a, end, b in have_groups])
    # print(have_groups)
    have_grp_total = sum([b - a + 1 for a, b in have_groups])
    rest = have_grp_total / len(tactics)

    score = rest
    return score


# Load the csv file
file_path = "benchmark/data/MAI/modularity_annotation_ablation2.csv"  # Placeholder for actual file path
output_path = "benchmark/data/MAI/better_mod/modularity_annotation_ablation2.csv"

df = pd.read_csv(file_path)


# Function to update scores and delta
def update_scores_and_delta(row):
    og_raw = str(row["og_raw"])
    new_raw = str(row["new_raw"])
    new_correct = str(row["new_correct"])

    # Recalculate scores using the rescore function
    row["og_score"] = remetrize_mod(og_raw)
    row["new_score"] = remetrize_mod(new_raw)

    # Recalculate delta as the percent change
    # if row["og_score"] != 0:  # Avoid division by zero
    #     row["delta"] = (row["new_score"] - row["og_score"]) / row["og_score"]
    # else:
    #     row["delta"] = None  # Handle zero original score case

    # if row["og_score"] == 0 and row["new_score"] == 0:
    #     row["delta"] = 0  # Handle zero original score case

    row["delta"] = row["new_score"] - row["og_score"]
    if og_raw.strip() == new_raw.strip() or new_correct == "False":
        row["delta"] = None

    return row


def update__delta(row):
    og_raw = str(row["og_raw"])
    new_raw = str(row["new_raw"])
    new_correct = str(row["new_correct"])

    if og_raw.strip() == new_raw.strip() or new_correct == "False":
        row["delta"] = None

    return row


# Apply the function to each row
df_updated = df.apply(update_scores_and_delta, axis=1)

# Save the updated dataframe to a new CSV file

df_updated.to_csv(output_path, index=False)


# string = """
# /-- Another version of `simply_connected_iff_paths_homotopic` -/
# theorem simply_connected_iff_paths_homotopic' {Y : Type*} [TopologicalSpace Y] :
#     SimplyConnectedSpace Y ↔
#       PathConnectedSpace Y ∧ ∀ {x y : Y} (p₁ p₂ : Path x y), Path.Homotopic p₁ p₂   := by
#   convert simply_connected_iff_paths_homotopic (Y := Y)
#   have h1 : ∀ x y : Y, Subsingleton (Path.Homotopic.Quotient x y) ↔ ∀ {p₁ p₂ : Path x y}, Path.Homotopic p₁ p₂ := by
#     intro x y
#     simp [Path.Homotopic.Quotient, Setoid.eq_top_iff]
#     rfl
#   simp only [h1]
# """


# print(remetrize_mod(string))
