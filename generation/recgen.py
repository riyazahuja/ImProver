import sys
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))
from evaluate.eval import *
from evaluate.metrics import *
from models.structures import *
from models.prompt import *
from evaluate.build_prooftree import *
import pandas as pd
import time
from models.rag import *
import itertools
from tqdm import tqdm
from multiprocessing import cpu_count


# need a function that takes the proof tree of a AnnotatedTheorem, and notes all mod edges (and their descendant branches)
# Then cross-checks these branches to see if there is any error message in that specific branch (going from leaves upwards)
# if so, then replace branch with a "sorry" and a "extract_goals" and coerce into a theorem. then annotate the theorem, and extract the "info"
# messages into new (empty) (annotated)Theorems, and also link them to the specific modular edges/nodes
# return a mapping of modular edges/branches to annotatedTheorems


def get_branch(tree: nx.DiGraph, node):
    branch = []
    stack = [node]
    while stack:
        current = stack.pop()
        branch.append(current)
        children = list(tree.successors(current))
        stack.extend(children)
    return branch


def to_offset(pos, thm: AnnotatedTheorem):
    line, col = pos
    max_pf = 0
    max_msg = 0
    if len(thm.proof) > 0:
        max_pf = max([max(tac.start[1], tac.end[1]) for tac in thm.proof])
    if len(thm.messages) > 0:
        max_msg = max([max(msg.start[1], msg.end[1]) for msg in thm.messages])

    max_cols = max(max_pf, max_msg)

    return line * (max_cols + 1) + col


def extract_subtheorem(thm: AnnotatedTheorem):
    print("extracting!")
    PT, positions, labels = getProofTree(thm)
    for edge in PT.edges():
        print(
            f"{edge}: spawned {PT.edges[edge].get('spawned',False)}, bifurcation {PT.edges[edge].get('bifurcation',False)}"
        )
    mod_edges = list(
        set([(u, v) for (u, v, s) in PT.edges.data("spawned", default=False) if s])
        | set(
            [(u, v) for (u, v, b) in PT.edges.data("bifurcation", default=False) if b]
        )
    )

    print(mod_edges)
    mod_nodes = [v for (u, v) in mod_edges]
    mod_branches = [get_branch(PT, v) for v in mod_nodes]
    mod_branch_tacs = []
    branch_mapping = {}

    for i, branch in enumerate(mod_branches):
        idxs = [PT.nodes[node]["data"] for node in branch]
        mod_branch_tacs.append([thm.proof[idx] for idx in idxs])
        for idx in idxs:
            branch_mapping[idx] = i

    # mod_branch_tacs = [
    #     [PT.nodes[node]["data"] for node in branch] for branch in mod_branches
    # ]
    used = set([PT.nodes[node]["data"] for node in PT.nodes])
    lru = 0
    for i, tactic in enumerate(thm.proof):
        if i in used:
            lru = i
        else:
            branch_idx = branch_mapping[lru]
            mod_branch_tacs[branch_idx].append(tactic)

    print([[tac.tactic for tac in branch] for branch in mod_branch_tacs])

    error_branches = []
    for branch in mod_branch_tacs:

        msg_ranges = [
            (to_offset(msg.start, thm), to_offset(msg.end, thm)) for msg in thm.messages
        ]

        tactic_ranges = [
            (to_offset(tactic.start, thm), to_offset(tactic.end, thm))
            for tactic in branch
        ]
        if any(
            msg_start <= tactic_end and tactic_start <= msg_end
            for (msg_start, msg_end) in msg_ranges
            for (tactic_start, tactic_end) in tactic_ranges
        ):
            error_branches.append(branch)

    # perhaps need also information of the original node before the spawn/bifurcation, and node numbers as well (for future insertion)
    print([[tac.tactic for tac in branch] for branch in error_branches])


if __name__ == "__main__":
    repo = getRepo("Tests", "configs/config_MIL.json")
    files = {file.file_path: file for file in repo.files}
    fs = [
        files[name]
        for name in files.keys()
        if "Solution" in name and "C04" in name and "S01" in name
    ]

    f = fs[0]
    thms = f.theorems
    thm = thms[0]

    ProofStep.update_forward_refs()
    proof = [
        ProofStep(tactic="rintro x (⟨xs, xt⟩ | ⟨xs, xu⟩)"),
        ProofStep(tactic=". use xs"),
        ProofStep(tactic="  right"),
        ProofStep(tactic="  exact xt"),
        ProofStep(tactic="use 3"),
        ProofStep(tactic="right"),
        ProofStep(tactic="exact xu"),
    ]

    thm_base = Theorem(
        decl=thm.decl,
        proof=proof,
        declID=thm.declID,
        src=thm.src,
        leanFile=thm.leanFile,
        context=thm.context,
        headerless_context=thm.headerless_context,
        project_path=thm.project_path,
    )
    thm = annotateTheorem(thm_base, force=True)
    print(len(thm.proof))
    print(parseTheorem(thm, context=False))
    for i, tac in enumerate(thm.proof):
        print(f"{i} : {tac.tactic}")

    save_tree(
        *getProofTree(thm, visualize=False),
        save_path=f".trees/MIL/new2.png",
        show_mod=True,
    )
    extract_subtheorem(thm)
