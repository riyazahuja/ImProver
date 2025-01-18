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
    return error_branches
    # perhaps need also information of the original node before the spawn/bifurcation, and node numbers as well (for future insertion)
    print([[tac.tactic for tac in branch] for branch in error_branches])


def replace_and_run(branches, thm: AnnotatedTheorem):
    # convert branches into syntax ranges:

    syntax_ranges = [(branch[0].start, branch[-1].end) for branch in branches]
    syntax_ranges = [
        ((start[0] - 1, start[1]), (end[0] - 1, end[1])) for start, end in syntax_ranges
    ]

    command = "extract_goal; sorry"
    contents = thm.pretty_print

    # conver syntax ranges into offset ranges:
    lines = contents.splitlines(True)
    offset_acc = [0]
    for l in lines:
        offset_acc.append(offset_acc[-1] + len(l))

    offset_ranges = []
    for start_pos, end_pos in syntax_ranges:
        s_off = offset_acc[start_pos[0]] + start_pos[1]
        e_off = offset_acc[end_pos[0]] + end_pos[1]
        offset_ranges.append((s_off, e_off))

    # print(offset_ranges)
    # print("\n[end]\n".join([contents[start:end] for (start, end) in offset_ranges]))

    thm_syntax_range = ((thm.start[0] - 1, thm.start[1]), (thm.end[0] - 1, thm.end[1]))

    text_list = list(contents)
    for start, end in sorted(offset_ranges, reverse=True):
        text_list[start:end] = command
    new_contents = "".join(text_list)

    theorem_start = offset_acc[thm_syntax_range[0][0]] + thm_syntax_range[0][1]
    theorem_end = offset_acc[thm_syntax_range[1][0]] + thm_syntax_range[1][1]

    adjusted_end = theorem_end
    for start_off, end_off in offset_ranges:
        if end_off <= theorem_start or start_off >= theorem_end:
            continue
        overlap_length = min(end_off, theorem_end) - max(start_off, theorem_start)
        if overlap_length > 0:
            replaced_diff = len(command) - (end_off - start_off)
            adjusted_end += replaced_diff

    theorem_slice = new_contents[theorem_start:adjusted_end]
    return theorem_slice


def make_theorem(thm_text: str, thm: AnnotatedTheorem) -> AnnotatedTheorem:

    src = thm.src
    path = thm.leanFile
    text = thm_text
    # print(thm.context)
    # print(thm.proof)
    # og_thm_start = len(f"{thm.context}\n\n{thm.decl} := by\n".splitlines())
    # og_thm_end = len(text.splitlines())

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, f".cache", src, os.path.dirname(path))
    project_path = thm.project_path

    # print(f"Cache Path: {cache_path}")
    file_name = os.path.basename(path).replace(".lean", "")
    binary_path = os.path.join(cache_path, file_name + ".o")
    # print(f"Binary Path: {binary_path}")

    cmd_text = thm.headerless_context + "\n\n" + text

    cmd_text = cmd_text.replace("\n", "\\n")  # .replace('"', '\\"')
    # print(thm.headerless_context)
    # print("##############")
    infile = f'{{"unpickleEnvFrom": "{binary_path}"}}\n\n{{"cmd": "{cmd_text}", "allTactics": true, "theorems": true, "env": 0}}'

    temp = tempfile.NamedTemporaryFile(suffix=".in", dir=root_path)
    with open(temp.name, "w") as f:
        f.write(infile)

    output = subprocess.run(
        [f"lake env repl/.lake/build/bin/repl < {temp.name}"],
        shell=True,
        text=True,
        capture_output=True,
        cwd=root_path,
    )
    # print(cmd_text)
    # print(output.stdout)
    out = "\n".join(output.stdout.splitlines()[2:])
    data = json.loads(out)
    return coerce_repl(data, thm, thm.context + "\n" + thm_text)


def make_empty_theorems(thm: AnnotatedTheorem) -> Tuple[Message, List[Theorem]]:

    infos = [msg for msg in thm.messages if msg.severity == "info"]
    out = []
    for info in infos:
        thm_text = info.content

        decl = thm_text.split(":=")[0]
        context = parseTheorem(thm, context=True) + "\n" + thm_text
        headerless_context = (
            parseTheorem(thm, headerless_context=True) + "\n" + thm_text
        )
        ProofStep.update_forward_refs()

        empty_theorem = Theorem(
            decl=decl,
            proof=[ProofStep(tactic="sorry")],
            declID=str(uuid.uuid4()),
            src=thm.src,
            leanFile=thm.leanFile,
            context=context,
            headerless_context=headerless_context,
            project_path=thm.project_path,
        )

        out.append((info, empty_theorem))
    return out


#: List[Tuple[Message, AnnotatedTheorem]]
def insert_theorems(thm: AnnotatedTheorem, subtheorems):
    contents = thm.pretty_print
    subtheorems.sort(key=lambda x: x[0].start[0], reverse=True)

    for subtheorem in subtheorems:
        # insertion_range = (info.start, info.end)

        subtheorem_proof_syntax_range = (subtheorem.start, subtheorem.end)
        subtheorem_text = parseTheorem(subtheorem, context=True)

        lines = subtheorem_text.splitlines(True)
        offset_acc = [0]
        for l in lines:
            offset_acc.append(offset_acc[-1] + len(l))

        offset_ranges = []

        start_offset = offset_acc[subtheorem.start[0]] + subtheorem.start[1]
        end_offset = offset_acc[subtheorem.end[0]] + subtheorem.end[1]

        # Retrieve the subtheorem text within the syntax range
        subtheorem_proof_text = subtheorem_text[start_offset:end_offset]

        print("WAHOO!!!!!!")
        print(subtheorem_subset)
        print(len(subtheorem_text))
        # print("nexts")
        # print(parseTheorem(subtheorem, context=True))
        print()
        print(
            "\n".join(
                [f"{i}:\t{val}" for i, val in enumerate(subtheorem_text.splitlines())]
            )
        )
        print(f"==[{subtheorem.start} -> {subtheorem.end}]==")
        print(start_offset, end_offset)


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
    err_branches = extract_subtheorem(thm)
    print("++++++++")
    thm_text = replace_and_run(err_branches, thm)
    print(f"==[{thm_text}]")
    new_thm = make_theorem(thm_text, thm)
    print("+++++++")
    print(parseTheorem(new_thm, context=True, annotation=False))

    emp_thms = make_empty_theorems(new_thm)
    print("\n\n".join([f"info : {info}" for info, empty in emp_thms]))

    insert_theorems(thm, [thm])
