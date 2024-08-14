from __future__ import annotations
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from typing import List, Union, Optional, Tuple
import json
import tempfile
import subprocess
from textwrap import indent
import re

# NOTE position encodings have top left as line 1, column 0
# -> we restandardize to line 1 column 1. makes more sense that way


class ProofStep(BaseModel):
    tactic: Union[str, Theorem] = Field(
        description="One line/tactic in a tactic proof (str) or a subtheorem/sublemma/subproof"
    )


class Theorem(BaseModel):
    decl: Optional[str] = Field(
        description="Theorem declaration. Optional argument, if not provided, will default to being an implicit case statement using dot notation.",
        default=None,
    )
    proof: List[ProofStep] = Field(
        ..., description="Sequence of proofsteps for full proof of theorem."
    )
    declID: str = Field(description="Unique theorem declaration ID")
    src: str = Field(description="Source repo of the theorem")
    leanFile: str = Field(description="Lean file in which theorem is located")
    context: str = Field(
        description="Context of the theorem (i.e. file contents up to decl)"
    )
    project_path: str = Field(description="Local path to src repo contents")


class File(BaseModel):
    src: str = Field(description="File source repo")
    file_name: str = Field(description="File Name")
    file_path: str = Field(description="File path (stem) relative to src repo root")
    file_type: str = Field(description="File type")
    contents: str = Field(description="File contents")
    project_path: str = Field(description="Local path to src repo contents")


class AnnotatedProofStep(BaseModel):
    prevState: List[str] = Field(
        description="Pretty printed tactic st ate before the tactic invocation"
    )
    tactic: str = Field(description="One line/tactic in a tactic proof.")
    nextState: List[str] = Field(
        description="Pretty printed tactic state after the tactic invocation"
    )
    srcUpToTactic: str = Field(
        description="Source code from file start to current tactic"
    )
    declUpToTactic: str = Field(
        description="Source code from theorem declaration to current tactic"
    )
    start: Tuple[Optional[int], Optional[int]] = Field(
        description="start coordinates from source file as (row,column)"
    )
    end: Tuple[Optional[int], Optional[int]] = Field(
        description="end coordinates from source file as (row,column)"
    )


class Message(BaseModel):
    severity: str = Field(description="Message severity")
    start: Tuple[Optional[int], Optional[int]] = Field(
        description="start coordinates from source file as (row,column)"
    )
    end: Tuple[Optional[int], Optional[int]] = Field(
        description="end coordinates from source file as (row,column)"
    )
    message_src: Optional[str] = Field(
        description="equivalent to source_contents[start:end]"
    )
    content: str = Field(description="Message contents")


class AnnotatedTheorem(BaseModel):
    decl: str = Field(description="Theorem declaration")
    declID: str = Field(description="Unique theorem declaration ID")
    src: str = Field(description="Source repo of the theorem")
    leanFile: str = Field(description="Lean file in which theorem is located")
    context: str = Field(
        description="Context of the theorem (i.e. file contents up to decl)"
    )
    proof: List[AnnotatedProofStep] = Field(
        ..., description="Sequence of annotated proofsteps for full proof of theorem."
    )
    project_path: str = Field(description="Local path to src repo contents")
    messages: List[Message] = Field(..., description="Messages from the lean server")
    pretty_print: str = Field(description="Content of theorem src file.")
    proof_tree: List[Tuple[str, List[int], List[int]]] = Field(
        description="data for efficient proof tree construction"
    )


class AnnotatedFile(BaseModel):
    src: str = Field(description="File source repo")
    file_name: str = Field(description="File Name")
    file_path: str = Field(description="File path (stem) relative to src repo root")
    file_type: str = Field(description="File type")
    contents: str = Field(description="File contents")
    theorems: List[AnnotatedTheorem] = Field(
        ..., description="List of all theorems in a file"
    )
    project_path: str = Field(description="Local path to src repo contents")


class Repo(BaseModel):
    type: str = Field(description="Repository type (git or local)")
    url: Optional[str] = Field(description="if git repo, github url")  # if remote
    commit: Optional[str] = Field(
        description="if git repo, github commit SHA code"
    )  # if remote
    path: Optional[str] = Field(description="if local repo, path")  # if local
    version: str = Field(description="Lean version")
    name: str = Field(description="Repo name")
    dependencies: List[Repo] = Field(description="Repository dependencies")
    files: List[Union[AnnotatedFile, File]] = Field(description="files in repository")
    project_path: str = Field(description="Local path to src repo contents")


def getMessages(start, end, msgs, contents: str):
    thm_start_row = start.get("line", None)
    thm_start_col = start.get("col", None)
    if end is not None:
        thm_end_row = end.get("line", None)
        thm_end_col = end.get("col", None)
    else:
        thm_end_row = 0
        thm_end_col = 0

    def msg_in_thm(msg):
        start_row = (
            msg.get("pos", {}).get("line", None)
            if msg.get("pos", None) is not None
            else None
        )
        end_row = (
            msg.get("endPos", {}).get("line", None)
            if msg.get("endPos", None) is not None
            else None
        )

        if start_row is None:
            return False
        if (
            thm_start_row <= start_row
            and end is not None
            and (end_row is None or thm_end_row >= end_row)
        ):
            return True
        elif thm_start_row <= start_row and end is None:
            return True
        else:
            return False

    msgs2 = [msg for msg in msgs if msg_in_thm(msg)]
    msgs = msgs2
    return [getMessage(msg, contents) for msg in msgs]


def getMessage(msg, contents: str):
    severity = msg["severity"]

    start_row = (
        msg.get("pos", {}).get("line", None)
        if msg.get("pos", None) is not None
        else None
    )
    start_col = (
        msg.get("pos", {}).get("column", None)
        if msg.get("pos", None) is not None
        else None
    )
    end_row = (
        msg.get("endPos", {}).get("line", None)
        if msg.get("endPos", None) is not None
        else None
    )
    end_col = (
        msg.get("endPos", {}).get("column", None)
        if msg.get("endPos", None) is not None
        else None
    )

    message_src = None
    if start_col is None:
        start_col = 1
        # message_src = ''
    if end_col is None and end_row is not None:
        end_col = len(end_row)
        # message_src = ''

    if end_row is None:

        try:
            lines = [contents.splitlines()[start_row - 1]]
        except:
            lines = []

    else:
        lines = contents.splitlines()[start_row - 1 : end_row]
    trimmed_lines = []

    for line in lines:
        if line == lines[0]:
            if len(lines) == 1:
                trimmed_lines.append(line)
            else:
                trimmed_lines.append(line[start_col:])
        elif line == lines[-1]:
            if end_col is None:
                trimmed_lines.append(line)
            else:
                trimmed_lines.append(line[: end_col + 1])

        else:
            trimmed_lines.append(line)
    content = msg["data"]

    # if message_src is not None:
    message_src = "\n".join(trimmed_lines)

    return Message(
        severity=severity,
        start=(start_row, start_col),
        end=(end_row, end_col),
        message_src=message_src,
        content=content,
    )


def getTheorems(
    data, src, path, project_path, contents, until_end=False
) -> List[AnnotatedTheorem]:
    temp = {}
    msgs = data["messages"]
    trees = data["proofTrees"]

    def remove_dupes(tacs):
        output = []
        for i, step in enumerate(tacs):
            dupe = any(
                [
                    step["startPos"] == other["startPos"]
                    and step["endPos"] == other["endPos"]
                    for j, other in enumerate(tacs)
                    if j < i
                ]
            )
            if not dupe:
                output.append(step)
        return output

    data = remove_dupes(data["tactics"])

    all_tacs = [re.sub(r"\s", "", step["tactic"]) for step in data]

    tactic_start_line = data[0]["startPos"]["line"] if len(data) != 0 else None
    for step in data:

        ps = AnnotatedProofStep(
            prevState=step["prevState"],
            tactic=step["tactic"],
            nextState=step["nextState"],
            srcUpToTactic=step["srcUpToTactic"],
            declUpToTactic=step["declUpToTactic"],
            start=(
                step["startPos"].get("line", None),
                step["startPos"].get("column", None),
            ),
            end=(step["endPos"].get("line", None), step["endPos"].get("column", None)),
        )

        def elim_by(text):
            return re.sub(r"\s*:=\s*by\s*$", "", text, flags=re.M)

        decl = elim_by(step["decl"])
        declID = step["declId"]
        thm_start, thm_end = step["thm_startPos"], step["thm_endPos"]
        if until_end:
            thm_end = None

        messages = getMessages(thm_start, thm_end, msgs, contents)
        messages.reverse()
        lines_src = step["srcUpToTactic"].split("\n")
        decl_lines = decl.split("\n")

        maybe_context = "\n".join(lines_src[: -len(decl_lines) - 1]).strip()

        pp = contents

        PT = trees.get(declID, None)
        if PT is not None:
            PT = [
                (node["tactic"], node["children"], node["spawned_children"])
                for node in PT
            ]

        if declID not in temp.keys():
            temp[declID] = {
                "proof": [ps],
                "decl": decl,
                "context": maybe_context,
                "messages": messages,
                "pretty_print": pp,
                "proof_tree": PT,
            }
        else:
            curr_proof = temp[declID]["proof"]
            curr_decl = temp[declID]["decl"]
            curr_ctxt = temp[declID]["context"]
            curr_msgs = temp[declID]["messages"]
            curr_pp = temp[declID]["pretty_print"]
            curr_pt = temp[declID]["proof_tree"]
            curr_proof.append(ps)
            temp[declID] = {
                "proof": curr_proof,
                "decl": curr_decl,
                "context": curr_ctxt,
                "messages": curr_msgs,
                "pretty_print": curr_pp,
                "proof_tree": curr_pt,
            }

    result = {}
    for ID, value in temp.items():
        result[ID] = AnnotatedTheorem(
            leanFile=path,
            src=src,
            decl=value["decl"],
            declID=ID,
            proof=value["proof"],
            context=value["context"],
            project_path=project_path,
            messages=value["messages"],
            pretty_print=value["pretty_print"],
            proof_tree=value["proof_tree"],
        )
    return [v for _, v in result.items()]


def get_stem(path):

    if path[-5:] == ".lean":
        return path[:-5]
    elif path[-5:] == ".json":
        return path[:-5]
    return path


def getAnnotatedFile(src, path, project_path, until_end=False):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, f".cache/{src}")

    with open(os.path.join(project_path, path), "r") as f:
        contents = f.read()

    stem, ftype = os.path.splitext(path)
    with open(os.path.join(cache_path, stem + ".json"), "r") as f:
        data = json.load(f)

    theorems = getTheorems(data, src, path, project_path, contents, until_end=until_end)

    return AnnotatedFile(
        src=src,
        file_name=os.path.basename(path),
        contents=contents,
        theorems=theorems,
        file_path=path,
        file_type=ftype,
        project_path=project_path,
    )


def getFile(src, path, project_path, annotate=True, force=False):

    if annotate:
        try:
            return getAnnotatedFile(src, path, project_path)
        except FileNotFoundError as e:
            if force:
                raise e
            else:
                pass
    stem, ftype = os.path.splitext(path)
    with open(os.path.join(project_path, path), "r") as f:
        content = f.read()

    return File(
        src=src,
        file_name=os.path.basename(path),
        file_path=path,
        file_type=ftype,
        contents=content,
        project_path=project_path,
    )


# TODO IMPLEMENT LOCAL CONFIGS TO ADD LOCAL CONFIGS TO GETREPO
def getRepoDirect(repo_data, annotate=True, force=False, recursive=True):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_path = ""
    if "path" in repo_data.keys():
        local = True
    elif "repo" in repo_data.keys():
        local = False
    else:
        raise ValueError(f"Invalid Config:\n {repo_data}")

    version = repo_data.get("lean", "")
    name = repo_data.get("name", "")

    if local:
        path = repo_data.get("path", "")
        url = None
        commit = None
        project_path = path
    else:
        url = repo_data.get("repo", "")
        commit = repo_data.get("commit", "")
        path = None
        project_path = os.path.join(root_path, ".lake", "packages", name)

    # depedencies:
    if recursive:
        manifest_path = os.path.join(project_path, "lake-manifest.json")
        toolchain_path = os.path.join(project_path, "lean-toolchain")

        with open(manifest_path, "r") as f:
            manifest_data = json.load(f).get("packages", [])
        with open(toolchain_path, "r") as f:
            pack_version = f.read()
        dependencies = []
        for package in manifest_data:
            dependency_names = [item.name for item in dependencies]
            if package["name"] in dependency_names:
                continue
            if package["type"] == "git":
                subdata = {
                    "repo": package["url"],
                    "commit": package["rev"],
                    "lean": pack_version,
                    "name": package["name"],
                }
            else:
                subdata = {
                    "path": package["path"],
                    "lean": pack_version,
                    "name": package["name"],
                }
            dependencies.append(
                getRepoDirect(subdata, annotate=False, force=False, recursive=False)
            )
    else:
        dependencies = []
    # Files:
    repo_files = []
    files_list = []
    ignore = [".lake"]

    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in ignore]
        for file in files:
            fp = os.path.join(root, file)
            if fp.endswith(".lean") and name not in ignore:
                files_list.append((name, os.path.relpath(fp, project_path)))

    # with tqdm(total=len(files_list)) as pbar:
    for name, rel in files_list:
        repo_files.append(
            getFile(name, rel, project_path, annotate=annotate, force=force)
        )
        # pbar.update(1)

    if local:
        return Repo(
            type="local",
            path=path,
            version=version,
            name=name,
            dependencies=dependencies,
            files=repo_files,
            project_path=project_path,
        )
    else:
        return Repo(
            type="git",
            url=url,
            commit=commit,
            version=version,
            name=name,
            dependencies=dependencies,
            files=repo_files,
            project_path=project_path,
        )


def getRepo(src, config=None, annotate=True, force=False):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config is None:
        config_files = []
        for root, _, files in os.walk(os.path.join(root_path, "configs")):
            for file in files:
                path = os.path.join(root, file)
                if path.endswith(".json"):
                    config_files.append(path)
        for path in config_files:
            with open(path, "r") as f:
                data = json.load(f)
            if src in [item.get("name", "") for item in data]:
                config = os.path.relpath(path, start=root_path)
                break
        if config is None:
            raise ValueError(f"{src} config not found")

    config_path = os.path.join(root_path, config)
    with open(config_path, "r") as f:
        data = json.load(f)
    data = [
        item
        for item in data
        if src.replace("«", "").replace("»", "")
        == item.get("name", "").replace("«", "").replace("»", "")
    ]
    if len(data) == 0:
        raise ValueError(f"{src} not in config file {config}")
    repo_data = data[0]
    repo_data["name"] = repo_data["name"].replace("«", "").replace("»", "")

    return getRepoDirect(repo_data, annotate=annotate, force=force)


def get_tactic_str(step: AnnotatedProofStep, content: str, annotation=False):
    content = content.splitlines()


def parseAnnotatedTheorem(thm: AnnotatedTheorem, context=True, annotation=False):
    context_str = thm.context
    decl = thm.decl
    proof = thm.proof
    tactics_start_pos = proof[0].start
    thm_end_pos = proof[-1].end
    contents = thm.pretty_print.splitlines()
    if thm_end_pos is None:
        thm_end_pos = len(contents) - 1

    if annotation:
        tactic_line_positions = [(step.start[0], step.end[0], step) for step in proof]
        tactic_line_positions = [
            (
                pos[0] - 1 if pos[0] is not None else None,
                pos[1] - 1 if pos[1] is not None else None,
                pos[2],
            )
            for pos in tactic_line_positions
        ]
        tactic_line_positions = [
            pos
            for pos in tactic_line_positions
            if pos[1] is not None and pos[0] is not None
        ]
        tactic_line_positions.sort(key=lambda x: x[0])

        # state_text to print after line n:
        state_text_after_line = {}
        for pos in tactic_line_positions:
            line_after = pos[1]
            text = pos[2].nextState
            text = (
                "/-\n" + "\n".join(text) + "\n-/"
                if len(text) != 0
                else "/-\nGoals Solved!\n-/"
            )
            # Add indentation stuff
            # if line_after not in state_text_after_line.keys():
            state_text_after_line[line_after] = text
        proof_text = []
        for curr_line in range(
            tactics_start_pos[0] - 1, min(thm_end_pos[0], len(contents) - 1)
        ):
            try:
                line_text = contents[curr_line]
            except:
                enum_cont = "\n".join(
                    [f"{i}\t|{contents[i]}" for i in range(len(contents))]
                )
                print(
                    f"len contents: {len(contents)}\ncurr_line = {curr_line}\nstart:{tactics_start_pos[0] - 1}\nend:{thm_end_pos[0]-1}\nthm:{decl}\npp:{enum_cont}"
                )
                raise KeyError("")

            try:
                line_one = line_text.splitlines()[0]
                indent_cnt = len(line_one) - len(line_one.lstrip(" "))
            except:
                indent_cnt = 0

            proof_text.append(line_text)
            if curr_line in state_text_after_line.keys():
                state_text = indent(state_text_after_line[curr_line], indent_cnt * " ")
                proof_text.append(state_text)
        proof_text = "\n".join(proof_text)

    else:
        if thm_end_pos is not None and thm_end_pos[0] is not None:
            proof_text = "\n".join(contents[tactics_start_pos[0] - 1 : thm_end_pos[0]])
        else:
            proof_text = "\n".join(contents[tactics_start_pos[0] - 1 :])
    return f"{context_str if context else ''}\n{decl} := by\n{proof_text}"
    # return f"{context_str if context else ''}\n{proof_text}"


def elim_overlap(pf: List[AnnotatedProofStep]):
    def pos_le(a, b):
        a_line, a_col = a
        b_line, b_col = b

        if a_line < b_line:
            return True
        elif a_line == b_line and a_col <= b_col:
            return True
        else:
            return False

    def pos_max(a, b):
        le = pos_le(a, b)
        if le:
            return b
        else:
            return a

    ptr = (0, 0)
    output = []
    for step in pf:
        start = step.start
        end = step.end
        if pos_le(start, ptr) and pos_le(end, ptr):
            # this is inside a have
            pass
        else:
            ptr = pos_max(start, end)
            output.append(step)
    return output


# def annotate(step: AnnotatedProofStep, prev_goal=False):
#     prev = step.prevState
#     tactic = step.tactic
#     next = step.nextState

#     def pp_state(state):
#         if state == []:
#             return "No Goals Left!"
#         return "\n".join(state)

#     prev_text = f"""
# /-
# {pp_state(prev)}
# -/
# """
#     text = f"""
# {tactic}
# /-
# {pp_state(next)}
# -/
# """
#     if prev_goal:
#         return indent(prev_text + text, "  ", lambda x: True)
#     else:
#         return indent(text, "  ", lambda x: True)


def parse_proof(thm, indent=1, dot=False):
    output = ""
    spaces = "  "
    proof = thm.proof
    for step in proof:
        content = step.tactic
        if type(content) == str:
            output += indent * spaces + content + "\n"
            # single tactic
        else:
            hasDecl = content.decl is not None
            if hasDecl:
                output += (
                    indent * spaces
                    + content.decl
                    + ("" if "." == content.decl.strip() else "\n")
                )  # f"{' => ' if case and not arrow else ''}" +'\n'
            output += parse_proof(content, indent=indent + 1, dot=(not hasDecl))
            # subtheorem

    depth = indent * len(spaces)
    if output[:depth] == spaces * indent and dot:
        output = (
            output[: depth - 2] + ". " + output[depth:]
        )  # output[depth: depth + 1].replace(' ', '.') + output[depth + 1:]
    return output


def parseTheoremBase(thm, context=True):
    statement = thm.decl
    if context:
        context = thm.context
    else:
        context = ""
    proof = parse_proof(thm, dot=False)

    return f"{context}\n\n{statement} := by\n{proof}"


def parseTheorem(thm, context=True, annotation=False, prompt=False):
    if type(thm) == AnnotatedTheorem:
        return parseAnnotatedTheorem(thm, context, annotation)  # thm.pretty_print
    else:
        return parseTheoremBase(thm, context)


def run_training_data(root_path, module_name):
    os.chdir(root_path)
    cmd = f"lake exe training_data {module_name}"
    output = subprocess.run([cmd], shell=True, text=True, capture_output=True)
    data_raw = output.stdout
    if data_raw == "":
        raise KeyError(f"BAD DATA: {output}")
    data = json.loads(data_raw)
    return data


def annotateTheorem(thm: Theorem, force=False) -> AnnotatedTheorem:

    src = thm.src
    path = thm.leanFile
    text = parseTheorem(thm)
    og_thm_start = len(f"{thm.context}\n\n{thm.decl} := by\n".splitlines())
    og_thm_end = len(text.splitlines())

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    project_path = thm.project_path

    path_dir = os.path.join(project_path, os.path.dirname(path))

    temp = tempfile.NamedTemporaryFile(suffix=".lean", dir=path_dir)
    with open(temp.name, "w") as f:
        f.write(text)
    temp_relpath = os.path.relpath(temp.name, project_path)

    mod_name = get_stem(temp_relpath.replace("/", "."))
    output_data = run_training_data(root_path, mod_name)
    # print(
    #     f"-----------------\n(s,e)=({og_thm_start}, {og_thm_end})\nOutput:\n{output_data}\n-----------------"
    # )
    thms = getTheorems(
        output_data, src, temp_relpath, project_path, text, until_end=True
    )

    if len(thms) == 0:
        output = AnnotatedTheorem(
            decl=thm.decl,
            declID=thm.declID,
            src=thm.src,
            leanFile=thm.leanFile,
            context=thm.context,
            proof=[],
            project_path=thm.project_path,
            messages=[getMessage(msg, text) for msg in output_data["messages"]],
            pretty_print=text,
            proof_tree=[],
        )
    else:
        output = thms[-1]

    output.messages = getMessages(
        {"line": og_thm_start, "col": None},
        {"line": og_thm_end, "col": None},
        output_data["messages"],
        text,
    )
    elim_pf = elim_overlap(output.proof)

    if len(output.messages) == 0:
        output.proof = elim_pf
        return output
    else:
        # pretty print value is original theorem's text.
        # use first calculations to do partial annotation, and for the rest do all of the original

        first = None

        def flattenProof(proof):
            new_proof = []
            for stepraw in proof:
                step = stepraw.tactic
                if type(step) == str:
                    new_proof.append(stepraw)
                else:
                    decl = step.decl

                    new_proof.append(ProofStep(tactic=decl))
                    new_proof.extend(flattenProof(step.proof))
            return new_proof

        og_proof = flattenProof(thm.proof)
        # print(
        #     f"+++++++++++++++\nRAW:{thm.proof}\nOG:\n{og_proof}\n\nelim\n{[step.tactic for step in elim_pf]}\nout\n{[step.tactic for step in output.proof]}\n++++++++++++++++++="
        # )
        # print(thm.proof)
        for idx in range(min(len(og_proof), len(elim_pf))):
            if og_proof[idx].tactic != elim_pf[idx].tactic:
                first = idx
                break
            # print(f"[[[\nOG: {og_proof[idx].tactic}\nNEW: {elim_pf[idx].tactic}\n]]]")

        if first is None:
            if len(og_proof) != len(elim_pf):
                first = min(len(og_proof), len(elim_pf))

        if first is not None:
            if first != 0:
                max_pos = elim_pf[first - 1].end
            else:
                max_pos = (1, 1)

            if force:

                def get_empty_annotated_proof_step(i):
                    proofstep = og_proof[i]
                    return AnnotatedProofStep(
                        prevState=["ERROR"],
                        tactic=proofstep.tactic,
                        nextState=["ERROR"],
                        srcUpToTactic="ERROR",
                        declUpToTactic="ERROR",
                        start=(max_pos[0] + i, max_pos[1] + i),
                        end=(max_pos[0] + i, max_pos[1] + i),
                    )

                proof = [
                    get_empty_annotated_proof_step(i) if i >= first else elim_pf[i]
                    for i in range(len(og_proof))
                ]

                # if len(output.messages) == 0:
                #     msgs = [
                #         Message(
                #             severity="error",
                #             start=(None, None),
                #             end=(None, None),
                #             message_src=elim_pf[first - 1].tactic,
                #             content="UNKNOWN",
                #         )
                #     ]
                # else:
                #     msgs = output.messages

                final = AnnotatedTheorem(
                    decl=output.decl,
                    declID=output.declID,
                    src=output.src,
                    leanFile=output.leanFile,
                    context=output.context,
                    proof=proof,
                    project_path=project_path,
                    messages=output.messages,
                    pretty_print=output.pretty_print,
                    proof_tree=output.proof_tree,
                )

                if [s.tactic for s in og_proof] != [s.tactic for s in proof]:
                    raise ValueError(
                        f"=============Forcing Failed:\n{parseTheorem(thm,context=False)}\n{[s.tactic for s in og_proof]}\n--------\n{parseTheorem(final,context=False)}\n{[s.tactic for s in proof]}\n+++++++++++++++++++\n{[s.tactic for s in elim_pf]}\n{output.messages}\nfirst: {first}\n============="
                    )

                return final
            else:
                raise ValueError(
                    f"input theorem is incorrect! \n{parseTheorem(thm,context=False)}\n{parseTheorem(output,context=False)}\nfirst={first}\n{og_proof}\n{elim_pf}"
                )
        output.proof = elim_pf
        return output


if __name__ == "__main__":
    repo = getRepo("Tests", "configs/config_test.json")
    files = {file.file_path: file for file in repo.files}
    print(files.keys())
    fs = [files["Tests/Basic.lean"]]

    for f in fs:
        print(f"{f.file_name}|==================================")
        for thm in f.theorems:
            print(parseTheorem(thm, annotation=True, context=False))
            print("-------------------------")
