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
    pretty_print: str = Field(description="Pretty printed (string) form of theorem.")
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
        if thm_start_row <= start_row and end is not None and thm_end_row >= end_row:
            # print(f'\nCASE 1 <================= {msg}\n')
            return True
        elif thm_start_row <= start_row and end is None:
            # print(f'\nCASE 2 <================= {msg}\n')
            return True
        else:
            return False

    msgs2 = [msg for msg in msgs if msg_in_thm(msg)]
    # s = "\n".join(map(lambda x: f'{x[0]+1} | {x[1]}',list(enumerate(contents.splitlines()))))
    # print(f'+++++++++++++++\nthm_start_row={thm_start_row}\nMSG CONTENTS:\n{s}\nMSGS:\n{msgs}\nMSGS2:\n{msgs2}\n+++++++++++++++\n')
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
    # print(data)
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

    # messages = [getMessage(msg,contents) for msg in msgs]
    all_tacs = [re.sub(r"\s", "", step["tactic"]) for step in data]
    # print([step['tactic'] for step in data])

    def process_children(data):
        output = []
        for child in data:
            if child.get("kind", None) == "TacticInfo":
                pp = child.get("node", {}).get("stx", {}).get("pp", "")
                trim_pp = re.sub(r"\s", "", pp)
                if trim_pp in all_tacs:
                    idx = [i for i, tac in list(enumerate(all_tacs)) if trim_pp == tac][
                        -1
                    ]
                    output.append(idx)
                    grandchildren = child.get("children", [])
                    output.extend(process_children(grandchildren))
        return output

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
        # print(f'\n###########\n{decl} : start -> {thm_start}\n################\n')
        messages = getMessages(thm_start, thm_end, msgs, contents)
        messages.reverse()
        lines_src = step["srcUpToTactic"].split("\n")
        decl_lines = decl.split("\n")
        # print(lines_src)

        # lines = [line for line in lines_src if not line in decl_lines]
        maybe_context = "\n".join(lines_src[: -len(decl_lines) - 1]).strip()

        if thm_end is not None:
            pp = "\n".join(
                contents.splitlines()[thm_start["line"] - 1 : thm_end["line"]]
            )
        else:
            pp = "\n".join(contents.splitlines()[thm_start["line"] - 1 :])

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
            # print(temp)
        else:
            # print(temp)
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
    # print(f'{path} : {path[-5:]} : {path[:-5]}')
    if path[-5:] == ".lean":
        return path[:-5]
    elif path[-5:] == ".json":
        return path[:-5]
    return path


def getAnnotatedFile(src, path, project_path, until_end=False):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, f".cache/{src}")

    # content_path = os.path.join(root_path,'.lake','packages',src)

    # print(f'{project_path}|{path}')
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
    # print(f'{src} | {path}')
    if annotate:
        try:
            return getAnnotatedFile(src, path, project_path)
        except FileNotFoundError as e:
            if force:
                raise e
            else:
                # print(f'ERROR: \n{e}\n\n')
                pass
    stem, ftype = os.path.splitext(path)
    with open(os.path.join(project_path, path), "r") as f:
        content = f.read()
    # print(f'{os.path.basename(path)} | {stem} | {ftype}')

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
    # print(repo_data)

    return getRepoDirect(repo_data, annotate=annotate, force=force)


def parseAnnotatedTheorem(thm, context=True, annotation=False):
    last_pstep = thm.proof[-1]
    if context:
        src = last_pstep.srcUpToTactic + last_pstep.tactic
    else:
        src = last_pstep.declUpToTactic + last_pstep.tactic
    return src


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


def annotate(step: AnnotatedProofStep, prev_goal=False):
    prev = step.prevState
    tactic = step.tactic
    next = step.nextState

    def pp_state(state):
        if state == []:
            return "No Goals Left!"
        return "\n".join(state)

    prev_text = f"""
/-
{pp_state(prev)}
-/
"""
    text = f"""
{tactic}
/-
{pp_state(next)}
-/
"""
    if prev_goal:
        return indent(prev_text + text, "  ", lambda x: True)
    else:
        return indent(text, "  ", lambda x: True)


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
    # print (f'PARSING PROOF: req {depth}, first [{output[:depth]}], dot? {dot}')
    if output[:depth] == spaces * indent and dot:
        # print('HELOO!!!!')
        output = (
            output[: depth - 2] + ". " + output[depth:]
        )  # output[depth: depth + 1].replace(' ', '.') + output[depth + 1:]
    return output


def parseTheoremBase(thm, context=True, prompt=False):
    statement = thm.decl
    if context:
        context = thm.context
    else:
        context = ""
    proof = parse_proof(thm, dot=False)

    if prompt:
        return f"CONTEXT:\n {context}\n\n THEOREM: {statement} := by\n{proof}"
    else:
        return f"{context}\n\n{statement} := by\n{proof}"


def parseTheorem(thm, context=True, annotation=False, prompt=False):
    if type(thm) == AnnotatedTheorem:
        return thm.pretty_print
    else:
        return parseTheoremBase(thm, context, prompt)


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

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    project_path = thm.project_path

    path_dir = os.path.join(project_path, os.path.dirname(path))

    temp = tempfile.NamedTemporaryFile(suffix=".lean", dir=path_dir)
    with open(temp.name, "w") as f:
        f.write(text)
    temp_relpath = os.path.relpath(temp.name, project_path)

    mod_name = get_stem(temp_relpath.replace("/", "."))
    output = run_training_data(root_path, mod_name)

    thms = getTheorems(output, src, temp_relpath, project_path, text, until_end=True)

    if len(thms) == 0:
        output = AnnotatedTheorem(
            decl=thm.decl,
            declID=thm.declID,
            src=thm.src,
            leanFile=thm.leanFile,
            context=thm.context,
            proof=[],
            project_path=thm.project_path,
            messages=[],
            pretty_print=parseTheorem(thm, context=False),
            proof_tree=[],
        )
    else:
        output = thms[-1]

    elim_pf = elim_overlap(output.proof)

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

    for idx in range(min(len(og_proof), len(elim_pf))):
        if og_proof[idx].tactic != elim_pf[idx].tactic:
            first = idx
            break

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

    return output
