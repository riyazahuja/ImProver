from __future__ import annotations
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from typing import List, Union, Optional, Tuple
import uuid
import json
import tempfile
import subprocess
from textwrap import indent
import re
import bisect

"""
This is the important file: Describes all datastructures that we will use, as well as how to coerce between them and interact with the cache
"""


class ProofStep(BaseModel):
    tactic: Union[str, Theorem] = Field(
        description="One line/tactic in a tactic proof (str) or a subtheorem/sublemma/subproof"
    )


# Improve by using (annotated) theorems and files.
class Dependency(BaseModel):
    dependency: str = Field(
        description="Constant term in theorem dependent on external module"
    )
    src_file: str = Field(description="Source file of dependency")
    src_content: str = Field(
        description="source content (decl, def, proof, etc), just a few lines"
    )
    explicit: bool = Field(
        description="Is dependency explicit in the pp form of the parent theorem"
    )
    direct: bool = Field(description="is dependency direct or unfolded")
    kind: str = Field(
        description="What type of object is the dependency (theorem, def,etc)"
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
    headerless_context: str = Field(
        description="Context of the theorem without file headers"
    )
    project_path: str = Field(description="Local path to src repo contents")
    # dependencies: List[Dependency] = Field(
    #     description="Theorem dependencies from all imported modules"
    # )


class File(BaseModel):
    src: str = Field(description="File source repo")
    file_name: str = Field(description="File Name")
    file_path: str = Field(description="File path (stem) relative to src repo root")
    file_type: str = Field(description="File type")
    contents: str = Field(description="File contents")
    project_path: str = Field(description="Local path to src repo contents")


class AnnotatedProofStep(BaseModel):
    # prevState: List[str] = Field(
    #     description="Pretty printed tactic state before the tactic invocation"
    # )
    tactic: str = Field(description="One line/tactic in a tactic proof.")
    nextState: List[str] = Field(
        description="Pretty printed tactic state after the tactic invocation"
    )
    # srcUpToTactic: str = Field(
    #     description="Source code from file start to current tactic"
    # )
    # declUpToTactic: str = Field(
    #     description="Source code from theorem declaration to current tactic"
    # )
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
    # message_tactic_src: Optional[AnnotatedProofStep] = Field(
    #     description="tactic object containing message_src"
    # )
    content: str = Field(description="Message contents")


class AnnotatedTheorem(BaseModel):
    decl: str = Field(description="Theorem declaration")
    declID: str = Field(description="Unique theorem declaration ID")
    src: str = Field(description="Source repo of the theorem")
    leanFile: str = Field(description="Lean file in which theorem is located")
    context: str = Field(
        description="Context of the theorem (i.e. file contents up to decl)"
    )
    headerless_context: str = Field(
        description="Context of the theorem without file headers"
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
    start: Tuple[Optional[int], Optional[int]] = Field(
        description="start coordinates from source file as (row,column)"
    )
    end: Tuple[Optional[int], Optional[int]] = Field(
        description="end coordinates from source file as (row,column)"
    )
    # dependencies: List[Dependency] = Field(
    #     description="Theorem dependencies from all imported modules"
    # )


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
    header: str = Field(description="Header of the file")


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


def search_main(module, project_path):
    fp = os.path.join(project_path, module.replace(".", "/") + ".lean")
    if os.path.isfile(fp):
        with open(fp, "r") as f:
            content = f.read()
        return content
    else:
        return None


def search_packages(module, project_path, package=None):
    packages_path = os.path.join(project_path, ".lake", "packages")
    if package is not None:
        paths = [os.path.join(packages_path, package)]
    else:
        paths = [
            os.path.join(packages_path, name) for name in os.listdir(packages_path)
        ]
    paths = [path for path in paths if os.path.isdir(path)]

    outputs = [search_main(module, path) for path in paths]
    outputs = [output for output in outputs if output is not None]
    if len(outputs) == 0:
        return None
    else:
        return outputs[0]


# THIS WILL NOT FIND ANY BASE LEAN LIBRARY DEPENDENCIES
def getDependencies(
    constants,
    declID,
    ignore_main=False,
    only_main=False,
):

    declIDs = {
        f"{obj['module']}.{obj['range']['start']['line']}_{obj['range']['start']['column']}": obj
        for obj in constants
    }

    main_module = ".".join(declID.split(".")[:-2])
    declID = ".".join(declID.split(".")[:-1])

    if declID not in declIDs.keys():
        return []

    data = declIDs[declID]
    dependencies_raw = data["dependents"]
    output = [
        Dependency(
            dependency=dep["name"],
            src_file=dep["module"],
            src_content=dep["content"],
            explicit=dep["explicit"],
            direct=dep["direct"],
            kind=dep["kind"],
        )
        for dep in dependencies_raw
    ]

    if ignore_main:
        output = [dep for dep in output if dep.src_file != main_module]
    if only_main:
        output = [
            dep
            for dep in output
            if dep.src_file.split(".")[0] == main_module.split(".")[0]
        ]

    return output


def getTheorems(
    data, src, path, project_path, contents, until_end=False
) -> List[AnnotatedTheorem]:
    temp = {}
    msgs = data["messages"]
    tactics = data["tactics"]
    theorems = data["theorems"]

    output = []
    for thm in theorems:
        thm_tactics = [tactics[i] for i in thm["tactics"]]
        proof = [
            AnnotatedProofStep(
                tactic=tac["tactic"],
                nextState=[tac["goals"]],
                start=(tac["pos"]["line"], tac["pos"]["column"]),
                end=(tac["endPos"]["line"], tac["endPos"]["column"]),
            )
            for tac in thm_tactics
        ]

        thm_msgs = [msgs[i] for i in thm["messages"]]
        thm_messages = [
            Message(
                severity=msg["severity"],
                start=(msg["pos"]["line"], msg["pos"]["column"]),
                end=(msg["endPos"]["line"], msg["endPos"]["column"]),
                content=msg["data"],
                message_src="\n".join(
                    contents.splitlines()[
                        msg["pos"]["line"] - 1 : msg["endPos"]["line"]
                    ]
                ),
            )
            for msg in thm_msgs
        ]

        decl = thm["decl"]
        context = thm["context"]
        headerless_context = thm["headerless_context"]
        declID = str(uuid.uuid4())

        if len(thm["proofTree"]) == len(proof):
            proof_tree = [
                (pair["tactic"], pair["children"], pair["spawned_children"])
                for i, pair in enumerate(thm["proofTree"])
            ]
        else:
            proof_tree = []
        start = (thm["start"]["line"], thm["start"]["column"])
        end = (thm["end"]["line"], thm["end"]["column"])

        pretty_print = contents

        output.append(
            AnnotatedTheorem(
                leanFile=path,
                src=src,
                decl=decl,
                declID=declID,
                proof=proof,
                context=context,
                headerless_context=headerless_context,
                project_path=project_path,
                messages=thm_messages,
                pretty_print=pretty_print,
                proof_tree=proof_tree,
                start=start,
                end=end,
            )
        )
    return output


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

    headers = data["headers"]

    return AnnotatedFile(
        src=src,
        file_name=os.path.basename(path),
        contents=contents,
        theorems=theorems,
        file_path=path,
        file_type=ftype,
        project_path=project_path,
        header=headers,
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


def parseAnnotatedTheorem(
    thm: AnnotatedTheorem, annotation=False, context=False, headerless_context=False
):
    context_str = ""
    if context:
        context_str = thm.context
    elif headerless_context:
        context_str = thm.headerless_context
    if context_str != "":
        context_str += "\n"
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

        state_text_after_line = {}
        for pos in tactic_line_positions:
            line_after = pos[1]
            text = pos[2].nextState
            text = (
                "/-\n" + "\n".join(text) + "\n-/"
                if len(text) != 0
                else "/-\nGoals Solved!\n-/"
            )
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
        pp = "\n".join(thm.pretty_print.splitlines()[thm.start[0] - 1 : thm.end[0]])
        return f"{context_str}{pp}"

    return f"{context_str}{decl} := by\n{proof_text}"


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


def parseTheoremBase(thm: Theorem, context=True, headerless_context=False):
    statement = thm.decl
    if context:
        context = thm.context
    elif headerless_context:
        context = thm.headerless_context
    else:
        context = ""
    proof = parse_proof(thm, dot=False)
    if context != "":
        context += "\n"
    return f"{context}{statement} := by\n{proof}"


def parseTheorem(thm, annotation=False, context=False, headerless_context=False):
    if type(thm) == AnnotatedTheorem:
        return parseAnnotatedTheorem(
            thm,
            annotation=annotation,
            context=context,
            headerless_context=headerless_context,
        )  # thm.pretty_print
    else:
        return parseTheoremBase(
            thm, context=context, headerless_context=headerless_context
        )


def coerce_repl(data, thm, contents=None):
    src = thm.src
    path = thm.leanFile

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_path = thm.project_path

    file_name = os.path.basename(path).replace(".lean", "")

    tactics = data.get("tactics", [])
    msgs = data.get("messages", {})

    target_thms = data.get("theorems", [])
    if len(target_thms) == 0:
        raise ValueError(f"BAD REPL CALL:\n{data}")

    target_thm = target_thms[-1]

    offset = len(thm.context.splitlines()) - (target_thm["start"]["line"] - 1) + 1
    # the +1 is to account for the lean line numbers starting at 1 and python starting at zero so
    # this ensures that we stay with that system here too (as the thm start/end values are supposed to be
    # from lean)

    thm_tactics = [tactics[i] for i in target_thm["tactics"]]
    proof = [
        AnnotatedProofStep(
            tactic=tac["tactic"],
            nextState=[tac["goals"]],
            start=(tac["pos"]["line"] + offset, tac["pos"]["column"]),
            end=(tac["endPos"]["line"] + offset, tac["endPos"]["column"]),
        )
        for tac in thm_tactics
    ]
    padding = "\n" * 5
    if contents is None:
        contents = parseTheorem(thm, context=True) + padding
    thm_msgs = [msgs[i] for i in target_thm["messages"]]
    thm_messages = [
        Message(
            severity=msg["severity"],
            start=(msg["pos"]["line"] + offset, msg["pos"]["column"]),
            end=(msg["endPos"]["line"] + offset, msg["endPos"]["column"]),
            content=msg["data"],
            message_src="\n".join(
                contents.splitlines()[
                    msg["pos"]["line"] + offset - 1 : msg["endPos"]["line"] + offset
                ]
            ),
        )
        for msg in thm_msgs
    ]

    decl = target_thm["decl"]
    context = thm.context
    headerless_context = thm.headerless_context
    declID = str(uuid.uuid4())

    if len(target_thm["proofTree"]) == len(proof):
        proof_tree = [
            (pair["tactic"], pair["children"], pair["spawned_children"])
            for i, pair in enumerate(target_thm["proofTree"])
        ]
    else:
        proof_tree = []

        proof_tree = [
            (pair["tactic"], pair["children"], pair["spawned_children"])
            for i, pair in enumerate(target_thm["proofTree"])
        ]
    start = (target_thm["start"]["line"] + offset, target_thm["start"]["column"])
    end = (target_thm["end"]["line"] + offset, target_thm["end"]["column"])

    pretty_print = contents

    return AnnotatedTheorem(
        leanFile=path,
        src=src,
        decl=decl,
        declID=declID,
        proof=proof,
        context=context,
        headerless_context=headerless_context,
        project_path=project_path,
        messages=thm_messages,
        pretty_print=pretty_print,
        proof_tree=proof_tree,
        start=start,
        end=end,
    )


def annotateTheorem(thm: Theorem) -> AnnotatedTheorem:
    """
    Before, in the old version, we literally ran "getTheorems" (i.e. lake exe training_data)
    and then parsed the output to get the proof steps. This was slow and inefficient.

    Now, we parse the theorem's headerless context and raw text into a REPL command,
    and construct a .in file that initializes the environment via the binary cache and
    runs the command. Then we parse the REPL output to get a theorem object.
    """

    src = thm.src
    path = thm.leanFile
    text = parseTheorem(thm, context=False)

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, f".cache", src, os.path.dirname(path))

    file_name = os.path.basename(path).replace(".lean", "")
    binary_path = os.path.join(cache_path, file_name + ".o")

    cmd_text = thm.headerless_context + "\n\n" + text

    cmd_text = cmd_text.replace("\n", "\\n")  # .replace('"', '\\"')

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

    out = "\n".join(output.stdout.splitlines()[2:])
    data = json.loads(out)
    print(data)
    print()
    return coerce_repl(data, thm)


# requires all theorems to have the same declaration/src/file/etc.
def annotateTheorems(thms: List[Theorem]) -> AnnotatedTheorem:
    """
    Before, in the old version, we literally ran "getTheorems" (i.e. lake exe training_data)
    and then parsed the output to get the proof steps. This was slow and inefficient.

    Now, we parse the theorem's headerless context and raw text into a REPL command,
    and construct a .in file that initializes the environment via the binary cache and
    runs the command. Then we parse the REPL output to get a theorem object.
    """

    src = thms[0].src
    path = thms[0].leanFile

    src = thm.src
    path = thm.leanFile
    text = parseTheorem(thm, context=False)

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, f".cache", src, os.path.dirname(path))

    file_name = os.path.basename(path).replace(".lean", "")
    binary_path = os.path.join(cache_path, file_name + ".o")

    cmd_text = thm.headerless_context + "\n\n" + text

    cmd_text = cmd_text.replace("\n", "\\n")  # .replace('"', '\\"')

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

    out = "\n".join(output.stdout.splitlines()[2:])
    data = json.loads(out)
    print(data)
    print()
    return coerce_repl(data, thm)


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

    # repo = getRepo("Tests", "configs/config_MIL.json")
    # files = {file.file_path: file for file in repo.files}

    # fs = [files["Tests/MIL/C04_Sets_and_Functions/solutions/Solutions_S01_Sets.lean"]]

    # for f in fs:
    #     print(f"{f.file_name}|==================================")
    #     for thm in [f.theorems[0]]:
    print(parseTheorem(thm, annotation=False, context=False))
    print(thm.messages)
    print("-------------------------")
    # print(parseTheorem(thm, annotation=True, headerless_context=True))
