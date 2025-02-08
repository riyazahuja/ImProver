from typing import List
from pantograph.data import CompilationUnit
from pantograph import Server
from pantograph.expr import *
from pantograph.server import ServerError
import re
import os
import tempfile
import subprocess
import json
import time

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# server = Server(
#     imports=["Mathlib.Logic.Basic"], project_path="/Users/ahuja/Desktop/mathlib4"
# )
# out = server.env_module_read("Mathlib.Logic.Basic")
# for k, v in out.items():
#     print(f"============ {k} =============")
#     print(" ".join(v))
# print("\n\n")

# out = server.env_inspect("Ne.dite_ne_left_iff")
# print(out)

# state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
# state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a b c")
# state2 = server.goal_tactic(state1, goal_id=0, tactic="have h : Or b a:= Or.comm.1 c")

# server.env_add(
#     "dumb",
#     "forall (p q: Prop), Or p q -> Or q p",
#     "sorry",
# )

# out = server.env_inspect("dumb", print_value=True, print_dependency=True)
# print(out)

# states = [state0, state1, state2]

# for i, state in enumerate(states):
#     print("===== " + "State " + str(i) + " =====")
#     print("------ PP: ------")
#     print(str(state))
#     print("------ raw: ------\n")
#     print(state.__repr__())


class Repo:
    def __init__(
        self,
        name: str,
        url: str = None,
        commit: str = None,
        project_path: str = None,
        imports: List[str] = ["Init"],
    ):
        self.name = name
        self.url = url
        self.commit = commit
        self.project_path = project_path
        self.imports = imports

        self.server = None
        self.files = None

        if url is not None and commit is not None:
            self.is_remote = True
        elif project_path is not None:
            self.is_remote = False
        else:
            raise ValueError(
                f"Please specify either a remote repo or a local project path.\n\n{name} | {url} | {commit} | {project_path} | {imports}"
            )

    def get_server(self, force=False) -> Server:
        if self.server is None or force:
            print("Getting server...")
            start = time.time()
            self.server = Server(
                imports=self.imports, project_path=self.get_project_path()
            )
            print("Server obtained after " + str(time.time() - start) + "s")

        return self.server

    def get_project_path(self, rebuild=False) -> str:
        if self.project_path is not None:
            return self.project_path
        if self.is_remote:
            repos_dir = ".repos"
            os.makedirs(repos_dir, exist_ok=True)
            clone_path = os.path.join(repos_dir, self.name)
            if not os.path.exists(clone_path):
                subprocess.run(["git", "clone", self.url, clone_path], check=True)
                subprocess.run(
                    ["git", "checkout", self.commit], cwd=clone_path, check=True
                )
                subprocess.run(
                    ["lake", "exe", "cache", "get"], cwd=clone_path, check=True
                )
                subprocess.run(["lake", "build"], cwd=clone_path, check=True)
            else:
                subprocess.run(
                    ["git", "checkout", self.commit], cwd=clone_path, check=True
                )
                if rebuild:
                    subprocess.run(
                        ["lake", "exe", "cache", "get"], cwd=clone_path, check=True
                    )
                    subprocess.run(["lake", "build"], cwd=clone_path, check=True)
            self.project_path = clone_path

        return self.project_path

    @staticmethod
    def from_config(config: str) -> "Repo":
        with open(config, "r") as f:
            data = json.load(f)

        if data["lean"] != "leanprover/lean4:v4.15.0":
            raise ValueError("Only lean 4.15.0 is supported.")

        return Repo(
            name=data["name"],
            url=data.get("url"),
            commit=data.get("commit"),
            project_path=data.get("path"),
            imports=data.get("imports", ["Init"]),
        )

    def get_files(self, calculate_modules=True, force=False) -> List["File"]:
        if self.files is not None and not force:
            return self.files

        project_path = self.get_project_path()
        files = []
        for root, dirs, file_list in os.walk(project_path):
            if ".lake" in dirs:
                dirs.remove(".lake")  # Skip .lake directory
            for file in file_list:
                if file.endswith(".lean"):
                    path = os.path.join(root, file)
                    if calculate_modules:
                        module_name = (
                            os.path.relpath(path, project_path)
                            .replace("/", ".")
                            .replace(".lean", "")
                        )
                        try:
                            output = self.get_server().env_module_read(module_name)
                            files.append(File(path, self, is_module=True))
                        except ServerError as e:
                            # print(
                            #     f"Error reading module {module_name} in file {path}: \n{e}"
                            # )
                            files.append(File(path, self, is_module=False))
                    else:
                        files.append(File(path, self))
        self.files = files
        return files


class File:
    def __init__(
        self,
        full_path: str,  # in relation to the root of this project
        repo: Repo,
        is_module: bool = None,
    ):
        self.full_path = full_path
        self.repo = repo
        self.units = None
        self.server = None
        self.theorems = None
        self.is_module = is_module
        self.path = os.path.relpath(full_path, repo.get_project_path())

    def get_units(self) -> List[CompilationUnit]:
        print(f"Verifying {self.path}")
        start = time.time()
        self.units = self.repo.get_server().tactic_invocations(self.path)
        print(f"Verified {self.path} after {time.time() - start}s")
        return self.units

    def get_server(self) -> Server:
        self.server = self.repo.get_server()


class Theorem:
    def __init__(
        self,
        contents: str,
        file: File,
        repo: Repo,
        original: bool = None,
        unit: CompilationUnit = None,
    ):
        self.contents = contents
        self.file = file
        self.repo = repo
        self.original = original
        self.unit = unit

    def compile(self, force=False) -> CompilationUnit:
        if self.unit is not None and not force:
            return self.unit

        server = self.file.get_server()

        temp_path = None
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            dir=os.path.dirname(self.file.full_path),
            delete=False,
        ) as tmp:
            tmp.write(self.contents)
            temp_path = tmp.name

        temp_file = File(temp_path, self.file.repo)
        units = temp_file.get_units()
        self.unit = units[-1]
        return units[-1]

    @staticmethod
    def compile_theorems(theorems: List["Theorem"], force=False) -> List["Theorem"]:
        if not force:
            theorems_to_compile = [
                (i, t) for i, t in enumerate(theorems) if t.unit is None
            ]
            if not theorems_to_compile:
                return theorems
        else:
            theorems_to_compile = theorems

        combined_contents = "\n\n\n".join(t.contents for _, t in theorems_to_compile)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            dir=os.path.dirname(theorems[0].file.full_path),
            delete=True,
        ) as tmp:
            tmp.write(combined_contents)
            temp_path = tmp.name

        temp_file = File(temp_path, theorems[0].repo)
        units = temp_file.get_units()

        if len(units) != len(theorems_to_compile):
            raise ValueError(
                f"Got {len(units)} units but expected {len(theorems_to_compile)}"
            )
        idxs = [i for i, t in theorems_to_compile]
        thms = [theorems[i] for i in idxs]
        for theorem, unit in zip(thms, units):
            theorem.unit = unit
        enum_thms = [(idxs[i], thms[i]) for i in range(len(thms))]
        rest_thms = [(i, t) for i, t in enumerate(theorems) if i not in idxs]

        return [t for _, t in sorted(enum_thms + rest_thms, key=lambda x: x[0])]


if __name__ == "__main__":
    repo = Repo.from_config("configs/config_PNT.json")
    files = {f.path: f for f in repo.get_files(calculate_modules=False)}
    file = files["PrimeNumberTheoremAnd/Sobolev.lean"]
    print(file.full_path)
    units = file.get_units()
    print("\n\n")

    theorem_strings: List[Theorem] = []
    with open(file.full_path, "rb") as f:
        content = f.read()
        for i, unit in enumerate(units):
            unit_text = content[unit.i_begin : unit.i_end].decode("utf-8")
            if (
                (
                    unit_text.startswith("lemma")
                    or unit_text.startswith("theorem")
                    or unit_text.startswith("example")
                )
                and unit.invocations is not None
                and len(unit.invocations) > 0
            ):
                # print(f"#{i}: [{unit.i_begin},{unit.i_end}]")
                # print(unit_text)
                theorem_strings.append(Theorem(unit_text, file, repo))
    thm = theorem_strings[0]
    unit = thm.compile()
    print(unit.__dict__)
