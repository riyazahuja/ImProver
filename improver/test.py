from typing import List
from pantograph.data import CompilationUnit
from pantograph import Server
from pantograph.expr import *
import re
import os
import tempfile
import subprocess
import json

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

        if url is not None and commit is not None:
            self.is_remote = True
        elif project_path is not None:
            self.is_remote = False
        else:
            raise ValueError(
                "Please specify either a remote repo or a local project path."
            )

    def get_server(self, force=False):
        if self.server is None or force:
            self.server = Server(imports=self.imports, project_path=self.project_path)
        return self.server

    def get_project_path(self):
        if self.is_remote:
            repos_dir = ".repos"
            os.makedirs(repos_dir, exist_ok=True)
            clone_path = os.path.join(repos_dir, self.name)
            if not os.path.exists(clone_path):
                subprocess.run(["git", "clone", self.url, clone_path], check=True)
            subprocess.run(["git", "checkout", self.commit], cwd=clone_path, check=True)
            self.project_path = clone_path
        return self.project_path

    @staticmethod
    def from_config(config: str):
        with open(config, "r") as f:
            data = json.load(f)

        if data["lean"] != "leanprover/lean4:v4.15.0":
            raise ValueError("Only lean 4.15.0 is supported.")

        return Repo(
            name=data["name"],
            url=data.get("url"),
            commit=data.get("commit"),
            project_path=data.get("project_path"),
            imports=data.get("imports", ["Init"]),
        )
        
    def get_files(self, only_modules = True):
        
        


class File:
    def __init__(
        self,
        path: str,
        repo: Repo,
    ):
        self.path = path
        self.repo = repo
        self.units = None
        self.server = None
        self.theorems = None
        self.is_module = None
    
    def get_units(self):
        self.units = self.repo.get_server().tactic_invocations(self.path)
        return self.units
     
    def get_server(self):
        repo_server = self.repo.get_server()
        
        
        
        
    def compile(self):

        if not self.server:
            imports = ["Init"]
            self.server = Server(imports=imports, project_path=self.project)

        temp_path = None
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=self.project, delete=False
        ) as tmp:
            tmp.write(self.contents)
            temp_path = tmp.name

        result = self.server.tactic_invocations(temp_path)
        unit = result[-1]

        with open(temp_path, "rb") as f:
            bytes = f.read()
            text = bytes[unit.i_begin : unit.i_end].decode("utf-8")

        return File(
            text,
            self.project,
            self.file,
            self.file_context,
            self.cached_environment,
            self.server,
        )


class Theorem:
    def __init__(
        self,
        contents: str,
        project: str = None,
        file: str = None,
        file_context: str = None,
        unit: CompilationUnit = None,
        cached_environment=None,
        server: Server = None,
    ):
        self.contents = contents
        self.project = project
        self.file = file
        self.file_context = file_context
        self.unit = unit
        self.cached_environment = cached_environment
        self.server = server

    def compile(self):

        if not self.server:
            imports = ["Init"]
            self.server = Server(imports=imports, project_path=self.project)

        temp_path = None
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=self.project, delete=False
        ) as tmp:
            tmp.write(self.contents)
            temp_path = tmp.name

        result = self.server.tactic_invocations(temp_path)
        unit = result[-1]

        with open(temp_path, "rb") as f:
            bytes = f.read()
            text = bytes[unit.i_begin : unit.i_end].decode("utf-8")

        return Theorem(
            text,
            self.project,
            self.file,
            self.file_context,
            unit,
            self.cached_environment,
            self.server,
        )
