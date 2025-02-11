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


class Repo:
    def __init__(
        self,
        name: str,
        url: str = None,
        commit: str = None,
        project_path: str = None,
        imports: List[str] = ["Init"],
        import_file: str = "",
    ):
        self.name = name
        self.url = url
        self.commit = commit
        self.project_path = project_path
        self.imports = imports
        self.import_file = import_file
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
            import_file=data.get("import_file", ""),
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
                        files.append(
                            File(
                                self,
                                imports=[
                                    os.path.relpath(path, self.get_project_path())
                                    .replace("/", ".")
                                    .replace(".lean", "")
                                ],
                                full_path=path,
                            )
                        )
        self.files = files
        return files


class File:
    def __init__(
        self,
        repo: Repo,
        imports: List[str] = ["Init"],
        full_path: str = None,  # in relation to the root of this project
        is_module: bool = None,
    ):
        self.full_path = full_path
        self.repo = repo
        self.units = None
        self.imports = imports
        self.server = None
        self.theorems = None
        self.is_module = is_module
        self.path = (
            os.path.relpath(full_path, repo.get_project_path()) if full_path else None
        )

    def get_path(self) -> str:
        self.path = (
            os.path.relpath(self.full_path, self.repo.get_project_path())
            if self.full_path
            else None
        )
        return self.path

    def get_units(self) -> List[CompilationUnit]:
        print(f"Verifying {self.path}")
        start = time.time()
        self.units = self.get_server().tactic_invocations(self.path)
        print(f"Verified {self.path} after {time.time() - start}s")
        return self.units

    def get_server(self, force=False) -> Server:
        if self.server is None or force:
            print("Getting (file) server...")
            start = time.time()
            self.server = Server(
                imports=self.imports, project_path=self.repo.get_project_path()
            )
            print("Server obtained after " + str(time.time() - start) + "s")

        return self.server

    def get_theorems(self):
        units = self.get_units()
        thms = []
        with open(self.full_path, "rb") as f:
            content = f.read()
        for unit in units:
            text = content[unit.i_begin : unit.i_end].decode("utf-8")
            if "theorem" in text or "lemma" in text or "example" in text:
                thms.append(Theorem(text, self.repo, self, unit))
        self.theorems = thms
        return thms


class Theorem:
    def __init__(
        self,
        contents: str,
        repo: Repo,
        file: File = None,
        unit: CompilationUnit = None,
    ):
        self.contents = contents
        self.file = (
            file
            if file is not None
            else File(os.path.join(repo.get_project_path(), repo.import_file), repo)
        )
        self.repo = repo
        self.context = None
        self.decl = None
        self.unit = unit

    def get_context(self, force=False) -> str:
        if self.context is not None and not force:
            return self.context

        with open(self.file.full_path, "rb") as f:
            content = f.read()
        unit = self.compile()

        ctx = content[0 : unit.i_begin].decode("utf-8")
        self.context = ctx
        return ctx

    def get_decl(self, force=False) -> str:
        if self.decl is not None and not force:
            return self.decl

        with open(self.file.full_path, "rb") as f:
            content = f.read()
        unit = self.compile()

        text = content[unit.i_begin : unit.i_end].decode("utf-8")
        decl = text.split(":=")[0]
        self.decl = decl
        return decl

    def compile(self, replace=False, force=False) -> CompilationUnit:
        if self.unit is not None and not force:
            return self.unit

        server = self.file.get_server()
        # print(server)
        content = self.contents
        if replace:
            content = re.sub(r"theorem\s+\w+", "example", content)
            content = re.sub(r"lemma\s+\w+", "example", content)

        units = server.load_sorry(content)

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


def parseTheorem(
    thm: Theorem, annotation=False, context=False, headerless_context=False
):
    thm_text = ""
    # TODO ANNOTATION

    thm_text = thm.contents

    if context:
        with open(thm.file.full_path, "rb") as f:
            content = f.read()
        unit = thm.compile()

        ctx = content[0 : unit[0].i_begin].decode("utf-8")
        thm_text = ctx + "\n\n" + thm_text
    return thm_text


if __name__ == "__main__":
    repo = Repo.from_config("configs/config_mathlib.json")
    # files = {f.path: f for f in repo.get_files(calculate_modules=False)}
    # file = files["Mathlib/Logic/Basic.lean"]
    content = """variable {α : Type*}
variable (s t u : Set α)
open Set

theorem fact : s ∩ t ∪ s ∩ u ⊆ s ∩ (t ∪ u) := by
  rintro x (⟨xs, xt⟩ | ⟨xs, xu⟩)
  · use xs; left; exact xt
  · use xs; right; exact xu
"""
    file = File(
        repo,
        imports=[
            "Mathlib.Data.Set.Lattice",
            "Mathlib.Data.Nat.Prime.Basic",
            "Mathlib.Tactic",
        ],
    )
    thm = Theorem(content, repo, file)
    unit = thm.compile()
    print(unit)
    for i in unit[-1].invocations:
        print(f"[Before]\n{i.before}")
        print(f"[Tactic]\n{i.tactic} (using {i.used_constants})")
        print(f"[After]\n{i.after}")
    print("\n---------------\n")
