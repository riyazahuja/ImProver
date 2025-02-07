import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from benchmark.tools import no_errors
import math


def get_theorem_idx(theorems, repo: Repo):
    files = {f.file_path: f for f in repo.files}
    print(files.keys())
    file_map = [
        (files[thm.leanFile], files[thm.leanFile].theorems.index(thm))
        for thm in theorems
    ]
    return file_map


def cache_it(thm: Theorem | AnnotatedTheorem, f: File | AnnotatedFile, idx: int):

    output_base_dir = os.path.join(".cache", f.src, f.file_path)
    if output_base_dir.endswith(".lean"):
        output_base_dir = output_base_dir[:-5]

    Path(output_base_dir).parent.mkdir(parents=True, exist_ok=True)

    pickle_path = os.path.join(output_base_dir, idx + ".o")

    text = json.dumps({"cmd": thm.context})
    text2 = json.dumps({"pickleTo": pickle_path, "env": 0})
    text = f"{text}\n\n{text2}"

    temp = tempfile.NamedTemporaryFile(suffix=".in", dir=cwd)
    with open(temp.name, "w") as f:
        f.write(text)

    subprocess.run(
        [f"lake env repl/.lake/build/bin/repl < {temp.name}"], shell=True, cwd=cwd
    )

    return 1


def cache_theorems(theorems, repo: Repo):
    idxs = get_theorem_idx(theorems, repo)
    for i in range(len(theorems)):
        cache_it(theorems[i], *idxs[i])


if __name__ == "__main__":
    repo = getRepo("htpi", "configs/config_htpi.json")
    files = {file.file_path: file for file in repo.files}
    fs = [files[name] for name in files.keys() if type(files[name]) == AnnotatedFile]
    fs = fs[: math.floor(0.8 * len(fs))]

    thms = [
        thm
        for f in fs
        for thm in f.theorems
        if type(f) == AnnotatedFile
        and no_errors([thm])
        and type(thm) == AnnotatedTheorem
        and len(thm.proof) > 4
        and len(thm.proof) < 30
        and ("lemma" in thm.decl or "example" in thm.decl or "theorem" in thm.decl)
    ]

    unique_thms = []
    seen_decls = set()
    for thm in thms:
        trim_decl = (
            thm.decl.replace("theorem ", "")
            .replace("lemma ", "")
            .replace("example ", "")
            .replace("problem ", "")
        ).strip()
        if trim_decl not in seen_decls:
            unique_thms.append(thm)
            seen_decls.add(trim_decl)
    thms = unique_thms
    thms = thms[:100]
    thms = sorted(thms, key=lambda x: len(x.proof))

    cache_theorems([thms[0]], repo)
