import os
import argparse
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import tempfile


def _get_stem(input_module, input_file_mode):
    if input_file_mode:
        stem = Path(input_module).stem.replace(".", "/")
    else:
        stem = input_module.replace(".", "/")
    return stem.replace("«", "").replace("»", "")


def _extract_module(input_module, input_file_mode, output_base_dir, cwd, start):
    # I am actually going to cry this is so scuffed
    # If you aren't riyaz and reading this, trust me I can actually code
    # This is just a disgusting patch that actually works for some reason
    # for the love of god fix this to be less stupid

    print(f"Extracting {input_module}")

    fp = os.path.join(start, input_module.replace(".", "/") + ".lean")
    output_path = os.path.join(
        output_base_dir, _get_stem(input_module, input_file_mode) + ".json"
    )
    # print(f'cmd: {cmd}\n input: {input_file}\n output {output_file}')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # if "C04" not in input_module or "Solutions_S01" not in input_module:
    #     print(f"File Path: {fp}")
    #     return 1
    repl_cmd = f'{{"path": "{fp}", "allTactics":true, "theorems":true}}'
    # repl_cmd = f'{{"path": "{fp}", "allTactics":true}}'
    temp = tempfile.NamedTemporaryFile(suffix=".in", dir=cwd)
    with open(temp.name, "w") as f:
        f.write(repl_cmd)

    subprocess.run(
        [f"lake env repl/.lake/build/bin/repl < {temp.name} > {output_path}"],
        shell=True,
        cwd=cwd,
    )

    with open(output_path, "r") as f:
        data = json.load(f)

    # need to add context and headerless context to each theorem, and headers to the overall file
    with open(fp, "r") as f:
        contents = f.read()
    thms = data.get("theorems", [])
    headers = ""
    header_end_line = 0  # exclusive
    if len(thms) > 0:

        fst = thms[0]
        header_end_line = max(fst["start"]["line"] - 1, 0)
        headers = "\n".join(contents.splitlines()[:header_end_line])

    new_theorems = []
    for thm in thms:
        contents_split = contents.splitlines()
        thm_start = thm["start"]["line"] - 1
        context = "\n".join(contents_split[:thm_start])
        headerless_context = "\n".join(contents_split[header_end_line:thm_start])
        new_theorems.append(
            {**thm, "context": context, "headerless_context": headerless_context}
        )

    new_data = {
        "tactics": data.get("tactics", []),
        "messages": data.get("messages", []),
        "theorems": new_theorems,
        "headers": headers,
    }

    with open(output_path, "w") as f:
        json.dump(new_data, f)

    pickle_path = os.path.join(
        output_base_dir, _get_stem(input_module, input_file_mode) + ".o"
    )

    text = f'{{"cmd": "{headers}"}}'.replace("\n", "\\n")  # .replace('"', '\\"')

    text2 = f'{{"pickleTo": "{pickle_path}", "env": 0}}'.replace("\n", "\\n")
    text = text + "\n\n" + text2

    temp = tempfile.NamedTemporaryFile(suffix=".in", dir=cwd)
    with open(temp.name, "w") as f:
        f.write(text)

    subprocess.run(
        [f"lake env repl/.lake/build/bin/repl < {temp.name}"], shell=True, cwd=cwd
    )

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base-dir", default="Examples/Mathlib")
    parser.add_argument(
        "--cwd", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    parser.add_argument(
        "--input-file", default=None, help="input file in the Examples folder"
    )
    parser.add_argument(
        "--import-file",
        help="Runs the pipeline on all modules imported in the given lean file.",
        default=".lake/packages/mathlib/Mathlib.lean",
    )
    parser.add_argument(
        "--max-workers",
        default=None,
        type=int,
        help="maximum number of processes; defaults to number of processors",
    )
    parser.add_argument(
        "--start", default=None, help="Start path in file tree of packages to be run"
    )
    parser.add_argument(
        "--proj-path", default="", help="Start path in file tree of project to be run"
    )
    args = parser.parse_args()

    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    if args.start is None:
        args.start = ""

    print("Building...")
    subprocess.Popen(["lake build training_data"], shell=True).wait()

    input_modules = []
    # if args.input_file is not None:
    #     input_modules.extend(glob.glob(args.input_file))
    if args.import_file is not None:
        with open(args.import_file) as f:
            for line in f.readlines():
                # ERROR: READING COMMENTS TOO!
                if "import " in line:
                    chunks = line.split("import ")
                    module = chunks[1].strip()
                    input_modules.append(module)
    else:
        raise AssertionError("one of --input-file or --import-file must be set")

    # modify input_modules to handle start
    files_in_path = []
    start = args.start
    proj_path = args.proj_path

    for root, dirs, files in os.walk(start):
        for file in files:
            path = os.path.relpath(os.path.join(root, file), start=proj_path)
            if path.endswith(".lean"):
                files_in_path.append(path)
    if os.path.isfile(start):
        files_in_path = [os.path.relpath(start, proj_path)]
    # print(f"walked from {start}: \n{files_in_path}")

    completed = []
    start_path = start
    start = time.time()
    with ProcessPoolExecutor(args.max_workers) as executor:
        input_file_mode = args.input_file is not None
        # print(
        #     f"{input_modules} | {input_file_mode} | {[_get_stem(mod, input_file_mode) for mod in input_modules]}"
        # )
        input_modules = [
            mod
            for mod in input_modules
            if _get_stem(mod, input_file_mode) + ".lean" in files_in_path
        ]
        # print(f'Input Modules: {input_modules}')
        # print('-----------------')
        futures = [
            executor.submit(
                _extract_module,
                input_module=input_module,
                input_file_mode=input_file_mode,
                output_base_dir=args.output_base_dir,
                cwd=args.cwd,
                start=start_path,
            )
            for input_module in input_modules
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            completed.append(future.result())

            if (len(completed) + 1) % 10 == 0:
                end = time.time()
                print("Elapsed %.2f" % (round(end - start, 2)))
                print("Avg/file %.3f" % (round((end - start) / len(completed), 3)))

    end = time.time()
    print("Elapsed %.2f" % (round(end - start, 2)))
