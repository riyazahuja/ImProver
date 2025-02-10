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
    input_module = input_module.replace("«", "").replace("»", "")
    if "Basic" not in input_module:
        return 0
    print(f"Extracting {input_module}")
    st = time.time()

    fp = os.path.join(start, input_module.replace(".", "/") + ".lean")
    output_path = os.path.join(
        output_base_dir, _get_stem(input_module, input_file_mode) + ".json"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    repl_cmd = f'{{"path": "{fp}", "allTactics":true, "theorems":true}}'
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

    def get_position_index(line: int, col: int) -> int:
        lines = contents.splitlines()
        index = sum(len(l) + 1 for l in lines[: line - 1])
        return index + col
    new_theorems = []
    ptr = 0
    for thm in thms:
        contents_split = contents.splitlines()
        thm_start = thm["start"]["line"] - 1
        context = "\n".join(contents_split[:thm_start])
        headerless_context = "\n".join(contents_split[header_end_line:thm_start])
        if get_position_index(thm["start"]["line"], thm["start"]["column"]) >= ptr:
            new_theorems.append(
                {**thm, "context": context, "headerless_context": headerless_context}
            )
            ptr = get_position_index(thm["end"]["line"], thm["end"]["column"])

    new_data = {
        "tactics": data.get("tactics", []),
        "messages": data.get("messages", []),
        "theorems": new_theorems,
        "headers": headers,
    }

    with open(output_path, "w") as f:
        json.dump(new_data, f)
    print(f"Extraction of {input_module} completed in {time.time()-st}s")
    print(f"Theorem cache preprocessing for {input_module} started...")
    st = time.time()

    # Getting file data: COMPLETE!
    # Now, need to parititon file into: C_0 = [0:thm[0].start), C_1= [thm[0].start:thm[1].start), ...
    # So \bigcup_{i<n} C_i is the context of the nth theorem
    # so if we make the infile: {cmd: C0} \n {pickle: theorem0, env 0} \n {cmd: C1, env 0} \n {pickle: theorem1, env 1} \n ...
    # So to compile theorem i, we just need to {unpack_pickle: C_i} \n {cmd: theorem_i, env 0}
    with open(fp, "r") as f:
        contents = f.read()


    def get_proof_range(theorem, tactics) -> tuple[int, int]:
        tactic_indices = theorem.get("tactics", [])
        if not tactic_indices:
            return None
        first = tactics[tactic_indices[0]]
        last = tactics[tactic_indices[-1]]

        start_position = get_position_index(
            first["pos"]["line"], first["pos"]["column"]
        )
        end_position = get_position_index(
            last["endPos"]["line"], last["endPos"]["column"]
        )
        return start_position, end_position

    thms = new_data.get("theorems", [])
    if len(thms) == 0:
        print(f"No theorems in {input_module}")
        return 0
    partition_syntax_ranges = [
        (0, get_position_index(thms[0]["start"]["line"], thms[0]["start"]["column"]))
    ] + [
        (
            get_position_index(
                thms[i - 1]["start"]["line"], thms[i - 1]["start"]["column"]
            ),
            get_position_index(thms[i]["start"]["line"], thms[i]["start"]["column"]),
        )
        for i in range(1, len(thms))
    ]
    tactics = new_data.get("tactics", [])
    content_slices = [
        contents[s:e] for s, e in partition_syntax_ranges
    ]  # so \bigcup_{i<= n} content_slices[i] is the context for theorem n (and contains theorem n-1)
    # however, with pickles, we can imagine content_slices[n] as the context for theorem n
    for n in range(len(content_slices)):
        print(f"Processing slice {n}/{len(content_slices)}")
        s,e = partition_syntax_ranges[n]
        print(f"Slice {n} range: {s} -> {e}")
        print(content_slices[n])
        print("----------")
    # now, going in reverse order, we replace the proof contents of theorems n-1 to theorem 0
    # (as given within the range get_proof_range thms[i]) with sorry

    tactics = new_data.get("tactics", [])
    print(f"Processing {len(content_slices)} content slices...")
    for n in range(1, len(content_slices)):
        
        # print(f"Processing slice {n}/{len(content_slices)}")
        # print(content_slices[n])
        # print("----------")
        try:
            start, end = get_proof_range(thms[n - 1], tactics)
            if start is None:
                continue
            s, _ = partition_syntax_ranges[n]
            start = start - s
            end = end - s

            # # Add bounds checking to prevent string index errors
            # if start >= len(content_slices[n]) or end > len(content_slices[n]):
            #     print(f"Warning: Invalid range for slice {n}: {start}:{end}")
            #     continue
            pre = content_slices[n][:start]
            post = content_slices[n][end:]
            content_slices[n] = (
                pre + "sorry" + post
            )
        except Exception as e:
            print(f"Error processing slice {n}: {e}")
            raise e

    Path(
        os.path.join(
            output_base_dir, _get_stem(input_module, input_file_mode) + "_theorems"
        )
    ).mkdir(parents=True, exist_ok=True)

    pickle_paths = [
        os.path.join(
            output_base_dir,
            _get_stem(input_module, input_file_mode) + "_theorems",
            f"{i}.o",
        )
        for i in range(len(content_slices))
    ]

    cmds = [
        json.dumps({"cmd": content_slices[0]})
        + "\n\n"
        + json.dumps({"pickleTo": pickle_paths[0], "env": 0})
    ] + [
        json.dumps({"cmd": content_slices[i], "env": i - 1})
        + "\n\n"
        + json.dumps({"pickleTo": pickle_paths[i], "env": i})
        for i in range(1, len(content_slices))
    ]
    print(f"Caching of {input_module} started...")
    st = time.time()
    # note here that the headers is content_slices[0]!

    temp = tempfile.NamedTemporaryFile(suffix=".in", dir=cwd, delete=False)
    with open(temp.name, "w") as f:
        f.write("\n\n".join(cmds))

    subprocess.run(
        [f"lake env repl/.lake/build/bin/repl < {temp.name}"], shell=True, cwd=cwd
    )

    print(f"Caching of {input_module} completed in {time.time() - st}s")

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
                start=proj_path,
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
