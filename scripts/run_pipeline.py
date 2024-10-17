import glob
import os
import argparse
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json


def _get_stem(input_module, input_file_mode):
    if input_file_mode:
        stem = Path(input_module).stem.replace(".", "/")
    else:
        stem = input_module.replace(".", "/")
    return stem.replace("«", "").replace("»", "")


def _run_cmd(cmd, cwd, input_file, output_file):
    # print(f'cmd: {cmd}\n input: {input_file}\n output {output_file}')
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        subprocess.Popen(
            ["lake exe %s %s" % (cmd, input_file)], cwd=cwd, shell=True, stdout=f
        ).wait()


def _extract_module(input_module, input_file_mode, output_base_dir, cwd):
    # Tactic prediction
    # ratchet and todo fix later
    training_file = os.path.join(
        output_base_dir, _get_stem(input_module, input_file_mode) + "_training.json"
    )
    constants_file = os.path.join(
        output_base_dir, _get_stem(input_module, input_file_mode) + "_constants.json"
    )
    output_file = os.path.join(
        output_base_dir, _get_stem(input_module, input_file_mode) + ".json"
    )
    _run_cmd(
        cmd="training_data",
        cwd=cwd,
        input_file=input_module,
        output_file=training_file,
    )
    # UNSTABLE, NOT IN ARXIV BUILD
    _run_cmd(
        cmd="constants",
        cwd=cwd,
        input_file=input_module,
        output_file=constants_file,
    )

    with open(training_file, "r") as f:
        training_data = json.load(f)
    with open(constants_file, "r") as f:
        constant_data = json.load(f)
    with open(output_file, "w") as f:
        training_data["constants"] = constant_data
        json.dump(training_data, f)
    os.remove(training_file)
    os.remove(constants_file)

    print(input_module)
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

    # subprocess.Popen(
    #     ['python3 scripts/data_stats.py --pipeline-output-base-dir %s' % (args.output_base_dir)],
    #     cwd=args.cwd,
    #     shell=True
    # ).wait()
