import glob
import os
import argparse
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


DIR_NAMES = {
    'training_data': 'TacticPrediction',
    'state_comments': 'StateComments',
    'full_proof_training_data': 'FullProof',
    'full_proof_training_data_states': 'FullProofWithStates',
}

def _get_stem(input_module, input_file_mode):
    if input_file_mode:
        stem = Path(input_module).stem.replace('.', '_')
    else:
        stem = input_module.replace('.', '_')
    return stem

def _run_cmd(cmd, cwd, input_file, output_file):
    with open(output_file, 'w') as f:
        subprocess.Popen(
            ['lake exe %s %s' % (cmd, input_file)], 
            cwd=cwd,
            shell=True,
            stdout=f
        ).wait()

def _extract_module(input_module, input_file_mode, output_base_dir, cwd):
    # Tactic prediction
    _run_cmd(
        cmd='training_data',
        cwd=cwd,
        input_file=input_module,
        output_file=os.path.join(
            output_base_dir, 
            DIR_NAMES['training_data'],
            _get_stem(input_module, input_file_mode) + '.jsonl'
        )
    )

    # Full proof generation
    _run_cmd(
        cmd='full_proof_training_data',
        cwd=cwd,
        input_file=input_module,
        output_file=os.path.join(
            output_base_dir, 
            DIR_NAMES['full_proof_training_data'],
            _get_stem(input_module, input_file_mode) + '.jsonl'
        )
    )

    #  # State comments
    # state_comments_output_file = os.path.join(
    #     output_base_dir, 
    #     DIR_NAMES['state_comments'],
    #     _get_stem(input_module, input_file_mode) + '.lean'
    # )
    # _run_cmd(
    #     cmd='state_comments',
    #     cwd=cwd,
    #     input_file=input_module,
    #     output_file=state_comments_output_file
    # )

    # # Full proof generation with state comments
    # _run_cmd(
    #     cmd='full_proof_training_data',
    #     cwd=cwd,
    #     input_file=state_comments_output_file,
    #     output_file=os.path.join(
    #         output_base_dir, 
    #         DIR_NAMES['full_proof_training_data_states'],
    #         _get_stem(input_module, input_file_mode) + '.jsonl'
    #     )
    # )

    print(input_module)
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-base-dir', default='Examples/Mathlib')
    parser.add_argument('--cwd', default='/Users/wellecks/projects/ntptutorial/partI_nextstep/ntp_lean/llm-training-data')
    parser.add_argument(
        '--input-file', 
        default=None, 
        help='input file in the Examples folder'
    )
    parser.add_argument(
        '--import-file', 
        help='Runs the pipeline on all modules imported in the given lean file.',
        default='.lake/packages/mathlib/Mathlib.lean'
    )
    parser.add_argument(
        '--max-workers', 
        default=None, 
        type=int,
        help="maximum number of processes; defaults to number of processors"
    )
    args = parser.parse_args()


    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    for name in DIR_NAMES.values():
        Path(os.path.join(args.output_base_dir, name)).mkdir(parents=True, exist_ok=True)

    print("Building...")
    subprocess.Popen(['lake build training_data'], shell=True).wait()
    subprocess.Popen(['lake build full_proof_training_data'], shell=True).wait()

    input_modules = []
    if args.input_file is not None:
        input_modules.extend(
            glob.glob(args.input_file)
        )
    elif args.import_file is not None:
        with open(args.import_file) as f:
            for line in f.readlines():
                if 'import ' in line:
                    chunks = line.split('import ')
                    module = chunks[1].strip()
                    input_modules.append(module)
    else:
        raise AssertionError('one of --input-file or --import-file must be set')

    completed = []
    start = time.time()
    with ProcessPoolExecutor(args.max_workers) as executor:
        input_file_mode = args.input_file is not None
        futures = [
            executor.submit(
                _extract_module,
                input_module=input_module,
                input_file_mode=input_file_mode,
                output_base_dir=args.output_base_dir,
                cwd=args.cwd
            )
            for input_module in input_modules
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            completed.append(future.result())

            if (len(completed)+1) % 10 == 0:
                end = time.time()
                print("Elapsed %.2f" % (round(end - start, 2)))
                print("Avg/file %.3f" % (round((end - start)/len(completed), 3)))

    end = time.time()
    print("Elapsed %.2f" % (round(end - start, 2)))

    subprocess.Popen(
        ['python scripts/data_stats.py --pipeline-output-base-dir %s' % (args.output_base_dir)], 
        cwd=args.cwd,
        shell=True
    ).wait()
