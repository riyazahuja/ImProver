import glob
import os
import argparse
import subprocess
import json
import tiktoken
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


JSONL_DIRS = {
    'training_data': 'TacticPrediction',
}

SRC_DIRS = {
    'split_rw': 'SplitRw',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline-output-base-dir', default='Examples/Mathlib')
    args = parser.parse_args()

    enc = tiktoken.encoding_for_model("gpt-4")

    stats = {}

    for k, v in SRC_DIRS.items():
        print(v)
        files = glob.glob(os.path.join(args.pipeline_output_base_dir, v, '*.lean'))
        stats['%s_files' % v] = len(files)

        num_lines = 0
        num_tokens = 0

        for f in tqdm(files, total=len(files)):
            with open(f, 'r') as f:
                text = f.read()
                tokens = len(enc.encode(text))
                num_tokens += tokens
                num_lines += len(text.split('\n'))
        stats['%s_lines' % v] = num_lines
        stats['%s_tokens' % v] = num_tokens


    for k, v in JSONL_DIRS.items():
        print(v)
        files = glob.glob(os.path.join(args.pipeline_output_base_dir, v, '*.jsonl'))
        stats['%s_files' % v] = len(files)

        num_examples = 0
        for f in tqdm(files, total=len(files)):
            with open(f, 'r') as f:
                for line in f.readlines():
                    num_examples += 1
        stats['%s_examples' % v] = num_examples

    for k, v in stats.items():
        print(k, v, sep='\t')

    with open(os.path.join(args.pipeline_output_base_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)
    