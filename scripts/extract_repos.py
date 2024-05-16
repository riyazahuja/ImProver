import argparse
import os
import subprocess
import json
from pathlib import Path


def _lakefile(repo, commit, name, cwd):
    contents = """import Lake
open Lake DSL

package «lean-training-data» {
  -- add any package configuration options here
}

require %s from git
  "%s.git" @ "%s"

@[default_target]
lean_lib TrainingData where

lean_lib Examples where

lean_exe training_data where
  root := `scripts.training_data

lean_exe full_proof_training_data where
  root := `scripts.full_proof_training_data

lean_exe state_comments where
  root := `scripts.state_comments

""" % (name, repo, commit)
    with open(os.path.join(cwd, 'lakefile.lean'), 'w') as f:
        f.write(contents)


def _examples(imports, cwd):
    contents = """
%s
""" % ('\n'.join(['import %s' % i for i in imports]))
    with open(os.path.join(cwd, 'Examples.lean'), 'w') as f:
        f.write(contents)

def _lean_toolchain(lean, cwd):
    contents = """%s""" % (lean)
    with open(os.path.join(cwd, 'lean-toolchain'), 'w') as f:
        f.write(contents)

def _setup(cwd):
    print("Building...")
    if Path(os.path.join(cwd, '.lake')).exists():
        subprocess.Popen(['rm -rf .lake'], shell=True).wait()
    if Path(os.path.join(cwd, 'lake-packages')).exists():
        subprocess.Popen(['rm -rf lake-packages'], shell=True).wait()
    if Path(os.path.join(cwd, 'lake-manifest.json')).exists():
        subprocess.Popen(['rm -rf lake-manifest.json'], shell=True).wait()
    subprocess.Popen(['lake update'], shell=True).wait()
    subprocess.Popen(['lake exe cache get'], shell=True).wait()
    subprocess.Popen(['lake build'], shell=True).wait()

def _import_file(name, import_file, old_version):
    name = name.replace('«', '').replace('»', '') 
    if old_version:
        return os.path.join('lake-packages', name, import_file)
    else:
        return os.path.join('.lake', 'packages', name, import_file)

def _run(cwd, name, import_file, old_version):
    flags = '' 
    subprocess.Popen(['python %s/scripts/run_pipeline.py --output-base-dir Examples/%s --cwd %s --import-file %s %s' % (
        cwd,
        name.capitalize(),
        cwd,
        _import_file(name, import_file, old_version),
        flags
    )], shell=True).wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd', default='/Users/wellecks/projects/ntp-training-data/')
    parser.add_argument(
        '--config', 
        default='configs/config.json', 
        help='config file'
    )
    parser.add_argument(
        '--max-workers', 
        default=None, 
        type=int,
        help="maximum number of processes; defaults to number of processors"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        sources = json.load(f)

    for source in sources:
        print("=== %s ===" % (source['name']))
        print(source)
        _lakefile(
            repo=source['repo'],
            commit=source['commit'],
            name=source['name'],
            cwd=args.cwd
        )
        _examples(
            imports=source['imports'],
            cwd=args.cwd
        )
        _lean_toolchain(
            lean=source['lean'],
            cwd=args.cwd
        )
        _setup(
            cwd=args.cwd
        )
        _run(
            cwd=args.cwd,
            name=source['name'],
            import_file=source['import_file'],
            old_version=False if 'old_version' not in source else source['old_version']
        )