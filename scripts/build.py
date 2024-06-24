import argparse
import os
import subprocess
import json
from pathlib import Path
import requests
import base64
import shutil

def _lakefile(repo, commit, name, cwd):
    envvar = os.getenv("GITHUB_ACCESS_TOKEN")
    headers={'Authorization': f'token {envvar}'}
    url = f'https://api.github.com/repos/{repo.replace("https://github.com/","")}/contents/lakefile.lean'
    if envvar is None:
        req = requests.get(url)
    else:
        req = requests.get(url,headers=headers)
    if req.status_code == requests.codes.ok:
        req = req.json()  # the response is a JSON
        # req is now a dict with keys: name, encoding, url, size ...
        # and content. But it is encoded with base64.
        text = str(base64.b64decode(req['content']))
    else:
        text = ''
        print('Content was not found.')

    
    mathlib_text = ''
    if 'require mathlib from git' not in text and name != 'mathlib':
        mathlib_text = 'require mathlib from git\n    "https://github.com/leanprover-community/mathlib4.git"'
    contents = """import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    %s

    require %s from git
    "%s.git" @ "%s"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data

    """ % (mathlib_text, name, repo, commit)
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

def _setup(cwd,rebuild):
    print("Building...")
    if Path(os.path.join(cwd, '.lake')).exists():
        subprocess.Popen(['rm -rf .lake'], shell=True).wait()
    if Path(os.path.join(cwd, 'lake-packages')).exists():
        subprocess.Popen(['rm -rf lake-packages'], shell=True).wait()
    if Path(os.path.join(cwd, 'lake-manifest.json')).exists():
        subprocess.Popen(['rm -rf lake-manifest.json'], shell=True).wait()
    subprocess.Popen(['lake update'], shell=True).wait()
    subprocess.Popen(['lake exe cache get'], shell=True).wait()
    if rebuild:
        subprocess.Popen(['lake build'], shell=True).wait()

def _import_file(name, import_file, old_version):
    name = name.replace('«', '').replace('»', '') 
    if old_version:
        return os.path.join('lake-packages', name, import_file)
    else:
        return os.path.join('.lake', 'packages', name, import_file)

def _run(cwd, name, import_file, old_version, max_workers,start):
    
    flags = ''
    if max_workers is not None:
        flags += ' --max-workers %d' % max_workers

    proj_path = os.path.join(cwd,'.lake','packages',name)
    start_path = os.path.join(proj_path,start)
    flags += ' --start %s --proj-path %s' % (start_path, proj_path)
    subprocess.Popen(['python3 %s/scripts/run_pipeline.py --output-base-dir .cache/%s --cwd %s --import-file %s %s' % (
        cwd,
        name.capitalize(),
        cwd,
        _import_file(name, import_file, old_version),
        flags
    )], shell=True).wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    parser.add_argument(
        '--run',
        action=argparse.BooleanOptionalAction,
        default=True,
        help = 'Run the module?'
        )
    parser.add_argument(
        '--rebuild', 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help='runs lake build on imported module'
        )
    parser.add_argument(
        '--start', 
        default='',
        help="Start path in file tree of packages to be run"
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
            cwd=args.cwd,
            rebuild = args.rebuild
        )
        

        if args.run:
            src_dir = os.path.join(args.cwd,'.cache',source['name'])
            if os.path.isdir(src_dir):
                shutil.rmtree(src_dir)

            _run(
                cwd=args.cwd,
                name=source['name'],
                import_file=source['import_file'],
                old_version=False if 'old_version' not in source else source['old_version'],
                max_workers=args.max_workers,
                start = args.start
            )