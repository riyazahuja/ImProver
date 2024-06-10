import os
import tempfile
import subprocess
import json
from utils import *
envvar = 'LEAN_REPL_PATH'
url = 'https://github.com/leanprover-community/repl'

def get_repl_path(force_clone_at = None):
    return open_repo(url,envvar,force_clone_at=force_clone_at)

def get_repl_version(err=False):
    get_repo_lean_version(envvar,err=err)

def check_version_correct(project_path,err=True):
    repl_version = get_repl_version().strip()
    with open(os.path.join(project_path,'lean-toolchain'),'r') as f:
        project_version = f.read().strip()
    if err:
        if repl_version != project_version:
            raise ValueError(f'directory toolchain and repl toolchain versions do not match:\nrepl: {repl_version}\nproj: {project_version}')
    return repl_version == project_version
  

def path_diff(dir_path, file_path):
    data = file_path[len(dir_path):]
    if data[0] == '/':
        data = data[1:]
    return data

def run_file(project_path, file_path, commands_path = None,force_clone_at=None):
    repl_path = open_repo(url,envvar,force_clone_at=force_clone_at)

    check_version_correct(project_path)

    cwd = os.getcwd()
    os.chdir(project_path)
    
    if commands_path is None:
        commands_path = tempfile.NamedTemporaryFile(suffix='.in').name

    with open(commands_path,'w+') as f:
        content = f'{{"path": "{file_path}", "allTactics": true}}'
        f.write(content)

    cmd = f"lake env {os.path.join(repl_path,'.lake/build/bin/repl')} < {commands_path}"
    output = subprocess.run([cmd],shell=True, text=True, capture_output = True)

    os.chdir(cwd)
    return json.loads(output.stdout)
    


def run_text(text, project_path, commands_path = None,force_clone_at=None):
    temp = tempfile.NamedTemporaryFile(suffix='.lean',dir=project_path)
    with open(temp.name,'w') as f:
        f.write(text)
    output = run_file(project_path,f.name,commands_path=commands_path,force_clone_at=force_clone_at)
    return output

