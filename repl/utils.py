from git import Repo
import os

def get_repo_path(envvar,err=False):
    if envvar not in os.environ.keys():
        if err:
            raise FileNotFoundError(f"{envvar} environment variable not set")
    return os.getenv(envvar,None)

def set_repo_path(envvar,path):
    os.environ[envvar] = path
    return

def get_repo_lean_version(envvar,err=False):
    repo_path = get_repo_path(envvar,err=err)
    with open(os.path.join(repo_path,'lean-toolchain'),'r') as f:
        repo_version = f.read()
    return repo_version

def open_repo(url,envvar,force_clone_at = None):
    repo_path = get_repo_path(envvar)
    if repo_path is not None:
        return repo_path
    if force_clone_at is None:
        raise FileNotFoundError(f"Repo {url} not locally found. Set {envvar} to local path if exists.")
    Repo.clone_from(url,force_clone_at)
    set_repo_path(envvar,force_clone_at)
    return force_clone_at

        
    