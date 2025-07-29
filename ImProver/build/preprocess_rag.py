import os
import json
import argparse
from tqdm import tqdm
import datetime
    

def main(args):
    with open(args.dataset_path, "r") as f:
        all_ds = json.load(f)
        all_splits = all_ds.keys()
        dataset = []
        for split in all_splits:
            dataset.extend(all_ds[split].values())
    files = []
    print(dataset)
    for repo in dataset:
        files.extend(repo)
    files_real = [file_info if type(file_info) is str else file_info["file"] for file_info in files]
    modules = set(f.replace(".lean", "").replace("/", ".") for f in files_real)
    modules_str = ",".join(sorted(modules))

    prompts_dir = os.path.join("rag", args.rag_id)

    os.makedirs(prompts_dir, exist_ok=True)

    cmd = f"lake exe preprocess_rag {modules_str} {prompts_dir}"
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {cmd}")
  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize RAG source")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--rag_id", type=str, default=f"rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    args = parser.parse_args()
    
    main(args)