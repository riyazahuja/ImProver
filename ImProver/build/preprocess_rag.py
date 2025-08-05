import multiprocessing
import os
import json
import argparse
import subprocess
import tqdm
import datetime
import asyncio    
import duckdb
import time




async def eval_repo(repo, modules_str, output_path):
    print(repo)
    st = time.time()
    # output at inference_dir/run_id/evals/[file_path].json
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # print(file)
    #['lake', 'exe', 'eval_improver', 'Compfiles.Usa2008P1', 'length', 'runs/RUN_20250515_031905', 'runs/RUN_20250515_031905/evals/Compfiles/Usa2008P1.json']['lake', 'exe', 'eval_improver', 'Compfiles.Usa2008P1', 'length', 'runs/RUN_20250515_031905', 'runs/RUN_20250515_031905/evals/Compfiles/Usa2008P1.json']
    cmd = ["lake", "exe", "preprocess_rag", output_path, modules_str]
    print(' '.join(cmd))
    proc = await asyncio.create_subprocess_exec(
            *cmd
            , stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
            )
    
    try:
        await asyncio.wait_for(proc.wait(),20*60)

        # stdout, stderr = await proc.communicate()
        # if proc.returncode != 0:
            # print(f">>> Error evaluating {file}: \n\tSTDOUT: {stdout.decode()}\n\tSTDERR: {stderr.decode()}\n")
        # else:
        print(f">>> success on {repo}! (took {time.time()-st}s)\n")
        return
    except asyncio.TimeoutError:
        proc.kill()
        print(f">>> [TIME-OUT] {repo} (> {20*60}s)")
        return
    except Exception as e:
        print(f">>> Exception running improver on {repo}: {str(e)}")
        return



async def runner(prompts_dir_base,modules_with_str,args):
    
    semaphore = asyncio.Semaphore(min(args.cpus,len(list(modules_with_str.keys()))))
    progress_bar = tqdm.tqdm(total=len(list(modules_with_str.keys())), desc="Processing files")

    async def worker(repo, module_str, output_path):
        async with semaphore:
            ok = await eval_repo(repo, module_str, output_path)
            progress_bar.update(1)
            return ok
    tasks = [asyncio.create_task(worker(repo,module_str, os.path.join(prompts_dir_base, repo))) for repo,module_str in modules_with_str.items()]
    await asyncio.gather(*tasks)
    progress_bar.close()
    
    
import duckdb
import glob

def build_duckdb_database(prompts_dir_base, db_path):
    """
    Build a duckdb database with decl_data and module_data tables,
    aggregating all decl_data.json and module_data.json files from all repos,
    using duckdb's read_json_auto for efficient loading.
    """
    import os

    # Find all decl_data.json and module_data.json files
    decl_data_files = glob.glob(os.path.join(prompts_dir_base, "*", "decl_data.json"))
    module_data_files = glob.glob(os.path.join(prompts_dir_base, "*", "module_data.json"))

    con = duckdb.connect(db_path)

    # Remove tables if they exist to allow re-creation
    con.execute("DROP TABLE IF EXISTS decl_data;")
    con.execute("DROP TABLE IF EXISTS module_data;")

    if decl_data_files:
        # Use duckdb's read_json_auto to read and concatenate all decl_data.json files
        decl_data_glob = os.path.join(prompts_dir_base, "*", "decl_data.json")
        con.execute(f"""
            CREATE TABLE decl_data AS
            SELECT * FROM read_json_auto('{decl_data_glob}', format='array');
        """)
    else:
        print("No decl_data.json files found.")

    if module_data_files:
        module_data_glob = os.path.join(prompts_dir_base, "*", "module_data.json")
        con.execute(f"""
            CREATE TABLE module_data AS
            SELECT * FROM read_json_auto('{module_data_glob}', format='array');
        """)
        # Remove duplicate modules, keeping only the row with the smallest depth for each module
        con.execute("""
            CREATE OR REPLACE TABLE module_data AS
            SELECT *
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY module ORDER BY depth ASC) as rn
                FROM module_data
            )
            WHERE rn = 1;
        """)
    else:
        print("No module_data.json files found.")

    con.close()

# Example usage:
# build_duckdb_database(prompts_dir_base, "improver_rag.duckdb")
    
    
    
    
   


def main(args):
    with open(args.dataset_path, "r") as f:
        all_ds = json.load(f)
    all_splits = all_ds.keys()
    dataset = {}
    for split in all_splits:
        if args.split is not None and split != args.split:
            continue
        
        contents = all_ds[split]
        for k,v in contents.items():
            if k not in dataset:
                dataset[k]=v
            else:
                dataset[k]= dataset[k] + v
    files = {}
    def fix_files_list(files_list):
        
        normalized = [file if type(file) is str else file["file"] for file in files_list]
        modules = set(f.replace(".lean", "").replace("/", ".") for f in normalized)
        return list(modules)

    # print(dataset)
    for repo,files_list in dataset.items():
        # print(repo, files_list)
        files[repo]=fix_files_list(files_list)
    # files_real = [file_info if type(file_info) is str else file_info["file"] for file_info in files]
    # modules = set(f.replace(".lean", "").replace("/", ".") for f in files_real)
    
    # modules = modules - {"Mathlib.Analysis.NormedSpace.ENormedSpace","Carleson.ToMathlib.ENorm"}
    
    modules_str = ""

    modules_with_str = {repo :  ",".join(modules) for repo,modules in files.items()}
    
    # modules_str = ",".join(sorted(modules))
    # modules_str2 = modules_str.replace(";", " ").replace(",", " ")
    # mod_cmd = f"lake build {modules_str2}"
    # with open("test.txt", "w") as test_file:
    #     for module in sorted(modules):
    #         test_file.write(f"import {module}\n")
    # ret = os.system(mod_cmd)

    prompts_dir_base = os.path.join("rag", args.rag_id, "src")
    
    asyncio.run(runner(prompts_dir_base,modules_with_str,args))

    build_duckdb_database(prompts_dir_base, os.path.join("rag", args.rag_id, "data.duckdb"))

    # os.makedirs(prompts_dir, exist_ok=True)
    

    # # Use subprocess instead of os.system to properly handle arguments
    # cmd = ["lake", "exe", "preprocess_rag", prompts_dir, modules_str]
    # # print(f"Running: {' '.join(cmd)}")
    # ret = subprocess.run(cmd, check=False)
    # if ret.returncode != 0:
    #     raise RuntimeError(f"Command failed with exit code {ret.returncode}: {' '.join(cmd)}")
  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize RAG source")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--rag_id", type=str, default=f"rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--cpus", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--split",type=str, default=None)
    args = parser.parse_args()
    
    main(args)