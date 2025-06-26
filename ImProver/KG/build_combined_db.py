import os
import json
import argparse
import duckdb
from tqdm import tqdm


def load_edges(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main(args):
    os.makedirs(os.path.join(args.KG_dir, args.KG_id), exist_ok=True)
    combined_path = os.path.join(args.KG_dir, args.KG_id, "combined.duckdb")
    con = duckdb.connect(combined_path)
    con.execute("DROP TABLE IF EXISTS theorems")
    con.execute(
        """
        CREATE TABLE theorems(
            module TEXT,
            name TEXT,
            text TEXT,
            informalStatement TEXT,
            informalProof TEXT,
            isExtracted BOOLEAN,
            isOriginal BOOLEAN,
            isCorrect BOOLEAN,
            errorMsgs JSON,
            C1Dependencies JSON,
            C2Dependencies JSON,
            C3Dependencies JSON
        )
        """
    )

    class3_edges = load_edges(os.path.join(args.KG_dir, args.KG_id, "c3edges.json"))

    with open(args.dataset_path, "r") as f:
        all_ds = json.load(f)
        dataset = all_ds[args.split]
    files = []
    for repo in dataset.values():
        files.extend(repo)
        
    
    files_real = [file_info if type(file_info) is str else file_info["file"] for file_info in files]
    modules = set(f.replace(".lean", "").replace("/", ".") for f in files_real)

    # modules = set(f.replace(".lean", "").replace("/", ".") for f in files)


    informal_path = os.path.join(args.KG_dir, args.KG_id, "informal_data.duckdb")
    informal_con = duckdb.connect(informal_path)

    for root, _, files in os.walk(args.KG_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            if "config" in file:
                continue
            if "edges" in file:
                continue
            module_path = os.path.relpath(os.path.join(root, file), args.KG_dir)
            module = module_path.replace("/", ".").replace(".json", "")
            with open(os.path.join(root, file), "r") as f:
                theorems = json.load(f)
            for thm in theorems:
                name = thm.get("id",{}).get("name")
                is_orig = thm.get("id",{}).get("module", module) in modules
                c1 = json.dumps(thm.get("C1_dependencies", []))
                c2 = json.dumps(thm.get("C2_dependencies", []))
                c3 = json.dumps(class3_edges.get(f"{module}:{name}", []))
                error_msgs_raw = thm.get("id",{}).get("errorMsgs", [])
                error_msgs = json.dumps(error_msgs_raw)
                is_correct = True if error_msgs_raw == [] else False
                
                
                informal_statement = ""
                informal_proof = ""
                try:
                    # Query for matching row in informal database
                    result = informal_con.execute(
                        "SELECT informal_statement, informal_proof FROM informal_data WHERE module = ? AND name = ?",
                        [module, name]
                    ).fetchone()
                    
                    # If result exists, update the variables
                    if result:
                        informal_statement = result[0] if result[0] else ""
                        informal_proof = result[1] if result[1] else ""
                except Exception as e:
                    print(f"Error querying informal data for {module}:{name}: {e}")
                
                
                con.execute(
                    "INSERT INTO theorems VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        module,
                        name,
                        thm.get("id",{}).get("content"),
                        informal_statement,
                        informal_proof,
                        thm.get("id",{}).get("isExtracted", False),
                        is_orig,
                        is_correct,
                        error_msgs,
                        c1,
                        c2,
                        c3,
                    ),
                )
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build combined KG database")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("KG_id", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs")
    args = parser.parse_args()
    main(args)
