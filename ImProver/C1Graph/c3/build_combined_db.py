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
    os.makedirs(os.path.join(args.KG_dir, "class3"), exist_ok=True)
    combined_path = os.path.join(args.KG_dir, "class3", "combined.duckdb")
    con = duckdb.connect(combined_path)
    con.execute("DROP TABLE IF EXISTS theorems")
    con.execute(
        """
        CREATE TABLE theorems(
            module TEXT,
            name TEXT,
            text TEXT,
            isExtracted BOOLEAN,
            isOriginal BOOLEAN,
            C1Dependencies JSON,
            C2Dependencies JSON,
            C3Dependencies JSON
        )
        """
    )

    class3_edges = load_edges(os.path.join(args.KG_dir, "class3", "edges.json"))

    with open(args.dataset_path, "r") as f:
        all_ds = json.load(f)
        dataset = all_ds[args.split]
    files = []
    for repo in dataset.values():
        files.extend(repo)
    modules = set(f.replace(".lean", "").replace("/", ".") for f in files)

    for root, _, files in os.walk(args.KG_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            if "filtered" in root and "config" in file:
                continue
            if "class3" in root and "edges" in file:
                continue
            module_path = os.path.relpath(os.path.join(root, file), args.KG_dir)
            module = module_path.replace("/", ".").replace(".json", "")
            with open(os.path.join(root, file), "r") as f:
                theorems = json.load(f)
            for thm in theorems:
                name = thm.get("name")
                is_orig = thm.get("module", module) in modules
                c1 = json.dumps(thm.get("C1_dependencies", []))
                c2 = json.dumps(thm.get("C2_dependencies", []))
                c3 = json.dumps(class3_edges.get(f"{module}:{name}", []))
                con.execute(
                    "INSERT INTO theorems VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        module,
                        name,
                        thm.get("text"),
                        thm.get("isExtracted", False),
                        is_orig,
                        c1,
                        c2,
                        c3,
                    ),
                )
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build combined KG database")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--KG_dir", type=str, default="KG2.75")
    args = parser.parse_args()
    main(args)
