import os
import json
import argparse
import duckdb
from chromadb import PersistentClient


def main(args):
    con = duckdb.connect(args.db_path, read_only=True)
    rows = con.execute("SELECT module, name, informal_statement, informal_proof FROM informal_data").fetchall()
    con.close()

    client = PersistentClient(path=args.chroma_dir)
    collection = client.get_collection("informal_theorems")

    edges = {}
    for module, name, stmt, proof in rows:
        query = f"{stmt}\n\n{proof}".strip()
        if not query:
            continue
        res = collection.query(query_texts=[query], n_results=args.k)
        deps = []
        for dep_id, score in zip(res["ids"][0], res["distances"][0]):
            if score <= args.threshold:
                dep_meta = collection.get(id=dep_id)["metadatas"][0]
                deps.append({"module": dep_meta["module"], "name": dep_meta["name"]})
        edges[f"{module}:{name}"] = deps

    out_path = os.path.join(os.path.dirname(args.db_path), "class3_edges.json")
    with open(out_path, "w") as f:
        json.dump(edges, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute class3 edges")
    parser.add_argument("db_path", type=str, help="Path to informal duckdb")
    parser.add_argument("chroma_dir", type=str, help="Path to chroma db")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
