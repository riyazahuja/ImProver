import os
import json
import argparse
import duckdb
import re
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import torch

def main(args):
    db_path = os.path.join("prompts", args.prompts_id, "informal_data.duckdb")
    chroma_dir = os.path.join("knowledge_graphs", args.kg_id, "chroma_db")
    
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute("SELECT module, name, informal_statement, informal_proof FROM informal_data").fetchall()
    con.close()

    client = PersistentClient(path=chroma_dir)
    embed = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embedding_model,
        device="cuda",                      # push model to available GPU(s)
        model_kwargs={
            # "device_map": "cuda:0",           # shard across multiple GPUs if present
            "torch_dtype": torch.float16    # cut VRAM/RAM usage in half
        }
    )
    collection = client.get_or_create_collection("informal_theorems", embedding_function=embed)

    edges = {}
    for module, name, stmt, proof in rows:
        query = f"What theorems/lemmas/facts does the following theorem depend on?\n\n{stmt}\n\n{proof}".strip()
        if not query:
            continue
        res = collection.query(query_texts=[query], n_results=args.k)
        deps = []
        for dep_id, score in zip(res["ids"][0], res["distances"][0]):
            if score <= args.threshold:
                dep_meta = collection.get(ids=[dep_id])["metadatas"][0]
                dep_name = dep_meta["name"]
                
                # Skip if the retrieved item has the same name
                if dep_name == name:
                    continue
                
                # Check for "extracted_split_{realName}_{some numbers}" pattern
                match = re.match(r"extracted_split_(.+?)_\d+$", dep_name)
                if match and match.group(1) == name:
                    continue
                
                deps.append({"module": dep_meta["module"], "name": dep_name})
        edges[f"{module}:{name}"] = deps
        print(f"Processed {module}:{name} with {len(deps)} dependencies")

    out_path = os.path.join("knowledge_graphs", args.kg_id, "c3edges.json")
    with open(out_path, "w") as f:
        json.dump(edges, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute class3 edges")
    parser.add_argument("prompts_id", type=str)

    parser.add_argument("kg_id", type=str)
    # parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs")
    # parser.add_argument("--chroma_dir", type=str, default = "chroma_db", help="Path to chroma db")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")

    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    main(args)
