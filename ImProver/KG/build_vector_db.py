import os
import argparse
import duckdb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import torch
import gc
import datetime


def build_db(db_path, out_dir, model):
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        "SELECT module, name, text, informal_statement, informal_proof FROM informal_data"
    ).fetchall()
    con.close()

    os.makedirs(out_dir, exist_ok=True)
    client = PersistentClient(path=out_dir)
    embed = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model,
        device="cuda",                      # push model to available GPU(s)
        model_kwargs={
            "device_map": "cuda:0",           # shard across multiple GPUs if present
            "torch_dtype": torch.float16    # cut VRAM/RAM usage in half
        }
    )
    collection = client.get_or_create_collection("informal_theorems", embedding_function=embed)

    documents, metadatas, ids = [], [], []
    for idx, row in enumerate(rows):
        module, name, text, stmt, proof = row
        doc = f"{text}\n\n{stmt}\n\n{proof}"
        metadata = {
            "module": module,
            "name": name,
            "text": text,
            "informal_statement": stmt,
            "informal_proof": proof,
        }
        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"{module}:{name}")

    if documents:
        print(f"Adding {len(documents)} documents to the collection.")

        # Add documents in batches
        BATCH_SIZE = 8  # smaller batches to avoid GPU OOM
        for i in range(2752, len(documents), BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, len(documents))
            # if i//BATCH_SIZE + 1 < 58*2:
            #     continue
            print(f"Adding batch {i//BATCH_SIZE + 1}/{(len(documents) + BATCH_SIZE - 1)//BATCH_SIZE}: documents {i} to {end_idx-1}")
            collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
            )
            # --- free GPU & CPU memory ---
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()


def main(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    
    db_path = os.path.join("prompts", args.prompts_id, "informal_data.duckdb")
    out_dir = os.path.join("knowledge_graphs", args.KG_id, "chroma_db")

    build_db(db_path, out_dir, args.model)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Build vector DB for informal theorems")
    parser.add_argument("prompts_id", type=str)
    parser.add_argument("KG_id", type=str, nargs='?', default="KG_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    # parser.add_argument("--prompts_dir", type=str, default=".prompts")
    # parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()
    