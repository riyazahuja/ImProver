import os
import argparse
import duckdb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions


def build_db(db_path, out_dir, model="intfloat/e5-base-v2"):
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        "SELECT module, name, text, informal_statement, informal_proof, isExtracted, isOriginal FROM informal_data"
    ).fetchall()
    con.close()

    os.makedirs(out_dir, exist_ok=True)
    client = PersistentClient(path=out_dir)
    embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
    collection = client.get_or_create_collection("informal_theorems", embedding_function=embed)

    documents, metadatas, ids = [], [], []
    for idx, row in enumerate(rows):
        module, name, text, stmt, proof, is_ex, is_orig = row
        doc = f"{text}\n\n{stmt}\n\n{proof}"
        metadata = {
            "module": module,
            "name": name,
            "text": text,
            "informal_statement": stmt,
            "informal_proof": proof,
            "isExtracted": is_ex,
            "isOriginal": is_orig,
        }
        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"{module}:{name}")
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector DB for informal theorems")
    parser.add_argument("db_path", type=str, help="Path to informal duckdb")
    parser.add_argument("--out_dir", type=str, default="chroma_db")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2")
    args = parser.parse_args()

    build_db(args.db_path, args.out_dir, args.model)
