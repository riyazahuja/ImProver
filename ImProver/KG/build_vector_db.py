import os
import argparse
import duckdb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import torch
import gc
import datetime


def build_db(db_path, out_dir, model):
    # Save current CUDA_VISIBLE_DEVICES and set to GPU 0
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        "SELECT module, name, text, informal_statement, informal_proof FROM informal_data"
    ).fetchall()
    con.close()

    os.makedirs(out_dir, exist_ok=True)
    client = PersistentClient(path=out_dir)
    print(f"Client created at: {out_dir}, now initializing embedding fn")
    embed = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model,
        device="cuda",                      # push model to available GPU(s)
        model_kwargs={
            # "device_map": "cuda:0",           # shard across multiple GPUs if present
            "torch_dtype": torch.float16    # cut VRAM/RAM usage in half
        }
    )
    print(f"initializing collection with embedding model: {model}")
    collection = client.get_or_create_collection("informal_theorems", embedding_function=embed)
    print(f"Collection 'informal_theorems' created with embedding model: {model}, now adding {len(rows)} documents.")
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
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(rows)} ({idx / len(rows) * 100:.2f}%) rows...")

    print(f"Total documents to add: {len(documents)}")
    if documents:
        print(f"Adding {len(documents)} documents to the collection.")

        # Add documents in batches, with dynamic batch size adjustment
        MAX_BATCH_SIZE = 64
        CURR_BATCH_SIZE = MAX_BATCH_SIZE
        i = 0
        while i < len(documents):
            end_idx = min(i + CURR_BATCH_SIZE, len(documents))
            # if i//BATCH_SIZE + 1 < 58*2:
            #     continue
            print(f"Adding batch (size = {CURR_BATCH_SIZE}), documents {i} to {end_idx-1} - currently {i / len(documents) * 100:.2f}% complete")

            try:
                collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
                )
                
                i += CURR_BATCH_SIZE
                if CURR_BATCH_SIZE < MAX_BATCH_SIZE:
                    CURR_BATCH_SIZE = min(MAX_BATCH_SIZE, CURR_BATCH_SIZE * 2)
                    print(f"Increasing batch size to {CURR_BATCH_SIZE} for next iteration.")
                
            except Exception as e:
                print(f"Error adding batch {i//CURR_BATCH_SIZE + 1}: {e}")
                CURR_BATCH_SIZE = max(1, CURR_BATCH_SIZE // 2)
                print(f"Reducing batch size to {CURR_BATCH_SIZE} to avoid OOM errors.")


            # --- free GPU & CPU memory ---
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            
        
        
        # BATCH_SIZE = 16  # smaller batches to avoid GPU OOM
        # for i in range(0, len(documents), BATCH_SIZE):
        #     end_idx = min(i + BATCH_SIZE, len(documents))
        #     # if i//BATCH_SIZE + 1 < 58*2:
        #     #     continue
        #     print(f"Adding batch {i//BATCH_SIZE + 1}/{(len(documents) + BATCH_SIZE - 1)//BATCH_SIZE}: documents {i} to {end_idx-1}")
        #     collection.add(
        #     documents=documents[i:end_idx],
        #     metadatas=metadatas[i:end_idx],
        #     ids=ids[i:end_idx]
        #     )
        #     # --- free GPU & CPU memory ---
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #         torch.cuda.ipc_collect()
        #     gc.collect()
    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices


def main(args):
    print("entered vector main")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print("garbage collected")
    
    db_path = os.path.join("prompts", args.prompts_id, "informal_data.duckdb")
    out_dir = os.path.join("knowledge_graphs", args.kg_id, "chroma_db")

    build_db(db_path, out_dir, args.embedding_model)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Build vector DB for informal theorems")
    parser.add_argument("prompts_id", type=str)
    parser.add_argument("kg_id", type=str, default="KG_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    # parser.add_argument("--prompts_dir", type=str, default=".prompts")
    # parser.add_argument("--KG_dir", type=str, default=".knowledge_graphs")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()
    
    main(args)
    