from __future__ import annotations
import argparse
from calendar import c
import json
from multiprocessing import process
import re
import argparse
from tarfile import data_filter
# from langchain.globals import set_debug
# from langchain_chroma import Chroma
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from chromadb import Client, PersistentClient
from chromadb.utils import embedding_functions
import torch
import os
import duckdb
# set_debug(False)





def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve related Mathlib theorems"
    )
    parser.add_argument("rag_id", type=str)
    parser.add_argument(
        "queries",
        help='JSON string containing the query in format {"query": "your query here"}',
    )
    return parser.parse_args()

if __name__ == "__main__":
    
        
        args = parse_args()

        # Parse the JSON query
        print(args.queries)
        query_data = json.loads(str(args.queries))
        queries = query_data.get("queries", "")
        k = query_data.get("k", 5)  # Default to 5 if not provided
        
        database_path = os.path.join("rag", args.rag_id, "informal_data.duckdb")
        conn = duckdb.connect(database_path, read_only=True)
        
        
        module_table_path = os.path.join("rag", args.rag_id, "data.duckdb")
        module_conn = duckdb.connect(module_table_path, read_only=True)

        
        processed_queries = []
        for query in queries:

            name = query.get("name", None)
            module = query.get("module", None)
            if name is None or module is None:
                print(f"Skipping query {query} because it has no name or module")
                continue
            name = name.replace("'", "''")
            module = module.replace("'", "''")
            df = conn.execute(f"SELECT text, informal_statement, informal_proof FROM informal_data WHERE name = '{name}' AND module = '{module}'").fetchall()
            if df is None or len(df) == 0:
                print(f"Skipping query {query} because it has no text")
                continue
            text = df[0][0]
            informal_statement = df[0][1]
            informal_proof = df[0][2]
            
            full_imports_result = module_conn.execute(f"SELECT fullImports FROM module_data WHERE module = '{module}'").fetchone()
            full_imports = full_imports_result[0] if full_imports_result is not None else []
            
            processed_queries.append({
                "name": name,
                "module": module,
                "full_imports": full_imports,
                "text": text,
                "informal_statement": informal_statement,
                "informal_proof": informal_proof
            })
        
        print(processed_queries)
        
        
        vector_db_path = os.path.join("rag", args.rag_id, "informal_retrieval_db")
        
        # Initialize embeddings and load the persisted Chroma vector store
        
        client = PersistentClient(path=vector_db_path)
        embed = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="taterowney/informal_proof_to_informal_statement_premise_selector",
            device="cuda",
            model_kwargs={
                "torch_dtype": torch.float16
            }
        )
        
        collection = client.get_or_create_collection("informal_retrieval_db", embedding_function=embed)
        
        
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="taterowney/informal_proof_to_informal_statement_premise_selector",
        #     device="cuda",
        # )

        # vectordb = Chroma(
        #     collection_name="informal_retrieval_db",
        #     persist_directory=vector_db_path,
        #     embedding_function=embeddings,
        # )
        # db = database.as_retriever(
        #     search_type="mmr", search_kwargs={"k": number_to_retrieve}
        # )

        # Retrieve documents for each processed query
        retrieval_results = []
        for item in processed_queries:        
            query_text = item["text"]
            if item["informal_statement"] is not None and item["informal_proof"] is not None:
                query_text = f"{item['informal_statement']}\n\n{item['informal_proof']}"
            elif item["informal_statement"] is not None:
                query_text = item["informal_statement"]
            
            
            imports = item["full_imports"]
            # print(imports)

            # Build a metadata filter so that we only retrieve from modules
            # that appear in the current query's full import list.
            filter_dict = {"module": {"$in": imports}} if imports else {}

            # Similarity search with metadata filtering
            # Use the ChromaDB collection's query method directly for flexible filtering and access to page_content and metadata.
            res = collection.query(
                query_texts=[query_text],
                n_results=k,
                where=filter_dict,
                include=["metadatas", "documents", "distances"]
            )
            docs = []
            for i in range(len(res["ids"][0])):
                doc = {
                    "page_content": res["documents"][0][i] if res["documents"] and res["documents"][0] else "",
                    "metadata": res["metadatas"][0][i] if res["metadatas"] and res["metadatas"][0] else {}
                }
                docs.append(doc)

            # Store the raw page content & metadata for each returned document
            retrieval_results.append({
            **item,
            "results": [
                {"page_content": doc["page_content"], "metadata": doc["metadata"]} for doc in docs
            ]
            })

        # Output the results in the same order as the original queries
        print("<OUTPUT>")
        print(json.dumps(retrieval_results))
        print("</OUTPUT>")
        