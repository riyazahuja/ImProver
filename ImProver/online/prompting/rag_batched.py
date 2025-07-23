from __future__ import annotations
from langchain.globals import set_debug
import asyncio
from typing import List, Dict, Any
# import sentence_transformers
set_debug(False)

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import os, json, sys
import argparse
import re
import duckdb
from rag import get_database_retriever, add_to_db, ROOT_PATH

# ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
METADATA_PATH = "/Users/ahuja/Desktop/ImProver_rewrite/RAG/annotated/Mathlib/"

devnull = open(os.devnull, "w")

old_stdout = sys.stdout
sys.stdout = devnull


# async def process_query(i, query: str, retriever, source_paths=None):
#     if source_paths:
#         docs = await retriever.ainvoke(
#             query,
#             filter={"source": {"$in": source_paths}},
#         )
#     else:
#         docs = await retriever.ainvoke(query)

#     results = []
#     for doc in docs:
#         src = doc.metadata.get("source", "Unknown source")
#         # src = src.replace(".lean", "")
#         contents = doc.metadata.get("decl", "Unknown decl")
#         # contents = doc.page_content
#         # contents = re.sub(r"/\-[\s\S]*?\-/", "", contents, flags=re.MULTILINE)
#         contents = "\n".join(line for line in contents.split("\n") if line.strip())

#         results.append(f"--src: {src.strip()}\n{contents}")

#     return (i, results)

async def get_from_db(id, conn, name, module):
    """
    Get the RAG documents from the database.
    """
    query = f"SELECT rag_docs FROM prompts WHERE name = ? AND module = ?"
    result = conn.execute(query, (name, module)).fetchone()
    
    if result:
        return (id, result)
    else:
        return (id, "")


async def main():
    parser = argparse.ArgumentParser(description="Retrieve related Mathlib theorems")
    parser.add_argument(
        "json_query",
        help='JSON string containing queries in format {"queries": [{"module": "module1", "name": "name1"}, {"module": "module2", "name": "name2"},], "k": 5, "imports": [...]}',
    )
    parser.add_argument(
        "--prompt_id",
        type=str,
        default="final_final_train",
        help="ID of the prompt to retrieve documents for.",
    )
    args = parser.parse_args()

    try:
        query_data = json.loads(args.json_query)


        queries = query_data.get("queries", [])
        if not queries:
            raise ValueError("Missing or empty 'queries' field in JSON")
        k = query_data.get("k", 5)
        imports = query_data.get("imports", None)
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON format"}))
        exit(1)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        exit(1)
    
    add_to_db(
        prompt_id=args.prompt_id,
        k=k,
    )

    prompt_id = args.prompt_id
    conn = duckdb.connect(os.path.join(ROOT_PATH, "prompts", prompt_id, "informal_data.duckdb"))

    # retriever = get_database_retriever(number_to_retrieve=k)

    source_paths = []
    if imports:
        source_paths = imports

    tasks = [
        # process_query(i, query, retriever, source_paths if imports else None)
        get_from_db(i, conn, query["name"], query["module"])
        for i, query in enumerate(queries)
    ]
    results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x[0])
    results = [result[1] for result in results]
    # print(results)
    sys.stdout = old_stdout
    devnull.close()
    print(json.dumps(results))


if __name__ == "__main__":
    asyncio.run(main())
