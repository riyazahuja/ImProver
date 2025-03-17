from __future__ import annotations
from langchain.globals import set_debug
import asyncio
from typing import List, Dict, Any

set_debug(False)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import os, json, sys
import argparse
import re

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_PATH = "/Users/ahuja/Desktop/ImProver_rewrite/RAG/annotated/Mathlib/"


def get_database_retriever(package_name="Mathlib", number_to_retrieve=5):
    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_annotated_db"
    )
    embeddings = OllamaEmbeddings(model="llama3.2")

    database = Chroma(
        collection_name="Annotated_Mathlib_Theorems",
        persist_directory=database_path,
        embedding_function=embeddings,
    )
    return database.as_retriever(
        search_type="mmr", search_kwargs={"k": number_to_retrieve}
    )


async def process_query(i, query: str, retriever, source_paths=None):
    if source_paths:
        docs = await retriever.ainvoke(
            query,
            filter={"source": {"$in": source_paths}},
        )
    else:
        docs = await retriever.ainvoke(query)

    results = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown source")
        src = src.replace(".lean", "")

        contents = doc.page_content
        contents = re.sub(r"/\-[\s\S]*?\-/", "", contents, flags=re.MULTILINE)
        contents = "\n".join(line for line in contents.split("\n") if line.strip())

        results.append(f"--src: {src.strip()}\n{contents}")

    return (i, results)


async def main():
    parser = argparse.ArgumentParser(description="Retrieve related Mathlib theorems")
    parser.add_argument(
        "json_query",
        help='JSON string containing queries in format {"queries": ["query1", "query2"], "k": 5, "imports": [...]}',
    )
    args = parser.parse_args()

    try:
        # query_data = json.loads(args.json_query)

        query_data = {
            "queries": [
                "theorem t8 : { n | Nat.Prime n } ∩ { n | n > 2 } ⊆ { n | ¬Even n } := by\n  /-\n    ⊢ HasSubset.Subset (Inter.inter (setOf fun n => Nat.Prime n) (setOf fun n => G …\n  -/\n  intro n\n  /-\n    n : Nat\n    ⊢ Membership.mem (Inter.inter (setOf fun n => Nat.Prime n) (setOf fun n => GT. …\n  -/\n  simp\n  /-\n    n : Nat\n    ⊢ Nat.Prime n → LT.lt 2 n → Odd n\n  -/\n  intro nprime n_gt\n  /-\n    n : Nat\n    nprime : Nat.Prime n\n    n_gt : LT.lt 2 n\n    ⊢ Odd n\n  -/\n  rcases Nat.Prime.eq_two_or_odd nprime with h | h\n    /-\n      case inl\n      n : Nat\n      nprime : Nat.Prime n\n      n_gt : LT.lt 2 n\n      h : Eq n 2\n      ⊢ Odd n\n    -/\n  · rw [h]\n    /-\n      case inl\n      n : Nat\n      nprime : Nat.Prime n\n      n_gt : LT.lt 2 n\n      h : Eq n 2\n      ⊢ Odd 2\n    -/\n    linarith\n    /-\n      🎉 no goals\n    -/\n    /-\n      case inr\n      n : Nat\n      nprime : Nat.Prime n\n      n_gt : LT.lt 2 n\n      h : Eq (HMod.hMod n 2) 1\n      ⊢ Odd n\n    -/\n  · rw [Nat.odd_iff, h]\n    /-\n      🎉 no goals\n    -/\n\n"
            ],
            "k": 5,
        }

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

    retriever = get_database_retriever(number_to_retrieve=k)

    source_paths = []
    if imports:
        source_paths = [os.path.join(METADATA_PATH, path) for path in imports]

    tasks = [
        process_query(i, query, retriever, source_paths if imports else None)
        for i, query in enumerate(queries)
    ]
    results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x[0])
    results = [result[1] for result in results]
    # print(results)
    print(json.dumps(results))


if __name__ == "__main__":
    asyncio.run(main())
