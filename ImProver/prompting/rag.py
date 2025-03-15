from __future__ import annotations
from langchain.globals import set_debug

set_debug(False)


from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings

import os, json, sys
import argparse


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METADATA_PATH = "/Users/ahuja/Desktop/ImProver_rewrite/RAG/annotated/Mathlib/"


def get_database_retriever(package_name="Mathlib", number_to_retrieve=5):
    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_annotated_db"
    )
    print(database_path)
    embeddings = OllamaEmbeddings(model="llama3.2")

    database = Chroma(
        collection_name="Annotated_Mathlib_Theorems",
        persist_directory=database_path,
        embedding_function=embeddings,
    )
    return database.as_retriever(
        search_type="mmr", search_kwargs={"k": number_to_retrieve}
    )


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Retrieve related Mathlib theorems"
        )
        # parser.add_argument(
        #     "k",
        #     type=int,
        #     help="Number of documents to retrieve",
        # )
        parser.add_argument(
            "json_query",
            help='JSON string containing the query in format {"query": "your query here"}',
        )
        return parser.parse_args()

    args = parse_args()

    # Parse the JSON query
    try:
        query_data = json.loads(args.json_query)
        query = query_data.get("query", "")
        k = query_data.get("k", 5)  # Default to 5 if not provided
        imports = query_data.get("imports", None)
        if not query:
            raise ValueError("Missing 'query' field in JSON")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Use command line args to adjust retriever and prompt
    retriever = get_database_retriever(number_to_retrieve=k)

    # Define a list of sources to retrieve from
    source_paths = imports
    # [
    #     "Mathlib.RingTheory.MvPolynomial.Groebner.lean",
    #     # Add more source paths here as needed
    # ]
    source_paths = [os.path.join(METADATA_PATH, path) for path in source_paths]

    # Use $in operator to match any of the specified sources
    if imports:

        output = retriever.invoke(
            query,
            filter={"source": {"$in": source_paths}},
        )
    else:
        output = retriever.invoke(query)

    for doc in output:
        src = doc.metadata.get("source", "Unknown source")
        src = src.replace(".lean", "")

        print(f"--src: {src}")
        print(doc.page_content)
        print("<BREAK>")
