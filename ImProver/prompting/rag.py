from __future__ import annotations
from langchain.globals import set_debug

set_debug(False)

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os, json, shutil, copy
import subprocess, threading
import argparse
import http.server


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_database_retriever(package_name="Mathlib", number_to_retrieve=6, filter={}):
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


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Retrieve related Mathlib theorems"
        )
        parser.add_argument(
            "k",
            type=int,
            help="Number of documents to retrieve",
        )
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
        if not query:
            raise ValueError("Missing 'query' field in JSON")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Use command line args to adjust retriever and prompt
    retriever = get_database_retriever(number_to_retrieve=args.k)

    output = retriever.invoke(query)
    print(output)
    print(type(output))
    for doc in output:
        print(f"[{doc.metadata}]")
        print(doc.page_content)
        print("===============")
