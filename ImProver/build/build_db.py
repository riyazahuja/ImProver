from __future__ import annotations
import argparse
import json
import re
import glob
import shutil
from langchain.globals import set_debug
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
set_debug(False)

import os
import duckdb



ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def process_informalization(model_output):
    informal_statement = ""
    informal_proof = ""
    if "<STATEMENT>" in model_output and "</STATEMENT>" in model_output:
        informal_statement = re.search(r"<STATEMENT>(.*?)</STATEMENT>", model_output, re.DOTALL)
        informal_statement = informal_statement.group(1).strip() if informal_statement else ""
    if "<PROOF>" in model_output and "</PROOF>" in model_output:
        informal_proof = re.search(r"<PROOF>(.*?)</PROOF>", model_output, re.DOTALL)
        informal_proof = informal_proof.group(1).strip() if informal_proof else ""
    return informal_statement, informal_proof

def main(args):
    """
    Create a vector database from the informal data in the specified prompt_id.
    """

    conn = duckdb.connect(os.path.join(ROOT_PATH, "rag", args.rag_id, "informal_data.duckdb"))
    assert conn is not None, "Failed to connect to the database."

    embeddings = HuggingFaceEmbeddings(
        model_name="taterowney/informal_proof_to_informal_statement_premise_selector"
    )

    database_path = os.path.join(
        ROOT_PATH, "rag", args.rag_id, f"informal_retrieval_db"
    )
    
    if os.path.exists(database_path):
        shutil.rmtree(database_path)
    os.makedirs(database_path, exist_ok=True)
    database = Chroma(
        collection_name="informal_retrieval_db",
        persist_directory=database_path,
        embedding_function=embeddings,
    )


    
    for name, module, text, informal_statement, informal_proof in conn.execute("SELECT name, module, text, informal_statement, informal_proof FROM informal_data").fetchall():
        # src = messages[0]["content"].split("<FORMAL>")[-1].split("</FORMAL>")[0].strip()
        # informal_statement, informal_proof = process_informalization(text)
        # if not informal_statement:
        #     continue
        content = text
        if informal_statement and informal_proof:
            content = f"{informal_statement}\n\n{informal_proof}"
        elif informal_statement:
            content = informal_statement
        

        # Create a document for each entry
        doc = Document(
            metadata={
                "name": name,
                "module": module,
                "formal": text,
                "informal_statement": informal_statement,
                "informal_proof": informal_proof
            },
            page_content=content
        )
        database.add_documents([doc])

    conn.close()

def get_parser():
    parser = argparse.ArgumentParser(description="Build RAG database")
    parser.add_argument("rag_id", type=str)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
