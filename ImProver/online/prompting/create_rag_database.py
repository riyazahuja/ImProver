from __future__ import annotations
import argparse
import json
import re
import glob

from langchain.globals import set_debug
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
set_debug(False)

import os
import duckdb



ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def _repair_db_from_parquet():

    conn = duckdb.connect(os.path.join(ROOT_PATH, "prompts", "final_final_train", "informal_data.duckdb"))
    assert conn is not None, "Failed to connect to the database."

    # Get all parquet files in the informal_data directory
    parquet_pattern = os.path.join(ROOT_PATH, "prompts", "final_final_train", "informal_data", "*.parquet")
    parquet_files = glob.glob(parquet_pattern)
    
    if not parquet_files:
        print(f"No parquet files found at {parquet_pattern}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to load")
    
    # Load all parquet files into a single table called 'prompts'
    # First, create the table from the first parquet file
    first_file = parquet_files[0]
    print(f"Creating table from {first_file}")
    conn.execute(f"CREATE TABLE IF NOT EXISTS prompts AS SELECT * FROM '{first_file}'")
    
    # Then insert data from remaining parquet files
    for parquet_file in parquet_files[1:]:
        print(f"Loading {parquet_file}")
        conn.execute(f"INSERT INTO prompts SELECT * FROM '{parquet_file}'")
    
    # Get count of loaded records
    result = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()
    print(f"Successfully loaded {result[0]} records into the database")
    
    conn.close()

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

def create_vectordb(prompt_id="final_final_train"):
    """
    Create a vector database from the informal data in the specified prompt_id.
    """

    conn = duckdb.connect(os.path.join(ROOT_PATH, "prompts", prompt_id, "informal_data.duckdb"))
    assert conn is not None, "Failed to connect to the database."

    embeddings = HuggingFaceEmbeddings(
        model_name="taterowney/informal_proof_to_informal_statement_premise_selector"
    )

    database_path = os.path.join(
        ROOT_PATH, ".db", f"{prompt_id.lower()}_informal_retrieval_db"
    )

    database = Chroma(
        collection_name="informal_retrieval_db",
        persist_directory=database_path,
        embedding_function=embeddings,
    )


    for name, module, text, messages in conn.execute("SELECT name, module, generated_text, messages FROM prompts").fetchall():
        src = messages[0]["content"].split("<FORMAL>")[-1].split("</FORMAL>")[0].strip()
        informal_statement, informal_proof = process_informalization(text)
        if not informal_statement:
            continue

        # Create a document for each entry
        doc = Document(
            metadata={
                "name": name,
                "module": module,
                "source": src,
            },
            page_content=informal_statement
        )
        database.add_documents([doc])

    conn.close()


if __name__ == "__main__":
    create_vectordb()
