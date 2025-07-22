from __future__ import annotations
import argparse
import json
import re

from langchain.globals import set_debug
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
import duckdb

set_debug(False)


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_database_retriever(prompt_id="final_final_train", number_to_retrieve=6, filter={}):
    # database_path = os.path.join(
    #     ROOT_PATH, ".db", f"{package_name.lower()}_initial_proofstate_db"
    # )
    database_path = os.path.join(
        ROOT_PATH, ".db", f"{prompt_id.lower()}_informal_retrieval_db"
    )
    if not os.path.exists(database_path):
        raise FileNotFoundError(
            f"Database {database_path} does not exist. Please ensure the database is created."
        )

    # embeddings = HuggingFaceEmbeddings(
    #     model_name="hanwenzhu/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-mar13"
    # )
    embeddings = HuggingFaceEmbeddings(
        model_name="taterowney/informal_proof_to_informal_statement_premise_selector"
    )

    database = Chroma(
        collection_name="informal_retrieval_db",
        persist_directory=database_path,
        embedding_function=embeddings,
    )
    db = database.as_retriever(
        search_type="mmr", search_kwargs={"k": number_to_retrieve}
    )

    return db

def _duckdb_escape(s):
    """
    Escape a string for use in a DuckDB query.
    """
    
    return s.replace("'", "''").replace("»", "").replace("«", "")

def process_informalization(model_output):
    informal_statement = ""
    informal_proof = ""
    if "<STATEMENT>" in model_output and "</STATEMENT>" in model_output:
        # informal_statement = re.search(r"<STATEMENT>(.*?)</STATEMENT>", model_output, re.DOTALL)
        # informal_statement = informal_statement.group(1).strip() if informal_statement else ""
        informal_statement = model_output.split("<STATEMENT>")[-1].split("</STATEMENT>")[0].strip()
    if "<PROOF>" in model_output and "</PROOF>" in model_output:
        # informal_proof = re.search(r"<PROOF>(.*?)</PROOF>", model_output, re.DOTALL)
        # informal_proof = informal_proof.group(1).strip() if informal_proof else ""
        informal_proof = model_output.split("<PROOF>")[-1].split("</PROOF>")[0].strip()
    return informal_statement, informal_proof

def add_to_db(prompt_id, k=6):
    """
    Add informal proof data to the database.
    """
    conn = duckdb.connect(os.path.join(ROOT_PATH, "prompts", prompt_id, "informal_data.duckdb"))
    assert conn is not None, "Failed to connect to the database."
        # Add rag_docs column if it doesn't exist
    try:
        conn.execute("ALTER TABLE prompts ADD COLUMN rag_docs TEXT")
    except Exception:
        # Column already exists: everything's already populated
        return

    vectordb = get_database_retriever(prompt_id=prompt_id, number_to_retrieve=k)
    assert vectordb is not None, "Failed to initialize the vector database."

    # Get all name/module pairs from the database
    # all_pairs_query = "SELECT DISTINCT name, module FROM prompts"
    # all_pairs = conn.execute(all_pairs_query).fetchall()

    # for name, module in all_pairs:
        # query = f"SELECT * FROM prompts WHERE name = '{name}' AND module = '{module}'"

        # result = conn.execute(query).fetchall()
        
        # if not result:
        #     continue  # Skip if no result found
        # columns = [col[0] for col in conn.execute("DESCRIBE prompts").fetchall()]

        # res = result[0]
        # res_as_dict = dict(zip(columns, res))
    for name, module, text, messages in conn.execute("SELECT name, module, generated_text, messages FROM prompts").fetchall():
        src = messages[0]["content"].split("<FORMAL>")[-1].split("</FORMAL>")[0].strip()
        _, informal_proof = process_informalization(text)
        if not informal_proof:
            rag_docs_str = ""
        else:
            docs = vectordb.invoke(informal_proof, k=k)

            # Put each retrieved theorem's formal code in a string, with informal statement as a docstring
            rag_docs_str = json.dumps(["/--" + doc.page_content + "-/" + doc.metadata.get("source", "") + "\n<BREAK>\n" if hasattr(doc, 'page_content') else str(doc) for doc in docs])
        
        # Update the database with the retrieved documents
        update_query = f"UPDATE prompts SET rag_docs = ? WHERE name = '{name}' AND module = '{module}'"
        try:
            conn.execute(update_query, [_duckdb_escape(rag_docs_str)])
        except duckdb.duckdb.ParserException as e:
            print(f"Error updating {name} in {module}: {e}")
            print(_duckdb_escape(rag_docs_str))
            continue

    conn.close()

def get_rag_string(prompt_id, name, module, k):
    """
    Retrieve the RAG string for a specific name and module.
    """
    conn = duckdb.connect(os.path.join(ROOT_PATH, "prompts", prompt_id, "informal_data.duckdb"))
    assert conn is not None, "Failed to connect to the database."

    columns = [col[0] for col in conn.execute("DESCRIBE prompts").fetchall()]
    if "rag_docs" not in columns:
        add_to_db(prompt_id, k=k)

    query = f"SELECT rag_docs FROM prompts WHERE name = '{_duckdb_escape(name)}' AND module = '{module}'"
    result = conn.execute(query).fetchone()

    conn.close()

    if result:
        return result  # Return the rag_docs string
    else:
        return ""  # No RAG docs found


if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser(description="Retrieve documents from the database.")
    # parser.add_argument("--prompt_id", type=str, required=True, help="ID of the prompt to retrieve documents for.")
    # parser.add_argument("--module", type=str, required=True, help="Module name to retrieve documents for.")
    # parser.add_argument("--name", type=str, required=True, help="Name to retrieve documents for.")
    # parser.add_argument("--k", type=int, default=6, help="Number of documents to retrieve.")

    # args = parser.parse_args()
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
        prompt_id = query_data.get("prompt_id", "final_final_train")
        name = query_data.get("name", "")
        module = query_data.get("module", "")
        if not prompt_id:
            raise ValueError("Missing 'prompt_id' field in JSON")
        if not query:
            raise ValueError("Missing 'query' field in JSON")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    source_paths = imports if imports else []
    if imports:
        filter={"source": {"$in": source_paths}}
    else:
        filter={} #TODO


    print(get_rag_string(prompt_id, name, module, k))