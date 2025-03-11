from __future__ import annotations
from langchain.globals import set_debug

set_debug(False)

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import os, json, shutil, copy
import subprocess, threading
import http.server


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_PATH, ".db", ".mathlib_annotated_db")


# Processed with the ntp-toolkit repository
def annotated_thms_generator_file(
    path_to_ntp_toolkit=os.path.join(
        os.path.abspath(ROOT_PATH),
        "ntp-toolkit",
        "Examples",
        "mathlib",
        "StateComments",
    )
):
    for file in os.listdir(path_to_ntp_toolkit):
        if file.endswith(".lean"):
            with open(os.path.join(path_to_ntp_toolkit, file), "r") as f:
                yield file, [f.read()]


def create_mathlib_database(
    replace=False,
    max_docs=None,
    save_annotations=False,
    path_to_annotated=os.path.join(
        os.path.abspath(ROOT_PATH),
        "ntp-toolkit",
        "Examples",
        "mathlib",
        "StateComments",
    ),
):
    if replace:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
    print(path_to_annotated)
    loader = DirectoryLoader(
        path_to_annotated, glob="**/*.lean", show_progress=True, loader_cls=TextLoader
    )

    docs = loader.load()
    lean_splitters = [
        "\ntheorem ",
        "\nlemma ",
        "\nexample ",
        "\ndef ",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    splitter = RecursiveCharacterTextSplitter(
        separators=lean_splitters,
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
    )
    docs = splitter.split_documents(docs)
    print("Number of chunks:", len(docs))
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = Chroma(
        collection_name="Annotated_Mathlib_Theorems",
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    def embed(doc):
        return embeddings.embed_query(doc.page_content)

    docs = docs[:max_docs] if max_docs is not None else docs
    # vectorstore.add_documents(docs)
    with ProcessPoolExecutor() as executor:
        emb = executor.map(embed, docs)
    vectorstore.add_documents(docs, embeddings=emb)
    return vectorstore


def get_mathlib_retriever(number_to_retrieve=6, filter={}):
    embeddings = OllamaEmbeddings(model="llama3.2")
    database = Chroma(
        collection_name="Annotated_Mathlib_Theorems",
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )
    return database.as_retriever(
        search_type="mmr", search_kwargs={"k": number_to_retrieve, "filter": filter}
    )


def test_average_speed():
    import time

    start = time.time()
    create_mathlib_database(replace=True, max_docs=1000)
    print(f"Time per chunk: {(time.time() - start) / 1000}")


if __name__ == "__main__":
    # clone ntp-toolkit repository
    parent_dir = os.path.abspath(ROOT_PATH)
    if not os.path.exists(os.path.join(parent_dir, "ntp-toolkit")):
        subprocess.run(
            ["git", "clone", "https://github.com/cmu-l3/ntp-toolkit.git"],
            cwd=os.path.abspath(ROOT_PATH),
        )
    subprocess.run(
        [
            "python3",
            "scripts/extract_repos.py",
            "--cwd",
            os.path.join(parent_dir, "ntp-toolkit"),
            "--config",
            f"{ROOT_PATH}/RAG/config_mathlib_v4.16.json",
            "--state_comments",
        ],
        cwd=os.path.abspath(os.path.join(ROOT_PATH, "ntp-toolkit")),
    )

    # Compile the database (takes a hot minute)
    create_mathlib_database(replace=True)

    # Test retrieval
    # retriever = get_mathlib_retriever()
    # output = retriever.invoke("""elab "generalize'" h:ident " : " t:term:51 " = " x:ident : tactic => do""")
    # for doc in output:
    #     print(f"[{doc.metadata}]")
    #     print(doc.page_content)
    #     print("===============")
