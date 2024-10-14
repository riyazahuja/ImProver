from __future__ import annotations
from langchain.globals import set_debug

set_debug(False)
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


"""
Retrieval stuff
"""


def get_mathlib_vs(
    path=os.path.join(root_path, ".lake", "packages", "mathlib", "Mathlib")
):
    dirs = [""]
    all_files = []
    for dir in dirs:
        for root, _, files in os.walk(os.path.join(path, dir)):
            for file in files:
                fp = os.path.join(root, file)
                if fp.endswith(".lean") and fp not in all_files:
                    all_files.append(fp)

    files = all_files

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
    lean_splitter = RecursiveCharacterTextSplitter(
        separators=lean_splitters,
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
    )

    docs = []
    for fp in files:
        name = os.path.relpath(fp, path)

        with open(fp, "r") as f:
            text = f.read()

        splits = lean_splitter.split_text(text)

        for doc in splits:
            new = {}  # doc.metadata
            new.update({"file": name})
            docs.append(Document(page_content=doc, metadata=new))

    n = 5000
    docs_chunked = [docs[i * n : (i + 1) * n] for i in range((len(docs) + n - 1) // n)]

    embeddings = OpenAIEmbeddings(show_progress_bar=True)

    vectorstore = Chroma.from_documents(
        documents=docs_chunked[0],
        embedding=embeddings,
        persist_directory=os.path.join(root_path, ".db", ".mathlib_chroma_db"),
    )
    print(f"==========Chunk 1/{len(docs_chunked)} successfully downloaded==========")

    # for i in range(1,len(docs_chunked)):
    num_done = 1

    def add_vs(i):
        vectorstore.add_documents(docs_chunked[i])
        print(
            f"==========Chunk {num_done}/{len(docs_chunked)} successfully downloaded=========="
        )
        num_done = num_done + 1
        return None

    with ThreadPoolExecutor(max_workers=len(docs_chunked) / 4) as executor:
        future_to_thm = [
            executor.submit(add_vs, i) for i in range(1, len(docs_chunked))
        ]

    return vectorstore


def get_TPiL4_vs(path=os.path.join(root_path, ".db", "src", "TPiL4")):
    files = []
    for fp in os.listdir(path):
        if fp.endswith(".md"):
            files.append(fp)
    docs = []
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    # Char-level splits
    chunk_size = 1000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        length_function=len,
        is_separator_regex=False,
    )

    for name in files:
        fp = os.path.join(path, name)
        with open(fp, "r") as f:
            text = f.read()
        md_header_splits = markdown_splitter.split_text(text)
        header_splits_named = []
        for doc in md_header_splits:
            new = doc.metadata
            new.update({"file": name})
            header_splits_named.append(
                Document(page_content=doc.page_content, metadata=new)
            )

        splits = text_splitter.split_documents(header_splits_named)
        docs.extend(splits)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=os.path.join(root_path, ".db", ".TPiL_chroma_db"),
    )
    return vectorstore


def get_metric_vs(examples, name):
    docs = [
        Document(page_content=f"Input:\n{ex['input']}\n\nOutput:\n{ex['output']}")
        for ex in examples
    ]
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=os.path.join(
            root_path, ".db", "metrics", f".{name}_chroma_db"
        ),
    )
    return vectorstore


def get_retriever(
    vectorstore=None,
    k=6,
    filterDB={},
    persist_dir=os.path.join(root_path, ".db", ".chroma_db"),
):
    if vectorstore is None:
        vectordb = Chroma(
            persist_directory=persist_dir, embedding_function=OpenAIEmbeddings()
        )
        retriever = vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "filter": filterDB}
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "filter": filterDB}
        )
    return retriever


if __name__ == "__main__":

    get_mathlib_vs()
#     db = Chroma(
#         persist_directory=os.path.join(root_path, ".db/.mathlib_chroma_db"),
#         embedding_function=OpenAIEmbeddings(),
#     )

#     retriever = get_retriever(
#         k=3, persist_dir=os.path.join(root_path, ".db/.mathlib_chroma_db")
#     )
#     output = retriever.invoke(
#         """theorem orrr (P Q :Prop):P∨ Q → Q∨ P:= by
#   intro hpq
#   rcases hpq with hp|hq
#   . right
#     exact hp
#   . left
#     exact hq"""
#     )
#     for doc in output:
#         print(f"[{doc.metadata}]")
#         print(doc.page_content)
#         print("===============")
