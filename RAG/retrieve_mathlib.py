from __future__ import annotations
from langchain.globals import set_debug
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os

set_debug(False)


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_database_retriever(package_name="Mathlib", number_to_retrieve=6, filter={}):
    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_initial_proofstate_db"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="hanwenzhu/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-mar13"
    )

    database = Chroma(
        collection_name="Mathlib_initial_proofstate_db",
        persist_directory=database_path,
        embedding_function=embeddings,
    )
    db = database.as_retriever(
        search_type="mmr", search_kwargs={"k": number_to_retrieve}
    )

    return db


if __name__ == "__main__":

    retriever = get_database_retriever()
    output = retriever.invoke(
        """α : Type u_1
E : Type u_2
inst✝ : NormedField E
f : α → E
hf : Multipliable f
⊢ Eq (Norm.norm (tprod fun i => f i)) (tprod fun i => Norm.norm (f i))
"""
    )
    print(output)
    print(type(output))
    for doc in output:
        print(f"[{doc.metadata['decl']}]")
        print(doc.page_content)
        print("===============")
