from __future__ import annotations
from langchain.globals import set_debug
# from ollama import embeddings

set_debug(False)

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os, json, shutil, copy
import subprocess, threading
import http.server


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DB_PATH = os.path.join(ROOT_PATH, ".db", ".mathlib_annotated_db")


# Processed with the ntp-toolkit repository (Deprecated)
# def annotated_thms_generator_file(
#     path_to_ntp_toolkit=os.path.join(
#         os.path.abspath(os.path.join(ROOT_PATH, os.pardir)),
#         "ntp-toolkit",
#         "Examples",
#         "mathlib",
#         "StateComments",
#     )
# ):
#     for file in os.listdir(path_to_ntp_toolkit):
#         if file.endswith(".lean"):
#             with open(os.path.join(path_to_ntp_toolkit, file), "r") as f:
#                 yield file, [f.read()]


def get_library_lean_files(
    library_name="Mathlib",
    path=os.path.join(ROOT_PATH, ".lake", "packages", "mathlib", "Mathlib"),
):
    cmd = f'find {path} -type f -name "*.lean" -print'
    files = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    ).stdout.split("\n")
    for i in range(len(files)):
        try:
            files[i] = (
                (library_name + files[i].split(library_name)[1])
                .replace("/", ".")
                .split(".lean")[0]
            )
        except IndexError:
            files[i] = ""
    return list(filter(None, files))


def save_annotated_library(
    library_name="Mathlib",
    path=os.path.join(ROOT_PATH, ".lake", "packages", "mathlib", "Mathlib"),
    style="new"
):
    modules = get_library_lean_files(library_name, path)
    # modules = ['Mathlib.Data.Finset.MulAntidiagonal', 'Mathlib.Data.Complex.Cardinality', 'Mathlib.Topology.Category.TopCat.EffectiveEpi']
    if style == "new":
        if not os.path.exists(os.path.join(ROOT_PATH, "RAG", "annotated_new", library_name)):
            os.makedirs(os.path.join(ROOT_PATH, "RAG", "annotated_new", library_name))
    else:
        if not os.path.exists(os.path.join(ROOT_PATH, "RAG", "annotated", library_name)):
            os.makedirs(os.path.join(ROOT_PATH, "RAG", "annotated", library_name))

    def annotateModule(module):
        if style == "new":
            cmd = ["lake", "exe", "extract_states", module]
            out = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                universal_newlines=True,
                cwd=ROOT_PATH,
            )
            if out.stdout:
                with open(
                    os.path.join(ROOT_PATH, "RAG", "annotated_new", library_name, f"{module}.jsonl"),
                    "w",
                ) as f:
                    f.write(out.stdout)
                print(f"Annotated {module}")
            return out.stdout
        else:
            cmd = ["lake", "exe", "StateComments", module]
            out = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                universal_newlines=True,
                cwd=ROOT_PATH,
            )
            with open(
                os.path.join(ROOT_PATH, "RAG", "annotated", library_name, f"{module}.lean"),
                "w",
            ) as f:
                f.write(out.stdout)
            print(f"Annotated {module}")
            return out.stdout

    futures = []
    with ThreadPoolExecutor() as executor:
        for module in modules:
            futures.append(executor.submit(annotateModule, module))
    # for future in futures:
    #     print(future.result())


def create_database_of_annotated(replace=False, max_docs=None, package_name="Mathlib"):
    path_to_annotated = os.path.join(ROOT_PATH, "RAG", "annotated", package_name)
    if not os.path.exists(path_to_annotated):
        print(
            f"No annotated theorems found at {path_to_annotated}. Run save_annotated_library() to generate them."
        )

    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_annotated_db"
    )

    if replace:
        if os.path.exists(database_path):
            shutil.rmtree(database_path)

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
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="riyazahuja/Improver-DeepSeek-R1-Distill-Qwen-7B_full_4096"
    # )


    vectorstore = Chroma(
        collection_name="Annotated_Mathlib_Theorems",
        persist_directory=database_path,
        embedding_function=embeddings,
    )

    docs = docs[:max_docs] if max_docs is not None else docs
    # docs = docs[:100]
    # vectorstore.add_documents(docs)

    vectorstore.add_documents(docs)
    return vectorstore



def create_database_initial_proofstate(replace=False, max_docs=None, package_name="Mathlib"):
    path_to_annotated = os.path.join(ROOT_PATH, "RAG", "annotated_new", package_name)
    if not os.path.exists(path_to_annotated):
        print(
            f"No annotated theorems found at {path_to_annotated}. Run save_annotated_library() to generate them."
        )

    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_initial_proofstate_db"
    )

    if replace:
        if os.path.exists(database_path):
            shutil.rmtree(database_path)

    # loader = DirectoryLoader(
    #     path_to_annotated, glob="**/*.lean", show_progress=True, loader_cls=TextLoader
    # )

    # docs = loader.load()

    docs = []
    for file in os.listdir(path_to_annotated):
        if file.endswith(".jsonl"):
            with open(os.path.join(path_to_annotated, file), "r") as f:
                for line in f:
                    j = json.loads(line)
                    # print(j["initialProofState"][:1000])
                    # print("\n\n----------------\n\n")
                    docs.append(Document(page_content=j["initialProofState"], metadata={"decl" : j["decl"]}))

    # embeddings = OllamaEmbeddings(model="llama3.2")
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="riyazahuja/Improver-DeepSeek-R1-Distill-Qwen-7B_full_4096"
    # )
    embeddings = HuggingFaceEmbeddings(
        model_name="hanwenzhu/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-mar13"
    )

    vectorstore = Chroma(
        collection_name="Mathlib_initial_proofstate_db",
        persist_directory=database_path,
        embedding_function=embeddings,
    )

    docs = docs[:max_docs] if max_docs is not None else docs
    # docs = docs[:100]
    # vectorstore.add_documents(docs)

    vectorstore.add_documents(docs)
    return vectorstore



def get_database_retriever_old(package_name="Mathlib", number_to_retrieve=6, filter={}):
    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_annotated_db2"
    )
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="riyazahuja/Improver-DeepSeek-R1-Distill-Qwen-7B_full_4096"
    # )
    embeddings = OllamaEmbeddings(model="llama3.2")
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="hanwenzhu/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-mar13"
    # )

    # database = Chroma(
    #     collection_name=f"Annotated_{package_name}_Theorems",
    #     # persist_directory=database_path,
    #     embedding_function=embeddings,
    # )
    database = Chroma(
        collection_name="Annotated_Mathlib_Theorems",
        persist_directory=database_path,
        embedding_function=embeddings,
    )
    return database.as_retriever(
        search_type="mmr", search_kwargs={"k": number_to_retrieve}
    )

def get_database_retriever(package_name="Mathlib", number_to_retrieve=6, filter={}):
    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_initial_proofstate_db"
    )
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="riyazahuja/Improver-DeepSeek-R1-Distill-Qwen-7B_full_4096"
    # )
    # embeddings = OllamaEmbeddings(model="llama3.2")
    embeddings = HuggingFaceEmbeddings(
        model_name="hanwenzhu/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-mar13"
    )

    # database = Chroma(
    #     collection_name=f"Annotated_{package_name}_Theorems",
    #     # persist_directory=database_path,
    #     embedding_function=embeddings,
    # )
    database = Chroma(
        collection_name="Mathlib_initial_proofstate_db",
        persist_directory=database_path,
        embedding_function=embeddings,
    )
    db = database.as_retriever(
        search_type="mmr", search_kwargs={"k": number_to_retrieve}
    )

    return db
    # def get(initialStateString):
    #     # return db.invoke(initialStateString, filter=filter).metadata["decl"]
    #     docs = database.max_marginal_relevance_search(initialStateString, k=number_to_retrieve, fetch_k=2*number_to_retrieve, filter=filter)
    #     # print(docs)
    #     return [doc.metadata["decl"] for doc in docs]
    # return get


def test_average_speed():
    import time

    start = time.time()
    get_database_retriever(replace=True, max_docs=1000)
    print(f"Time per chunk: {(time.time() - start) / 1000}")


def annotate_all_packages(project_home=ROOT_PATH):
    for package_dir in os.listdir(os.path.join(project_home, ".lake", "packages")):
        if os.path.isdir(
            os.path.join(
                ROOT_PATH, ".lake", "packages", package_dir, package_dir.title()
            )
        ):
            save_annotated_library(
                library_name=package_dir,
                path=os.path.join(ROOT_PATH, ".lake", "packages", package_dir),
            )


if __name__ == "__main__":
    # Annotate and save theorems
    # save_annotated_library()

    # Compile the database (takes a hot minute)
    create_database_initial_proofstate(replace=True)

    # Old version
    # create_database_of_annotated(replace=True)


    # Test retrieval
    # IMPORTANT: for new version of retrieval, make sure to run the "getInitialProofState" function in scripts.extract_states, and use that as your query
    # (it will still return the normal text of the theorem)
#     retriever = get_database_retriever()
#     output = retriever.invoke(
#         """α : Type u_1
# E : Type u_2
# inst✝ : NormedField E
# f : α → E
# hf : Multipliable f
# ⊢ Eq (Norm.norm (tprod fun i => f i)) (tprod fun i => Norm.norm (f i))
# """
#     )
#     print(output)
#     print(type(output))
#     for doc in output:
#         print(f"[{doc.metadata}]")
#         print(doc.page_content)
#         print("===============")
