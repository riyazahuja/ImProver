from __future__ import annotations
from langchain.globals import set_debug

set_debug(False)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ProcessPoolExecutor
import os, json, shutil
import subprocess

# TODO:
# - Does it work with definitions, proof terms, etc.?
# - Parallelism
# - Amount of goals to be cut off

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_PATH, ".db", ".mathlib_annotated_db")


def get_leanfile_output(cmd=("lake", "exe", "AnnotateTheorems")):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=ROOT_PATH
    )

    try:
        for line in iter(process.stdout.readline, ''):
            yield line.strip()
        process.stdout.close()
        process.wait()  # Ensure process completes

    except KeyboardInterrupt:
        process.kill()

# Returns tuple of (path, [annotated_theorems])
def annotated_thms_generator():
    # for i in range(10):
    #     yield "./.lake/packages/mathlib/Mathlib/Analysis/Complex/RealDeriv.lean", ["theorem real_tendsto_real ..."]
    for line in get_leanfile_output():
        try:
            data = json.loads(line)
            yield data["filename"], data["theorems"]
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {line}")
            continue

def create_mathlib_database(replace=False, path_to_mathlib=os.path.join(ROOT_PATH, ".lake", "packages", "mathlib", "Mathlib"), max_docs=None):
    if replace:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OllamaEmbeddings(model="llama3.2")
    gen = annotated_thms_generator()
    vectorstore = Chroma(collection_name="Annotated_Mathlib_Theorems", embedding_function=embeddings, persist_directory=DB_PATH)

    i=0
    docs_buffer = []
    with ProcessPoolExecutor() as executor:
        for (file_path, annotated_theorems) in gen:
            # print(file_path, annotated_theorems)
            relative_path = os.path.relpath(file_path, path_to_mathlib)
            for thm in annotated_theorems:
                # for chunk in splitter.split(thm):
                doc = Document(
                    page_content=thm,
                    metadata={"file": relative_path}, # TODO: Add more metadata? should make bigger chunks?
                )
                # vectorstore.add_documents([doc], ids=[str(uuid4())])
                # vectorstore.add_documents([doc])

                docs_buffer.append(doc)
                if len(docs_buffer) > 10:
                    vectorstore.add_documents(docs_buffer)
                    docs_buffer = []
                    # executor.submit(lambda docs : vectorstore.add_document(docs), copy.deepcopy(docs))
                i+=1
                if (max_docs is not None and i >= max_docs):
                    if docs_buffer != []:
                        vectorstore.add_documents(docs_buffer)
                    return vectorstore
    if docs_buffer != []:
        vectorstore.add_documents(docs_buffer)
    return vectorstore

def get_mathlib_retriever(number_to_retrieve=6, filter={}):
    embeddings = OllamaEmbeddings(model="llama3.2")
    database = Chroma(collection_name="Annotated_Mathlib_Theorems", persist_directory=DB_PATH, embedding_function=embeddings)
    return database.as_retriever(search_type="mmr", search_kwargs={"k": number_to_retrieve, "filter": filter})

def test_average_speed():
    import time
    start = time.time()
    create_mathlib_database(replace=True, max_docs=100)
    print(f"Time per theorem: {(time.time() - start) / 100}")

if __name__ == '__main__':
    # Compile the database (takes a hot minute)
    # create_mathlib_database(replace=True)

    # Test retrieval
    # retriever = get_mathlib_retriever()
    # output = retriever.invoke("""elab "generalize'" h:ident " : " t:term:51 " = " x:ident : tactic => do""")
    # for doc in output:
    #     print(f"[{doc.metadata}]")
    #     print(doc.page_content)
    #     print("===============")

    # Test speed
    test_average_speed()

    # for l in get_leanfile_output():
    #     print(l)
