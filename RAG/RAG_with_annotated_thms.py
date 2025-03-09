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
        pass
    finally:
        process.kill()

class LeanRequestHandler(http.server.BaseHTTPRequestHandler):
    received = []

    def log_message(self, format, *args):
        return

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        content_type = self.headers.get('Content-Type', '')

        if content_type != 'application/json':
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"error": "Invalid Content-Type. Expected application/json"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        try:
            request_body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"error": "Invalid JSON format"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return
        response = {
            "status": "success",
            "message": "Data received",
            "data": data
        }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
        LeanRequestHandler.received.append(data)

def annotated_thms_generator_file(path_to_ntp_toolkit=os.path.join(os.path.abspath(os.path.join(ROOT_PATH, os.pardir)), "ntp-toolkit", "Examples", "mathlib", "StateComments")):
    for file in os.listdir(path_to_ntp_toolkit):
        if file.endswith(".lean"):
            with open(os.path.join(path_to_ntp_toolkit, file), "r") as f:
                yield file, [f.read()]

def annotated_thms_generator_http(server_class=http.server.HTTPServer, handler_class=LeanRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    # print(f'Starting httpd server on port {port}')
    # serve on a different thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True).start()
    process = subprocess.Popen(
        ("lake", "exe", "AnnotateTheorems"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=ROOT_PATH
    )
    try:
        while True:
            if LeanRequestHandler.received != []:
                for data in LeanRequestHandler.received:
                    if "filename" in data and "theorems" in data:
                        yield data["filename"], data["theorems"]
                    elif "status" in data and data["status"] == "done":
                        break
                LeanRequestHandler.received = []
    except KeyboardInterrupt:
        pass
    finally:
        process.kill()
        httpd.shutdown()
        httpd.server_close()

# Returns tuple of (path, [annotated_theorems])
def annotated_thms_generator_stdio():
    for line in get_leanfile_output():
        try:
            data = json.loads(line)
            yield data["filename"], data["theorems"]
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {line}")
            continue

# def create_mathlib_database(replace=False, path_to_mathlib=os.path.join(ROOT_PATH, ".lake", "packages", "mathlib", "Mathlib"), max_docs=None, save_annotations=False):
#     if replace:
#         if os.path.exists(DB_PATH):
#             shutil.rmtree(DB_PATH)
#     if save_annotations:
#         f = open("mathlib_annotated.txt", "w")
#     lean_splitters = [
#         "\ntheorem ",
#         "\nlemma ",
#         "\nexample ",
#         "\ndef ",
#         "\n\n",
#         "\n",
#         " ",
#         "",
#     ]
#     splitter = RecursiveCharacterTextSplitter(
#         separators=lean_splitters,
#         chunk_size=1000,
#         chunk_overlap=200,
#         add_start_index=True,
#         length_function=len,
#         is_separator_regex=False,
#         keep_separator=True,
#     )
#     embeddings = OllamaEmbeddings(model="llama3.2")
#     # gen = annotated_thms_generator_stdio()
#     # gen = annotated_thms_generator_http()
#     gen = annotated_thms_generator_file()
#     vectorstore = Chroma(collection_name="Annotated_Mathlib_Theorems", embedding_function=embeddings, persist_directory=DB_PATH)
#
#     i=0
#     docs_buffer = []
#     futures = []
#     with ProcessPoolExecutor() as executor:
#         for (file_name, annotated_theorems) in gen:
#             if save_annotations:
#                 f.write(f"{{filename: {file_name}, theorems: {annotated_theorems}}}\n")
#             for thm in annotated_theorems:
#                 for chunk in splitter.split_text(thm):
#                     doc = Document(
#                         page_content=chunk,
#                         metadata={"file": file_name}, # TODO: Add more metadata? should make bigger chunks?
#                     )
#                     # vectorstore.add_documents([doc])
#
#                     docs_buffer.append(doc)
#                     # if len(docs_buffer) >= 500:
#                     #     futures.append(executor.submit(lambda docs : vectorstore.add_documents(docs), copy.deepcopy(docs_buffer)))
#                     #
#                     #     # vectorstore.add_documents(docs_buffer)
#                     #     docs_buffer = []
#                     i+=1
#                     # if i % 500 == 0:
#                     #     print(f"Processed {i} chunks")
#                     if (max_docs is not None and i >= max_docs):
#                         break
#                 if (max_docs is not None and i >= max_docs):
#                     break
#             if (max_docs is not None and i >= max_docs):
#                 break
#                         # if docs_buffer != []:
#                         #     chunked_docs = [docs_buffer[i * 500 : (i + 1) * 500] for i in range((len(docs_buffer) + 500 - 1) // 500)]
#                         #     futures.extend([executor.submit(lambda docs : vectorstore.add_documents(docs), copy.deepcopy(chunk)) for chunk in chunked_docs])
#                         #     print(wait(futures, return_when=ALL_COMPLETED, timeout=None))
#                         #     # vectorstore.add_documents(docs_buffer)
#                         # return vectorstore
#         if docs_buffer != []:
#             chunked_docs = [docs_buffer[i * 500 : (i + 1) * 500] for i in range((len(docs_buffer) + 500 - 1) // 500)]
#             futures.extend([executor.submit(lambda docs : vectorstore.add_documents(docs), copy.deepcopy(chunk)) for chunk in chunked_docs])
#             print(wait(futures, return_when=ALL_COMPLETED, timeout=None))
#     # if docs_buffer != []:
#     #     vectorstore.add_documents(docs_buffer)
#     if save_annotations:
#         f.close()
#     return vectorstore

def create_mathlib_database(replace=False, max_docs=None, save_annotations=False, path_to_annotated=os.path.join(os.path.abspath(os.path.join(ROOT_PATH, os.pardir)), "ntp-toolkit", "Examples", "mathlib", "StateComments")):
    if replace:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)

    # file_contents = ""
    # for filename in os.listdir(path_to_annotated):
    #     if filename.endswith(".lean"):
    #         with open(os.path.join(path_to_annotated, filename), "r") as f:
    #             file_contents += f.read()
    loader = DirectoryLoader(path_to_annotated,
                             glob="**/*.lean",
                             show_progress=True,
                             loader_cls=TextLoader
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
    vectorstore = Chroma(collection_name="Annotated_Mathlib_Theorems", persist_directory=DB_PATH, embedding_function=embeddings)

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
    database = Chroma(collection_name="Annotated_Mathlib_Theorems", persist_directory=DB_PATH, embedding_function=embeddings)
    return database.as_retriever(search_type="mmr", search_kwargs={"k": number_to_retrieve, "filter": filter})

def test_average_speed():
    import time
    start = time.time()
    create_mathlib_database(replace=True, max_docs=1000)
    print(f"Time per chunk: {(time.time() - start) / 1000}")

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

