from __future__ import annotations
from langchain.globals import set_debug

set_debug(False)

from langchain_core.documents import Document
from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import subprocess
import os, json

# import http.server

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_leanfile_output(cmd=("lake", "exe", "AnnotateImports")):
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

# Returns tuple of (path, [annotated_theorems])
def annotated_thms_generator_stdio():
    for line in get_leanfile_output():
        try:
            data = json.loads(line)
            yield data["filename"], data["theorems"]
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {line}")
            continue

# HTTP-based communication between Lean and Python if we ever need to employ threading, etc.

# class LeanRequestHandler(http.server.BaseHTTPRequestHandler):
#     received = []
#
#     def log_message(self, format, *args):
#         return
#
#     def do_POST(self):
#         content_length = int(self.headers.get('Content-Length', 0))
#         content_type = self.headers.get('Content-Type', '')
#
#         if content_type != 'application/json':
#             self.send_response(400)
#             self.send_header('Content-Type', 'application/json')
#             self.end_headers()
#             response = {"error": "Invalid Content-Type. Expected application/json"}
#             self.wfile.write(json.dumps(response).encode('utf-8'))
#             return
#
#         try:
#             request_body = self.rfile.read(content_length).decode('utf-8')
#             data = json.loads(request_body)
#         except json.JSONDecodeError:
#             self.send_response(400)
#             self.send_header('Content-Type', 'application/json')
#             self.end_headers()
#             response = {"error": "Invalid JSON format"}
#             self.wfile.write(json.dumps(response).encode('utf-8'))
#             return
#         response = {
#             "status": "success",
#             "message": "Data received",
#             "data": data
#         }
#
#         self.send_response(200)
#         self.send_header('Content-Type', 'application/json')
#         self.end_headers()
#         self.wfile.write(json.dumps(response).encode('utf-8'))
#         LeanRequestHandler.received.append(data)
#
# def annotated_thms_generator_http(server_class=http.server.HTTPServer, handler_class=LeanRequestHandler, port=8000):
#     server_address = ('', port)
#     httpd = server_class(server_address, handler_class)
#     # print(f'Starting httpd server on port {port}')
#     # serve on a different thread
#     server_thread = threading.Thread(target=httpd.serve_forever, daemon=True).start()
#     process = subprocess.Popen(
#         ("lake", "exe", "AnnotateTheorems"),
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#         text=True,
#         bufsize=1,
#         universal_newlines=True,
#         cwd=ROOT_PATH
#     )
#     try:
#         while True:
#             if LeanRequestHandler.received != []:
#                 for data in LeanRequestHandler.received:
#                     if "filename" in data and "theorems" in data:
#                         yield data["filename"], data["theorems"]
#                     elif "status" in data and data["status"] == "done":
#                         break
#                 LeanRequestHandler.received = []
#     except KeyboardInterrupt:
#         pass
#     finally:
#         process.kill()
#         httpd.shutdown()
#         httpd.server_close()


def make_import_database(number_to_retrieve=6, filter=None):
    embeddings = OllamaEmbeddings(model="llama3.2")
    gen = annotated_thms_generator_stdio()
    # gen = annotated_thms_generator_http()
    vectorstore = Chroma(collection_name="Annotated_Mathlib_Theorems", embedding_function=embeddings, persist_directory=DB_PATH)

    docs_buffer = []
    for (file_name, annotated_theorems) in gen:
        for thm in annotated_theorems:
            doc = Document(
                page_content=thm,
                metadata={"file": file_name}, # TODO: Add more metadata? should make bigger chunks?
            )
            docs_buffer.append(doc)
    vectorstore.add_documents(docs_buffer)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": number_to_retrieve, "filter": filter})
