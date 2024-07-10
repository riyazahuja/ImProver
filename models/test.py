from __future__ import annotations
from typing import Iterator
from langchain.globals import set_verbose,set_debug
set_debug(False)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from evaluate.metrics import *
from models.prompt import *
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class RepoLoader(BaseLoader):
    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that flattens the repo data and combines metadata.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        files = self.repo.files
        location_data = {"url": self.repo.url, "commit":self.repo.commit} if self.repo.type == 'git' else {"path":self.repo.path}
        for f in files:
            annotated = type(f) == AnnotatedFile
            if annotated:
                for thm in f.theorems:
                    metadata={
                            "repo_name": self.repo.name,
                            "lean_version": self.repo.version,
                            "repo_type": self.repo.type,
                            "file_name": f.file_name,
                            "file_path": f.file_path,
                            "file_type": f.file_type,
                            "decl": thm.decl,
                            "context": thm.context
                        }
                    metadata.update(location_data)
                    
                    yield Document(
                        page_content= parseTheorem(thm,
                                                   context=False,
                                                   annotation=True,
                                                   prompt=False
                                                    ),
                        metadata=metadata
                    )
            else:
                metadata = {
                        "repo_name": self.repo.name,
                        "lean_version": self.repo.version,
                        "repo_type": self.repo.type,
                        "file_name": f.file_name,
                        "file_path": f.file_path,
                        "file_type": f.file_type
                    }
                metadata.update(location_data)
                
                yield Document(page_content=f.contents, metadata=metadata)
        



def get_content_vs(repo:Repo):
    lean_splitters = ['\nnamespace ','\ntheorem ','\nlemma ','\nexample ','\nstructure ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']

    lean_splitter = RecursiveCharacterTextSplitter(separators=lean_splitters,
                                                    chunk_size=1000,
                                                    chunk_overlap=200,
                                                    add_start_index=True,
                                                    length_function=len,
                                                    is_separator_regex=False
                                                )

    docs = RepoLoader(repo).lazy_load()
    split = lean_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=split, embedding=OpenAIEmbeddings(),persist_directory=os.path.join(root_path,'.chroma_db'))
    
    return vectorstore


def get_TPiL4_vs(path = os.path.join(root_path,'TPiL4')):
    print(f"TPIL\n\n\n\n{path}")
    files = []
    for fp in os.listdir(path):
        if fp.endswith('.md'):
            files.append(fp)
    print(f'{files} | {len(files)}')
    docs=[]
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"),("###", "Header 3"),("####", "Header 4")]
        # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    # Char-level splits
    chunk_size = 1000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap, 
                                                add_start_index=True,
                                                length_function=len,
                                                is_separator_regex=False
                                                )

    for name in files:
        fp = os.path.join(path,name)
        with open(fp,'r') as f:
            text = f.read()
        md_header_splits = markdown_splitter.split_text(text)
        header_splits_named = []
        for doc in md_header_splits:
            new = doc.metadata
            new.update({'file':name})
            header_splits_named.append(Document(page_content=doc.page_content,metadata=new))
        
        splits = text_splitter.split_documents(header_splits_named)
        docs.extend(splits)

    
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(),persist_directory=os.path.join(root_path,'.chroma_db'))
    return vectorstore






def get_retriever(vectorstore=None,k=6,filterDB = {}, persist_dir = os.path.join(root_path,'.chroma_db')):

    if vectorstore is None:
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k, 'filter': filterDB})
    else:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k,'filter': filterDB})
    return retriever




def prompt_structured(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[],retries=6,annotation=True) -> Theorem:
    k=4
    retriever = get_retriever(k=k)

    model = ChatOpenAI(model=model)#,temperature=0)

    class trimmedTheorem(BaseModel):
        decl : str = Field(description="Theorem declaration (does not include \":= by\")")
        proof : List[Union[str,trimmedTheorem]] = Field(..., description="Sequence of proofsteps for full proof of theorem. Each proofstep is one line/tactic in a tactic proof (str) or a subtheorem/sublemma/subproof in the format of (trimmedTheorem)")

    parser = PydanticOutputParser(pydantic_object= trimmedTheorem)

    actual_prompt=metric.prompt.replace('{',r'{{').replace('}',r'}}')
    prev_data = parse_prev_data(prev_data)
    

    prompt = ChatPromptTemplate.from_messages([
        ('system',f'''{actual_prompt}
         You will be given the proof context (i.e. the lean file contents/imports leading up to the theorem declaration) wrapped by <CONTEXT>...</CONTEXT>.
         {f"You will be given the previous {len(prev_data)} input/output pairs as well as their metric ({metric.name}) score and correctness score, as well as any error messages, for your reference to improve upon. Each of these previous results will be wrapped with <PREV I=0></PREV I=0>,...,<PREV I={len(prev_data)-1}></PREV I={len(prev_data)-1}>, with I={len(prev_data)-1} being the most recent result." if len(prev_data)!= 0 else ""}
         Remember to use lean 4 syntax, which has significant changes from the lean 3 syntax. You will be given {k} documents relevant to the current theorem {"and most recent error messages" if len(prev_data)!= 0 else ""} to refer to for tricky and important syntax. Each of these documents will be wrapped with <SYNTAX_DOC>...</SYNTAX_DOC>.
         Output in a proof tree format that aligns with the pydantic output parsing object schema that splits off subproofs and subtheorems.
         {"You will be given the tactic states as comments for reference." if annotation else ""} The current theorem will be wrapped in <CURRENT>...</CURRENT>
         '''),
         ('system','{format_instructions}'),
         ('human','<CONTEXT>\n{context}\n</CONTEXT>'),
         ("placeholder", "{prev_results}"),
         ("placeholder", "{syntax_docs}"),
         ('human','<CURRENT>\n{theorem}\n</CURRENT>')
    ])

    def format_docs(docs):
        return [('human',f'<SYNTAX_DOC>\nfilename: {doc.metadata["file"]}\nContent:\n{doc.page_content}\n</SYNTAX_DOC>') for doc in docs if 'file' in doc.metadata.keys()]#"\n\n=======================\n".join(f"filename: {doc.metadata['file']}\nContent:\n{doc.page_content}" for doc in docs if 'file' in doc.metadata.keys())
    


    chain = (RunnablePassthrough().assign(format_instructions=lambda _: parser.get_format_instructions())
             | prompt
             | model
             | parser)
    
    @retry(
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
    wait=wait_random_exponential(multiplier=1, max=60),
    )
    def invoke_throttled(chain,config):
        return chain.invoke(config)
    
    output=invoke_throttled(chain,{"context" : thm.context, 'prev_results' : prev_data, 'theorem':parseTheorem(thm,annotation=annotation,context=False)})
    
    def coerce_PS(step):
        ProofStep.update_forward_refs()
        if type(step) == str:
            return ProofStep(tactic=step)
        return ProofStep(tactic=coerce_trimmedThm(step))

    def coerce_trimmedThm(curr):
        return Theorem(decl=curr.decl,declID=thm.declID,src=thm.src,leanFile=thm.leanFile,context=thm.context,proof=[coerce_PS(step) for step in curr.proof],project_path=thm.project_path)
     
    final = coerce_trimmedThm(output)
    
    return final







def prompt_rag(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[],retries=6,annotation=True) -> Theorem:

    retriever = get_retriever(k=4)

#    print("retriever recieved")
    model = ChatOpenAI(model=model,temperature=0)

    class trimmedTheorem(BaseModel):
        decl : str = Field(description="Theorem declaration.")
        proof : List[Union[str,trimmedTheorem]] = Field(..., description="Sequence of proofsteps for full proof of theorem. Each proofstep is one line/tactic in a tactic proof (str) or a subtheorem/sublemma/subproof in the format of (trimmedTheorem)")

    # Define the output parser
    parser = PydanticOutputParser(pydantic_object= trimmedTheorem)


    # Define the prompt template
    rag_prompt = "Here is some retrieved lean documentation content relevent for use in your new proof."
    
    actual_prompt=metric.prompt.replace('{',r'{{').replace('}',r'}}')
    prev_data_text = parse_prev_data(prev_data).replace('{',r'{{').replace('}',r'}}')
    # Define the prompt template
    prompt = PromptTemplate(
        template=actual_prompt + "{data_str}\n\n"+prev_data_text+'\n'+rag_prompt+'\n{rag}'+"format:\n{format_instructions}",
        input_variables=["data_str","rag"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )



    def format_docs(docs):
        return "\n\n=======================\n".join(f"filename: {doc.metadata['file']}\nContent:\n{doc.page_content}" for doc in docs if 'file' in doc.metadata.keys())


    chain = (
        {"rag": retriever | format_docs, "data_str": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    @retry(
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
    wait=wait_random_exponential(multiplier=1, max=60),
    )
    def invoke_throttled(chain,data):
        return chain.invoke(data)
    
    output=invoke_throttled(chain,parseTheorem(thm,annotation=annotation,prompt=True))

    def coerce_PS(step):
        ProofStep.update_forward_refs()
        if type(step) == str:
            return ProofStep(tactic=step)
        return ProofStep(tactic=coerce_trimmedThm(step))

    def coerce_trimmedThm(curr):
        return Theorem(decl=curr.decl,declID=thm.declID,src=thm.src,leanFile=thm.leanFile,context=thm.context,proof=[coerce_PS(step) for step in curr.proof],project_path=thm.project_path)
     
    final = coerce_trimmedThm(output)
    
    #print(f'Running:\n{parseTheorem(thm,annotation=True)}\n')
    #thm = Theorem(decl=thm.decl,declID=thm.declID, proof=proof, leanFile=thm.leanFile, src=thm.src, context = thm.context)
    return final

retriever = get_retriever(k=3)

model = ChatOpenAI(model='gpt-4-turbo')

parser = JsonOutputParser()

template = ChatPromptTemplate.from_messages([('system','Your name is {name}.'),
                                             ('system','Your job is two answer two questions, and output the answer to each question in a single json output. You may be provided some documents that may assist in answering these questions, each wrapped in <DOC>...</DOC>.\n{format}'),
                                             ('placeholder','{docs}'),
                                             ('human','Question 1: {user_q1}\nQuestion 2: {user_q2}')])

#retriever = get_retriever(k=3)
#docs = retriever.invoke('What is the syntax for the "case" tactic?')
#def format_docs(docs):
#        return "\n\n=======================\n".join(f"filename: {doc.metadata.get('file','Unknown')}\nContent:\n{doc.page_content}" for doc in docs)

#print(format_docs(docs))

def format_docs(docs):
    output= []
    for doc in docs:
        msg = ('human',f'<DOC>\n{doc.page_content}\n</DOC>')
        output.append(msg)
    return output

chain = (RunnablePassthrough().assign(format=lambda _: parser.get_format_instructions(), docs = lambda x : format_docs(retriever.invoke(x['user_q2'])))# | retriever | format_docs)
             | template
             | model
             | parser)



output = chain.invoke({'name':'Bob','user_q1':'What is your name?','user_q2':'What is the syntax for the "case" tactic?'})
print(output)