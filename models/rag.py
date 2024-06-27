# given (Annotated?)Theorem and Repo, parse Repo into documents and use document 
# splitter to split each file in repo into theorems (if lean) and standard markdown split (if md)
# Then embed into vector space and save for future vector search queries? This processing will probably want to be done in the build script?
from __future__ import annotations
from typing import Iterator

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
                            #"proof": [step.tactic for step in thm.proof],
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
        
        # for dependency in self.repo.dependencies:
        #     loader = RepoLoader(dependency)
        #     data = loader.lazy_load()
        #     for doc in data:
        #         yield doc
        

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






def get_retriever(vectorstore=None,k=6,persist_dir = os.path.join(root_path,'.chroma_db')):

    if vectorstore is None:
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    else:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever




def prompt_structured(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[],retries=6) -> Theorem:

    model = ChatOpenAI(model=model,temperature=0)

    class trimmedTheorem(BaseModel):
        decl : str = Field(description="Theorem declaration")
        proof : List[Union[str,trimmedTheorem]] = Field(..., description="Sequence of proofsteps for full proof of theorem. Each proofstep is one line/tactic in a tactic proof (str) or a subtheorem/sublemma/subproof in the format of (trimmedTheorem)")

    # Define the output parser
    parser = PydanticOutputParser(pydantic_object= trimmedTheorem)

    actual_prompt=metric.prompt.replace('{',r'{{').replace('}',r'}}')
    prev_data_text = parse_prev_data(prev_data).replace('{',r'{{').replace('}',r'}}')
    # Define the prompt template
    prompt = PromptTemplate(
        template=actual_prompt + "{data_str}\n\n"+prev_data_text+'\n'+"{format_instructions}",
        input_variables=["data_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the chain
    chain = prompt | model | parser

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(retries))
    def invoke_throttled(chain,config):
        return chain.invoke(config)
    
    output=invoke_throttled(chain,{"data_str" : parseTheorem(thm,annotation=True,prompt=True)})
    
         
    decl,pf = output.decl, output.proof
    print(f'DECL: {decl}\nPF:\n {pf}')

    def coerce_PS(step):
        ProofStep.update_forward_refs()
        if type(step) == str:
            return ProofStep(tactic=step)
        return ProofStep(tactic=coerce_trimmedThm(step))

    def coerce_trimmedThm(curr):
        return Theorem(decl=curr.decl,declID=thm.declID,src=thm.src,leanFile=thm.leanFile,context=thm.context,proof=[coerce_PS(step) for step in curr.proof])
     
    final = coerce_trimmedThm(output)
    
    #print(f'Running:\n{parseTheorem(thm,annotation=True)}\n')
    #thm = Theorem(decl=thm.decl,declID=thm.declID, proof=proof, leanFile=thm.leanFile, src=thm.src, context = thm.context)
    return final





def prompt_rag(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[],retries=6) -> Theorem:

    retriever = get_retriever(k=4)

#    print("retriever recieved")
    model = ChatOpenAI(model=model,temperature=0)

    class trimmedTheorem(BaseModel):
        decl : str = Field(description="Theorem declaration")
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
        return "\n\n".join(doc.page_content for doc in docs)


    chain = (
        {"rag": retriever | format_docs, "data_str": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(retries))
    def invoke_throttled(chain,data):
        return chain.invoke(data)
    
    output=invoke_throttled(chain,parseTheorem(thm,annotation=True,prompt=True))
    
         
    decl,pf = output.decl, output.proof
    print(f'DECL: {decl}\nPF:\n {pf}')

    def coerce_PS(step):
        ProofStep.update_forward_refs()
        if type(step) == str:
            return ProofStep(tactic=step)
        return ProofStep(tactic=coerce_trimmedThm(step))

    def coerce_trimmedThm(curr):
        return Theorem(decl=curr.decl,declID=thm.declID,src=thm.src,leanFile=thm.leanFile,context=thm.context,proof=[coerce_PS(step) for step in curr.proof])
     
    final = coerce_trimmedThm(output)
    
    #print(f'Running:\n{parseTheorem(thm,annotation=True)}\n')
    #thm = Theorem(decl=thm.decl,declID=thm.declID, proof=proof, leanFile=thm.leanFile, src=thm.src, context = thm.context)
    return final





#prompt()

repo = getRepo('Tests','configs/config_test.json')
files = {f.file_name: f for f in repo.files}
print(files.keys())
file = files['Tests/Basic.lean']
thms = file.theorems
thm = thms[0]


output = refinement(thm,length_metric(),1,promptfn=prompt_rag)
print(output)
print(parseTheorem(output,context=False))