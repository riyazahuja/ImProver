from __future__ import annotations
from langchain.globals import set_debug
import time

# set_debug(True)
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures_new import *
from models.rag import *
from evaluate.metrics import *
from evaluate.eval import eval_correctness
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from typing import Final
import tiktoken
from multiprocessing import cpu_count

log_req_info = True

if log_req_info:
    logger: Final = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    wait_random_exponential,
)


def parse_prev_data(data):

    output = []
    for idx, curr in list(enumerate(data)):
        inp = curr["input"]
        out = curr["output"]
        correct = curr["correct"]
        msgs = curr["messages"]
        msgs_txt = "\n".join([f"{msg.message_src}\t|\t{msg.content}" for msg in msgs])
        score = curr["score"]
        delta = curr["delta"]
        minmax = curr["minmax"]

        msg = f"""<PREV I={idx}>
        Input:
        {parseTheorem(inp,context=False)}
        Output:
        {parseTheorem(out,context=False)}

        Correct? {correct}
        Messages:
        {msgs_txt}

        Metric Score: {None if score is None else f"{score[1]} ({score[0]})"}
        Metric Delta: {None if delta is None else f"{delta[1]} ({delta[0]})"}
        {"(Bigger is Better)" if minmax == "MAX" else "(Smaller is Better)" if minmax=='MIN' else ""}
        </PREV I={idx}>"""
        output.append(("human", msg))
    return output


def prompt_raw(
    thm: AnnotatedTheorem,
    metric: Metric,
    obj=str,
    model="gpt-4-turbo",
    prev_data=[],
    annotation=True,
    rag=None,
    examples=0,
    token=False,
    improved_context=False,
):
    model_name = model
    rag_on = rag is not None
    if rag_on:
        rag_k = rag
    else:
        rag_k = 0

    model = ChatOpenAI(model=model)

    str_output = obj == str
    if obj == str:
        parser = StrOutputParser()
    else:
        parser = PydanticOutputParser(pydantic_object=obj)

    def fix_prompt(prompt):
        return (prompt[0], prompt[1].replace("{", r"{{").replace("}", r"}}"))

    system_prompts = [
        fix_prompt(prompt) for prompt in metric.prompt if prompt[0] == "system"
    ]
    user_prompts = [
        fix_prompt(prompt) for prompt in metric.prompt if prompt[0] == "human"
    ]

    prev_data_parsed = parse_prev_data(prev_data)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{system_prompts}"),
            (
                "system",
                f"""You will be given the proof context (i.e. the lean file contents/imports leading up to the theorem declaration) wrapped by <FILE_CONTEXT>...</FILE_CONTEXT>.
         {'Additional context from other imports and modules will be wrapped by <MOD_CONTEXT>...</MOD_CONTEXT>, containing metadata on what dependency this context is for, where it was imported from, and whether it was explicit in the current theorem - in addition to the source declaration of the dependency.' if improved_context else ''}
         {f"You will be given the previous {len(prev_data)} input/output pairs as well as their metric ({metric.name}) score and correctness score, as well as any error messages, for your reference to improve upon. Each of these previous results will be wrapped with <PREV I=0></PREV I=0>,...,<PREV I={len(prev_data)-1}></PREV I={len(prev_data)-1}>, with I={len(prev_data)-1} being the most recent result." if len(prev_data)!= 0 else ""}
         Remember to use lean 4 syntax, which has significant changes from the lean 3 syntax. {f"To assist with the syntax relating to the current theorem and current error messages, as well as other relevant information, you will be given {rag_k} documents to refer to for fixing these syntax issues a. Each of these documents will be wrapped with <DOC>...</DOC>." if rag_on else ""}
         {"You will be given the tactic states as comments for reference." if annotation else ""} Return only the proof of the theorem, starting at the first tactic. Do not include the theorem statement or hypotheses. The current theorem will be wrapped in <CURRENT>...</CURRENT>
         """,
            ),
            ("system", "{format_instructions}"),
            ("placeholder", "{syntax_docs}"),
            # ("placeholder", "{mathlib_docs}"),
            ("placeholder", "{examples}"),
            ("human", "<FILE_CONTEXT>\n{context}\n</FILE_CONTEXT>"),
            ("placeholder", "{mod_context}"),
            ("placeholder", "{prev_results}"),
            ("placeholder", "{user_prompts}"),
            ("human", "<CURRENT>\n{theorem}\n</CURRENT>"),
        ]
    )

    def format_docs(docs, wrapper):
        return [
            ("human", f"<{wrapper}>\n{doc.page_content}\n</{wrapper}>") for doc in docs
        ]

    def parse_dependency(dep: Dependency):
        return f"""<MODULE_CONTEXT>
        Dependency Name: {dep.dependency} (type = {dep.kind})
        Source: {dep.src_file}
        Explicit? {dep.explicit}
        Content:
        
        {dep.src_content}
    </MODULE_CONTEXT>"""

    def get_module_context(data):
        curr_thm = data["theorem"]
        if not improved_context:
            return []

        deps = [
            (parse_dependency(dep), dep.dependency, dep.explicit)
            for dep in thm.dependencies
        ]
        explicit = []
        nonexplicit = []
        for dep in deps:
            if dep[2]:
                explicit.append((dep[0], dep[1]))
            else:
                nonexplicit.append((dep[0], dep[1]))

        def key_it(x):
            content = data["theorem"]
            return -1 * content.count(x[1])

        # def key_it(x):
        #     lines = list(enumerate(data["theorem"].splitlines()))
        #     found = [line[0] for line in lines if x[1] in line[1]]
        #     if len(found) == 0:
        #         return len(lines) + 1
        #     else:
        #         return found[0]

        explicit.sort(key=key_it)
        return [("human", x[0]) for x in explicit + nonexplicit]

    def get_syntax(data):
        if not rag_on:
            return []
        retriever = get_retriever(
            k=rag_k, persist_dir=os.path.join(root_path, ".db", ".TPiL_chroma_db")
        )
        curr_thm = data["theorem"]
        if len(prev_data) != 0:
            recent = prev_data[-1]
            msgs = recent["messages"]

            msg_text = "\n".join([f"{msg.content} {msg.message_src}" for msg in msgs])
        else:
            msg_text = ""
        err = f"\nCurrent Errors:\n{msg_text}" if msg_text != "" else ""
        prompt = f"Current Theorem:\n{curr_thm}{err}"

        out = format_docs(retriever.invoke(prompt), "DOC")
        return out

    # def get_mathlib(data):
    #     if not mathlib_search:
    #         return []
    #     retriever = get_retriever(
    #         k=mathlib_k,
    #         persist_dir=os.path.join(root_path, ".db", ".mathlib_chroma_db"),
    #     )
    #     curr_thm = data["theorem"]

    #     out = format_docs(retriever.invoke(curr_thm), "CONTENT_DOC")
    #     return out

    def get_examples(data):
        if examples == 0:
            return []
        retriever = get_retriever(
            k=examples,
            persist_dir=os.path.join(
                root_path, ".db", "metrics", f".{metric.name}_chroma_db"
            ),
        )
        curr_thm = data["theorem"]

        out = format_docs(retriever.invoke(curr_thm), "EXAMPLE")
        return out

    if token:
        chain = (
            RunnablePassthrough().assign(
                format_instructions=lambda _: (
                    parser.get_format_instructions() if not str_output else ""
                ),
                syntax_docs=get_syntax,
                # mathlib_docs=get_mathlib,
                mod_context=get_module_context,
                examples=get_examples,
            )
            | prompt
        )

        input_str = chain.invoke(
            {
                "context": thm.context,
                "prev_results": prev_data_parsed,
                "theorem": parseTheorem(thm, annotation=annotation, context=False),
                "system_prompts": system_prompts,
                "user_prompts": user_prompts,
            }
        ).to_string()

        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(input_str))
        return num_tokens

    chain = (
        RunnablePassthrough().assign(
            format_instructions=lambda _: (
                parser.get_format_instructions() if not str_output else ""
            ),
            syntax_docs=get_syntax,
            # mathlib_docs=get_mathlib,
            mod_context=get_module_context,
            examples=get_examples,
        )
        | prompt
        | model
        | parser
    )

    @retry(
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO) if log_req_info else None,
        after=after_log(logger, logging.INFO) if log_req_info else None,
        wait=wait_random_exponential(multiplier=1, max=60),
    )
    def invoke_throttled(chain, config):
        return chain.invoke(config)

    st = time.time()
    output = invoke_throttled(
        chain,
        {
            "context": thm.context,
            "prev_results": prev_data_parsed,
            "theorem": parseTheorem(thm, annotation=annotation, context=False),
            "system_prompts": system_prompts,
            "user_prompts": user_prompts,
        },
    )
    if log_req_info:
        print(f"API Call completed in {time.time()-st}s")

    return output


def prompt_basic(
    thm: AnnotatedTheorem,
    metric: Metric,
    model="gpt-4-turbo",
    prev_data=[],
    annotation=True,
    rag=None,
    examples=0,
    token=False,
    improved_context=False,
):

    class strProof(BaseModel):
        content: str = Field(
            description='The entire proof of the given theorem, without the declaration or context. begin after the ":= by" at the first tactic.'
        )

    output = prompt_raw(
        thm,
        metric,
        strProof,
        model=model,
        prev_data=prev_data,
        annotation=annotation,
        rag=rag,
        examples=examples,
        token=token,
        improved_context=improved_context,
    )

    if token:
        return output

    return PromptResult(content=f"{thm.context}\n\n{thm.decl} := by\n{output}")


def prompt_flat(
    thm: AnnotatedTheorem,
    metric: Metric,
    model="gpt-4-turbo",
    prev_data=[],
    annotation=True,
    rag=None,
    examples=0,
    token=False,
    improved_context=False,
):

    class Proof(BaseModel):
        proof: List[str] = Field(
            ...,
            description="Sequence of proofsteps for full proof of theorem. Each proofstep is one line/tactic in a tactic proof",
        )

    output = prompt_raw(
        thm,
        metric,
        Proof,
        model=model,
        prev_data=prev_data,
        annotation=annotation,
        rag=rag,
        examples=examples,
        token=token,
        improved_context=improved_context,
    )

    if token:
        return output
    
    return PromptResult(content=f"{thm.context}\n\n{thm.decl} := by\n{'\n'.join(output)}")

def prompt_structured(
    thm: AnnotatedTheorem,
    metric: Metric,
    model="gpt-4-turbo",
    prev_data=[],
    annotation=True,
    rag=None,
    examples=0,
    token=False,
    improved_context=False,
):
    class trimmedTheorem(BaseModel):
        decl: str = Field(
            description='(sub)Theorem declaration (does not include ":= by") For example, a "have" statement would be the decl, and the proof of the have statement would be the (sub)proof. Similarly, a "case [Name] =>" statement would be the decl, and the case proof would be the proof.'
        )
        proof: List[Union[str, trimmedTheorem]] = Field(
            ...,
            description="Sequence of proofsteps for full proof of theorem. Each proofstep is one line/tactic in a tactic proof (str) or a subtheorem/sublemma/subproof in the format of (trimmedTheorem)",
        )

    output = prompt_raw(
        thm,
        metric,
        trimmedTheorem,
        model=model,
        prev_data=prev_data,
        annotation=annotation,
        rag=rag,
        examples=examples,
        token=token,
        improved_context=improved_context,
    )

    if token:
        return output
    

    def parse_proof(thm, indent=1):
        output = ""
        spaces = "  "
        proof = thm.proof
        for content in proof:
            if type(content) == str:
                output += indent * spaces + content + "\n"
                # single tactic
            else:
                hasDecl = content.decl is not None
                if hasDecl:
                    output += (
                        indent * spaces
                        + content.decl
                        + ("" if "." == content.decl.strip() else "\n")
                    )
                output += parse_proof(content, indent=indent + 1, dot=(not hasDecl))
        return output
                
    return PromptResult(content=f"{thm.context}\n\n{thm.decl} := by\n{parse_proof(output)}")



if __name__ == "__main__":
    
    args = sys.argv
    if len(args) < 2:
        raise ValueError(f'bad input: {' '.join(args)}')
    
    file_name = args[1]
    with open(file_name,'r') as f:
        data = json.load(f)
    
    request = parse_prompt_request(data)
    
    thm = request.theorem
    metric= request.metric
    model=request.model
    prev_data=request.previous_data
    annotation=request.annotation
    rag=request.rag
    examples=request.examples
    token=request.token
    improved_context=request.improved_context
    
    output_format = request.output_format
    if output_format == 'prompt_basic':
        output = prompt_basic(thm,metric,model=model,prev_data=prev_data,annotation=annotation,rag=rag,examples=examples,token=token,improved_context=improved_context)
    elif output_format == 'prompt_flat':
        output = prompt_basic(thm,metric,model=model,prev_data=prev_data,annotation=annotation,rag=rag,examples=examples,token=token,improved_context=improved_context)
    elif output_format == 'prompt_structured':
        output = prompt_basic(thm,metric,model=model,prev_data=prev_data,annotation=annotation,rag=rag,examples=examples,token=token,improved_context=improved_context)
    else:
        raise ValueError(f'requested a weird output format: {output_format}')
    
    outtt= json.dumps(output.__dict__)
    print(outtt)