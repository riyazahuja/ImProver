from __future__ import annotations
from langchain.globals import set_debug
import time

# set_debug(True)
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from models.rag import *
from evaluate.metrics import *
from evaluate.eval import *

# from generation.recgen import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from typing import Final
import tiktoken
from multiprocessing import cpu_count
from pydantic import BaseModel, Field

log_req_info = False

if log_req_info:
    logger: Final = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    wait_random_exponential,
    stop_after_attempt,
)


"""
All prompting/sampling functions that interact with the LLM
"""


def parse_prev_data(data):

    output = []
    for idx, curr in list(enumerate(data)):
        inp = curr["input"]
        out = curr["output"]
        correct = curr["correct"]
        msgs = curr["messages"]
        # msgs_txt = "\n".join([f"{msg.message_src}\t|\t{msg.content}" for msg in msgs])
        msgs_txt = "\n".join(msgs)
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
    thm: str,
    context: str,
    metric: Metric,
    obj=str,
    model="gpt-4-turbo",
    prev_data=[],
    n=None,
    annotation=True,
    syntax_search=False,
    mathlib_search=False,
    examples=0,
    token=False,
    improved_context=False,
):
    syntax_k = 5
    mathlib_k = 5
    model_name = model

    model = ChatOllama(
        model="llama3.2",
        temperature=0.7,
    )
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
         Remember to use lean 4 syntax, which has significant changes from the lean 3 syntax. {f"To assist with the syntax relating to the current theorem and current error messages, you will be given {syntax_k} documents to refer to for fixing these syntax issues. Each of these documents will be wrapped with <SYNTAX_DOC>...</SYNTAX_DOC>." if syntax_search else ""}
         {f"You will also recieve {mathlib_k} documents relevant to the current theorem to help with formulating your modified proof. Each of these will be wrapped with <CONTENT_DOC>...<CONTENT_DOC>" if mathlib_search else ""}
         {"You will be given the tactic states as comments for reference." if annotation else ""} Return only the full theorem, starting with the same theorem declaration/statement and then your optimized proof of the theorem, starting at the first tactic. This theorem statement must match that of the current input theorem, which your goal is to optimize the proof of. The current theorem will be wrapped in <CURRENT>...</CURRENT>. Return valid lean 4 code of the theorem, and do not output the theorem context.
         """,
            ),
            ("system", "{format_instructions}"),
            ("placeholder", "{syntax_docs}"),
            ("placeholder", "{mathlib_docs}"),
            ("placeholder", "{examples}"),
            ("human", "<FILE_CONTEXT>\n{context}\n</FILE_CONTEXT>"),
            ("placeholder", "{prev_results}"),
            ("placeholder", "{user_prompts}"),
            ("human", "<CURRENT>\n{theorem}\n</CURRENT>"),
        ]
    )

    def format_docs(docs, wrapper):
        return [
            ("human", f"<{wrapper}>\n{doc.page_content}\n</{wrapper}>") for doc in docs
        ]

    def get_syntax(data):
        if not syntax_search:
            return []
        retriever = get_retriever(
            k=syntax_k, persist_dir=os.path.join(root_path, ".db", ".TPiL_chroma_db")
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

        out = format_docs(retriever.invoke(prompt), "SYNTAX_DOC")
        return out

    def get_mathlib(data):
        if not mathlib_search:
            return []
        retriever = get_retriever(
            k=mathlib_k,
            persist_dir=os.path.join(root_path, ".db", ".mathlib_chroma_db"),
        )
        curr_thm = data["theorem"]

        out = format_docs(retriever.invoke(curr_thm), "CONTENT_DOC")
        return out

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
                mathlib_docs=get_mathlib,
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
            mathlib_docs=get_mathlib,
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
        stop=stop_after_attempt(8),
    )
    def invoke_throttled(chain, config):
        return chain.invoke(config)

    st = time.time()
    output = invoke_throttled(
        chain,
        {
            "context": context,
            "prev_results": prev_data_parsed,
            "theorem": thm,
            "system_prompts": system_prompts,
            "user_prompts": user_prompts,
        },
    )
    if log_req_info:
        print(f"API Call completed in {time.time()-st}s")

    return output


# Note: all prompt functions return a Theorem object, unless token=True, in which case they return the number of tokens in the prompt
# however, we now make them output a (Theorem, trajectories), where trajectories is a list of Theorems


def prompt_basic(
    thm: str,
    context: str,
    metric: Metric,
    model="gpt-4-turbo",
    prev_data=[],
    n=None,
    annotation=True,
    syntax_search=False,
    mathlib_search=False,
    examples=0,
    token=False,
    improved_context=False,
):

    class strProof(BaseModel):
        content: str = Field(
            description="The entire proof of the given theorem, including the declaration but without any context. Output syntactically correct Lean 4 code."
        )

    output = prompt_raw(
        thm,
        context,
        metric,
        strProof,
        model=model,
        prev_data=prev_data,
        n=n,
        annotation=annotation,
        syntax_search=syntax_search,
        mathlib_search=mathlib_search,
        examples=examples,
        token=token,
        improved_context=improved_context,
    )

    if token:
        return output
    # out = Theorem(output.content, thm.repo, thm.file)
    print(output.content)
    # return out, out


if __name__ == "__main__":
    thm = sys.argv[1]
    context = sys.argv[2]
    metric = sys.argv[3]

    if metric == "LENGTH":
        metric = length_metric()

    prompt_basic(thm, context, metric)
