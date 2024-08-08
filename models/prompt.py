from __future__ import annotations
from langchain.globals import set_debug

# set_debug(True)
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from models.rag import *
from evaluate.metrics import *
from evaluate.eval import eval_correctness
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Final
import tiktoken

log_req_info = False

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
    n=None,
    annotation=True,
    syntax_search=False,
    mathlib_search=False,
    examples=0,
    token=False,
):
    syntax_k = 2
    mathlib_k = 3
    model_name = model

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
                f"""You will be given the proof context (i.e. the lean file contents/imports leading up to the theorem declaration) wrapped by <CONTEXT>...</CONTEXT>.
         {f"You will be given the previous {len(prev_data)} input/output pairs as well as their metric ({metric.name}) score and correctness score, as well as any error messages, for your reference to improve upon. Each of these previous results will be wrapped with <PREV I=0></PREV I=0>,...,<PREV I={len(prev_data)-1}></PREV I={len(prev_data)-1}>, with I={len(prev_data)-1} being the most recent result." if len(prev_data)!= 0 else ""}
         Remember to use lean 4 syntax, which has significant changes from the lean 3 syntax. {f"To assist with the syntax relating to the current theorem and current error messages, you will be given {syntax_k} documents to refer to for fixing these syntax issues. Each of these documents will be wrapped with <SYNTAX_DOC>...</SYNTAX_DOC>." if syntax_search else ""}
         {f"You will also recieve {mathlib_k} documents relevant to the current theorem to help with formulating your modified proof. Each of these will be wrapped with <CONTENT_DOC>...<CONTENT_DOC>" if mathlib_search else ""}
         {"You will be given the tactic states as comments for reference." if annotation else ""} Return only the proof of the theorem, starting at the first tactic. Do not include the theorem statement or hypotheses. The current theorem will be wrapped in <CURRENT>...</CURRENT>
         """,
            ),
            ("system", "{format_instructions}"),
            ("placeholder", "{syntax_docs}"),
            ("placeholder", "{mathlib_docs}"),
            ("placeholder", "{examples}"),
            ("human", "<CONTEXT>\n{context}\n</CONTEXT>"),
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
    )
    def invoke_throttled(chain, config):
        return chain.invoke(config)

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

    return output


def prompt_basic(
    thm: AnnotatedTheorem,
    metric: Metric,
    model="gpt-4-turbo",
    prev_data=[],
    n=None,
    annotation=True,
    syntax_search=False,
    mathlib_search=False,
    examples=0,
    token=False,
):

    class strProof(BaseModel):
        content: str = Field(
            description='The entire proof of the given theorem, without the declaration or context. begin after the ":= by".'
        )

    output = prompt_raw(
        thm,
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
    )

    if token:
        return output

    def coerce_Thm(curr):
        ProofStep.update_forward_refs()
        return Theorem(
            decl=thm.decl,
            declID=thm.declID,
            src=thm.src,
            leanFile=thm.leanFile,
            context=thm.context,
            proof=[ProofStep(tactic=curr.content)],
            project_path=thm.project_path,
        )

    final = coerce_Thm(output)

    return final


def prompt_flat(
    thm: AnnotatedTheorem,
    metric: Metric,
    model="gpt-4-turbo",
    prev_data=[],
    n=None,
    annotation=True,
    syntax_search=False,
    mathlib_search=False,
    examples=0,
    token=False,
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
        n=n,
        annotation=annotation,
        syntax_search=syntax_search,
        mathlib_search=mathlib_search,
        examples=examples,
        token=token,
    )

    if token:
        return output

    def coerce_trimmedThm(curr):
        ProofStep.update_forward_refs()
        return Theorem(
            decl=thm.decl,
            declID=thm.declID,
            src=thm.src,
            leanFile=thm.leanFile,
            context=thm.context,
            proof=[ProofStep(tactic=step) for step in curr.proof],
            project_path=thm.project_path,
        )

    final = coerce_trimmedThm(output)

    return final


def prompt_structured(
    thm: AnnotatedTheorem,
    metric: Metric,
    model="gpt-4-turbo",
    prev_data=[],
    n=None,
    annotation=True,
    syntax_search=False,
    mathlib_search=False,
    examples=0,
    token=False,
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
        n=n,
        annotation=annotation,
        syntax_search=syntax_search,
        mathlib_search=mathlib_search,
        examples=examples,
        token=token,
    )

    if token:
        return output

    def coerce_PS(step):
        ProofStep.update_forward_refs()
        if type(step) == str:
            return ProofStep(tactic=step)
        return ProofStep(tactic=coerce_trimmedThm(step))

    def coerce_trimmedThm(curr, force_decl=None):
        if force_decl is not None:
            decl = force_decl
        else:
            decl = curr.decl
        return Theorem(
            decl=decl,
            declID=thm.declID,
            src=thm.src,
            leanFile=thm.leanFile,
            context=thm.context,
            proof=[coerce_PS(step) for step in curr.proof],
            project_path=thm.project_path,
        )

    final = coerce_trimmedThm(output, force_decl=thm.decl)

    return final


# TODO Must fix output parsing to strict syntaxtree/tactic information. Then we can at least know for sure that we are splitting into two cases after an rcases etc.


def prompt_recursive_gen(
    thm: AnnotatedTheorem,
    metric: Metric,
    model="gpt-4-turbo",
    prev_data=[],
    n=None,
    annotation=True,
    syntax_search=False,
    mathlib_search=False,
) -> Theorem:
    output = prompt_structured(
        thm, metric, model, prev_data, n, annotation, syntax_search, mathlib_search
    )

    real_correct, real_msgs, real_anno_output = eval_correctness(
        output, sorries_are_errors=True
    )
    if real_correct:
        return output

    def sorry_replacement(thm: Theorem):
        new_proof = [
            (
                step
                if type(step.tactic) == str
                else ProofStep(
                    tactic=Theorem(
                        decl=step.decl,
                        proof=[ProofStep(tactic="sorry")],
                        declID=step.declID,
                        leanFile=step.leanFile,
                        context=step.context,
                        project_path=step.project_path,
                    )
                )
            )
            for step in thm.proof
        ]
        return Theorem(
            decl=thm.decl,
            proof=new_proof,
            declID=thm.declID,
            src=thm.src,
            leanFile=thm.leanFile,
            context=thm.context,
            project_path=thm.project_path,
        )

    sorry_output = sorry_replacement(output)
    sorry_correct, sorry_msgs, sorry_anno_output = eval_correctness(sorry_output)

    if sorry_correct:
        # if sorry is correct (i.e. base structure WILL solve the theorem), then we want to convert the goal state of said sorry
        # (extracted either directly from lean via a "sorries" attribute in json like repl, or taken from the annotatedThm)
        # into a full fledged theorem. Then we recurse on each of these subtheorems (in parallel?)
        #
        # if sorry is not correct (i.e. base structure WONT solve the theorem), then we will recurse the whole thing
        # but basically either fix it or redo (best of n on og thm, or refinement on output, idk)
        # then we keep doing this until we are correct up until sorries and we chilling.

        pass
    else:
        pass


def best_of_n(prompt_fn, max_workers=1, mixup=0):

    def best_of_n(
        thm: AnnotatedTheorem,
        metric: Metric,
        n: int,
        model="gpt-4-turbo",
        annotation=True,
        syntax_search=True,
        mathlib_search=True,
        examples=0,
        token=False,
    ):
        thms = []

        if token:
            return (
                prompt_fn(
                    thm,
                    metric,
                    model=model,
                    annotation=annotation,
                    syntax_search=syntax_search,
                    mathlib_search=mathlib_search,
                    examples=examples,
                    token=token,
                )
                * n
            )

        if max_workers == 1:
            for i in range(n):
                output = prompt_fn(
                    thm,
                    metric,
                    model=model,
                    annotation=annotation,
                    syntax_search=syntax_search,
                    mathlib_search=mathlib_search,
                    examples=examples,
                )
                correct, _, _ = eval_correctness(output)
                thms.append((output, correct))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    (
                        executor.submit(
                            prompt_structured,
                            thm,
                            metric,
                            model=model,
                            annotation=annotation,
                            syntax_search=syntax_search,
                            mathlib_search=mathlib_search,
                            examples=examples,
                        )
                        if i >= mixup * n
                        else executor.submit(
                            prompt_structured,
                            thm,
                            metric,
                            model=model,
                            annotation=annotation,
                            examples=examples,
                        )
                    )
                    for i in range(n)
                ]
                for future in futures:
                    output = future.result()
                    correct, _, _ = eval_correctness(output)
                    thms.append((output, correct))

        correct_thms = [item for item in thms if item[1]]
        if len(correct_thms) == 0:
            return thms[0][0]

        best = correct_thms[0][0]
        for t, correct in correct_thms:
            if not correct:
                continue
            best = metric.cmp(best, t)
        return best

    return best_of_n


def refinement(prompt_fn, prev_data_num=1, keep_best=False):

    def refinement(
        thm: AnnotatedTheorem,
        metric: Metric,
        n: int,
        model="gpt-4-turbo",
        annotation=True,
        syntax_search=True,
        mathlib_search=True,
        examples=0,
        token=False,
    ):

        if token:
            cost_one = prompt_fn(
                thm,
                metric,
                model=model,
                annotation=annotation,
                syntax_search=syntax_search,
                mathlib_search=mathlib_search,
                examples=examples,
                token=token,
            )
            encoding = tiktoken.encoding_for_model(model)
            cost_prev_data = 2 * len(encoding.encode(parseTheorem(thm, context=False)))
            total = 0
            for i in range(n):
                total += cost_one + min(prev_data_num, i) * cost_prev_data
            return total

        curr = thm
        prev_data = []

        for i in range(n):

            output = prompt_fn(
                curr,
                metric,
                model=model,
                prev_data=prev_data[-prev_data_num:],
                annotation=annotation,
                syntax_search=syntax_search,
                mathlib_search=mathlib_search,
                examples=examples,
            )
            correct, messages, new_thm = eval_correctness(output)
            curr_data = {"input": curr, "output": new_thm}
            curr_data["correct"] = correct
            curr_data["messages"] = messages
            curr_data["score"] = (
                (metric.name, metric.score(new_thm))
                if correct and metric.score_fn is not None
                else None
            )
            curr_data["delta"] = (
                (metric.name, metric.metric(curr, new_thm)) if correct else None
            )
            curr_data["minmax"] = metric.minmax

            prev_data.append(curr_data)

            if not metric.lock_refinement_state:
                if not keep_best:
                    curr = new_thm
                else:
                    old_correct = eval_correctness(curr)[0]
                    new_correct = correct

                    # if old correct and new incorrect, old
                    # if old correct, and new correct, min
                    # if old incorrect and new correct, new
                    # if both incorrect, one with less messages.

                    if old_correct and new_correct:
                        curr = metric.cmp(curr, new_thm)
                    elif old_correct and not new_correct:
                        pass
                    elif not old_correct and new_correct:
                        curr = new_thm
                    else:
                        curr = new_thm  # min(curr,new_thm,key=lambda x:len(x.messages))

        return curr

    return refinement


# def best_of_refined(prompt_fn, n_best,n_ref,prev_data_num=1, keep_best=False,max_workers=1):
#     def best_of_refined(
#         thm: AnnotatedTheorem,
#         metric: Metric,
#         n: int,
#         model="gpt-4-turbo",
#         annotation=True,
#         syntax_search=True,
#         mathlib_search=True,
#         examples=0,
#         token=False,
#     ):
#         best_of_n(prompt_fn=prompt_fn)(thm,metric,n_best,model=model,max_workers=max_workers,annotation=annotation,syntax_search=syntax_search,mathlib_search=mathlib_search,examples=examples,token=token)

if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    repo = getRepo("Tests", "configs/config_test.json")
    files = {file.file_name: file for file in repo.files}
    f = files["Solutions_S01_Implication_and_the_Universal_Quantifier.lean"]

    thms = f.theorems
    for thm in [thms[0]]:
        metric = length_metric()
        out = prompt_structured(thm, length_metric(), model="gpt-4o", examples=1)
        print("\n")
        print(parseTheorem(out, context=False))
