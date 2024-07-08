from __future__ import annotations
from langchain.globals import set_verbose,set_debug
set_debug(True)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import sys
#print(sys.path)
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from evaluate.metrics import *
from evaluate.eval import eval_correctness
from concurrent.futures import ThreadPoolExecutor
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
import logging
from typing import Final

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)



# def prompt_basic(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[],retries=3) -> Theorem:

#     model = ChatOpenAI(model=model,temperature=0)

#     class Proof(BaseModel):
#         contents : str = Field(description= "Contents of a lean proof (not including the theorem declaration or context. If a tactic proof, do not include the \"by\")")

#     # Define the output parser
#     parser = JsonOutputParser(pydantic_object= Proof)

#     actual_prompt=metric.prompt.replace('{',r'{{').replace('}',r'}}')
#     prev_data_text = parse_prev_data(prev_data).replace('{',r'{{').replace('}',r'}}')
#     # Define the prompt template
#     prompt = PromptTemplate(
#         template=actual_prompt + "{data_str}\n\n"+prev_data_text+'\n'+"{format_instructions}",
#         input_variables=["data_str"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )

#     # Create the chain
#     chain = prompt | model | parser
#     output=None
#     err = None
#     for _ in range(retries):
#         try:
#             output = chain.invoke({"data_str" : parseTheorem(thm,annotation=True,prompt=True)})
#             break
#         except Exception as e:
#             err = e
#             time.sleep(3)
            
#     if output is None:
#         print(err)
#         raise TimeoutError('ERROR')
#     proof = [ProofStep(tactic=output.get('contents',''))]
    
#     #print(f'Running:\n{parseTheorem(thm,annotation=True)}\n')
#     thm = Theorem(decl=thm.decl,declID=thm.declID, proof=proof, leanFile=thm.leanFile, src=thm.src, context = thm.context,project_path=thm.project_path)
#     return thm


'''
Prompt Template: 


System: You are a ... optimize for __metric__ ... Use documents, [have included the previous x iterations]? Use context.

{format instructions}

Examples?

Context: {cxt}


Syntax Documents: "Here are documents that may be useful for syntax issues..."
--> documents are taken from error messages content primarily, and secondarily from the main input 

Context Documents:

=====================| i=0
Input: {thm}

Output:
    {outputted thm}
---------------------|               <-- each is a user message
correct? 
messages:
score:
=====================|i=1
Input: {thm}

Output:
    {outputted thm}
---------------------| i=2
    correct? 
    messages:
    score:
---------------------| i=3

...
===(end refinement)===

---------------------|
Input: {thm}


Output:



'''

def parse_prev_data(data):
    
    output=[]
    for idx,curr in list(enumerate(data)):
        inp = curr['input']
        out = curr['output']
        correct = curr['correct']
        msgs = curr['messages']
        msgs_txt = "\n".join([f"{msg.message_src}\t|\t{msg.content}" for msg in msgs])
        score = curr['score']


        msg = f'''<PREV I={idx}>
        Input:
        {parseTheorem(inp,context=False)}
        Output:
        {parseTheorem(out,context=False)}

        Correct? {correct}
        Messages:
        {msgs_txt}

        Metric Score: {None if score is None else f"{score[1]} ({score[0]})"}
        </PREV I={idx}>'''
        output.append(('human',msg))
    return output

def prompt_structured(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[],retries=6,annotation=True) -> Theorem:

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
         Remember to use lean 4 syntax, which has significant changes from the lean 3 syntax. Output in a proof tree format that aligns with the pydantic output parsing object schema that splits off subproofs and subtheorems.
         {"You will be given the tactic states as comments for reference." if annotation else ""} The current theorem will be wrapped in <CURRENT>...</CURRENT>
         '''),
         ('system','{format_instructions}'),
         ('human','<CONTEXT>\n{context}\n</CONTEXT>'),
         ("placeholder", "{prev_results}"),
         ('human','<CURRENT>\n{theorem}\n</CURRENT>')
    ])

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





def best_of_n(thm:AnnotatedTheorem, metric: Metric, n: int, model = 'gpt-4-turbo', max_workers=1,promptfn = prompt_structured,annotation=True) -> Theorem:
    thms = []
    if max_workers == 1:
        for i in range(n):
            output = promptfn(thm,metric,model=model,annotation=annotation)
            correct,_,_ = eval_correctness(output)
            thms.append((output,correct))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(promptfn, thm, metric, model=model,annotation=annotation) for _ in range(n)]
            for future in futures:
                output = future.result()
                correct, _, _ = eval_correctness(output)
                thms.append((output,correct))
    
    correct_thms = [item for item in thms if item[1]]
    if len(correct_thms) == 0:
        return thms[0][0]
    
    best = correct_thms[0][0]
    for t,_ in correct_thms:
        best = metric.cmp(best,t)
    return best



def refinement(thm:AnnotatedTheorem,metric:Metric,n:int,model='gpt-4-turbo', prev_data_num = 1, keep_best = False,promptfn=prompt_structured,annotation=True) -> Theorem:
    curr = thm
    prev_data = []

    for i in range(n):
        #print(f'=== i: {i} ===\n curr:\n {parseTheorem(curr,context=False)}\n prev_data = {parse_prev_data(prev_data[-prev_data_num:])}\n\n========')
        output = promptfn(curr,metric,model=model,prev_data=prev_data[-prev_data_num:],annotation=annotation) 
        correct,messages,new_thm = eval_correctness(output)

        # if type(output) == Theorem:
        #     try:
        #         output = annotateTheorem(output,force=True)
        #     except Exception as e:
        #         print(output)
        #         raise e

        curr_data = {'input':curr,'output':new_thm}
        curr_data['correct'] = correct
        curr_data['messages'] = messages
        curr_data['score'] = (metric.name, metric.metric(new_thm)) if correct else None

        prev_data.append(curr_data)

        if not keep_best:
            curr = new_thm
        else:
            old_correct = eval_correctness(curr)[0]
            new_correct = correct

            #if old correct and new incorrect, old
            # if old correct, and new correct, min
            # if old incorrect and new correct, new
            # if both incorrect, one with less messages.


            if old_correct and new_correct:
                curr = metric.cmp(curr,new_thm)
            elif old_correct and not new_correct:
                pass
            elif not old_correct and new_correct:
                curr = new_thm
            else:
                curr = new_thm#min(curr,new_thm,key=lambda x:len(x.messages))
    
    return curr
        


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    r = getRepo('Tests','configs/config_test.json')
    files = {file.file_name:file for file in r.files}
    #print(files.keys())


    f = files['Basic.lean']
    thms = f.theorems
    for thm in [thms[1]]:
        #print(f"RAW: \n\n {parseTheorem(thm,context=False)} \n\nSending to GPT:\n")
        #out=prompt_structured(thm,length_metric())
        #out = best_of_n(thm,length_metric(),3,max_workers=3)
        out=refinement(thm,length_metric(),3,prev_data_num=3)
        #print(out)
        
        correct,msgs,anno = eval_correctness(out)
        msgs_txt = "\n".join([f"{msg.message_src}\t|\t{msg.content}" for msg in msgs])
        print('\n')
        print(parseTheorem(out,context=False))
        print(f'CORRECT? {correct}\nMSGS:\n{msgs_txt}')#\nMSGS_RAW:\n{msgs}\nOUT_RAW:\n{anno}')
        print('=========\n\n\n=========')
        #print(out)

        







