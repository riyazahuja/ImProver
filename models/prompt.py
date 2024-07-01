from __future__ import annotations
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
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

#REQUIRES OPENAI_API_KEY envvar to be set



def parse_prev_data(data):
    intro = "For reference, here is the previous thread of inputs and GPT outputs (from most recent to least recent), along with any errors encountered in the compilation, and if compilation was successful, the metric score.\n Prioritize fixing any correctness errors before trying to shorten the proof as we require a short AND correct proof."
    data.reverse()
    rev = data
    text = ''
    for item in rev:
        err_msg=''
        score_msg=''
        if item['err'] == []:
            score_msg = f"Evaluation: Correct\n Metric ({item['score'][0]}) score = {item['score'][1]}"
        else:
            for msg in item['err']:
                #print(msg)
                try:
                    err_msg += f"[ERROR MESSAGE]\n(line {msg.get('pos',{}).get('line',-1)}, col {msg.get('pos',{}).get('column',-1)}"
                    try:
                        err_msg += f"- line {msg.get('endPos',{}).get('line',-1)}, col {msg.get('endPos',{}).get('column',-1)}"
                    except:
                        pass
                    err_msg+= f")\n{msg['data']}\n[END MESSAGE]"
                except Exception as e:
                    raise KeyError(str(msg))
            
        
        text += f'''========
INPUT: {parseTheorem(item['input'],context=False)}

OUTPUT: {parseTheorem(item['output'],context=False)}

{err_msg}{score_msg}
========
'''
    if text != '':
        return intro + text
    else:
        return ''


def prompt_basic(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[],retries=3) -> Theorem:

    model = ChatOpenAI(model=model,temperature=0)

    class Proof(BaseModel):
        contents : str = Field(description= "Contents of a lean proof (not including the theorem declaration or context. If a tactic proof, do not include the \"by\")")

    # Define the output parser
    parser = JsonOutputParser(pydantic_object= Proof)

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
    output=None
    err = None
    for _ in range(retries):
        try:
            output = chain.invoke({"data_str" : parseTheorem(thm,annotation=True,prompt=True)})
            break
        except Exception as e:
            err = e
            time.sleep(3)
            
    if output is None:
        print(err)
        raise TimeoutError('ERROR')
    proof = [ProofStep(tactic=output.get('contents',''))]
    
    #print(f'Running:\n{parseTheorem(thm,annotation=True)}\n')
    thm = Theorem(decl=thm.decl,declID=thm.declID, proof=proof, leanFile=thm.leanFile, src=thm.src, context = thm.context,project_path=thm.project_path)
    return thm



def prompt_structured(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[],retries=6,annotation=True) -> Theorem:

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

    #@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(retries))
    @retry(
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
    wait=wait_random_exponential(multiplier=1, max=60),
    )
    def invoke_throttled(chain,config):
        return chain.invoke(config)
    
    output=invoke_throttled(chain,{"data_str" : parseTheorem(thm,annotation=annotation,prompt=True)})
    
         
    decl,pf = output.decl, output.proof
    #print(f'DECL: {decl}\nPF:\n {pf}')

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





def best_of_n(thm:AnnotatedTheorem, metric: Metric, n: int, model = 'gpt-4-turbo', max_workers=1,promptfn = prompt_structured,annotation=True) -> Theorem:
    thms = []
    if max_workers == 1:
        for i in range(n):
            output = promptfn(thm,metric,model=model,annotation=annotation)
            correct,_ = eval_correctness(output)
            thms.append((output,correct))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(promptfn, thm, metric, model=model,annotation=annotation) for _ in range(n)]
            for future in futures:
                output = future.result()
                correct, _ = eval_correctness(output)
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
        correct,data = eval_correctness(output)

        if type(output) == Theorem:
            try:
                output = annotateTheorem(output,force=True)
            except Exception as e:
                print(output)
                raise e

        curr_data = {'input':curr,'output':output}
        curr_data['err'] = [msg for msg in data.get('messages',[]) if msg['severity'] == 'error']
        curr_data['score'] = (metric.name, metric.metric(output))

        prev_data.append(curr_data)

        if not keep_best:
            curr = output
        else:
            old_correct = eval_correctness(curr)[0]
            new_correct = correct
            if not old_correct or new_correct:
                curr = metric.cmp(curr,output)
    
    return curr
        


if __name__ == '__main__':
    src = 'Tests3'
    name = 'Tests3/Basic.lean'

    f = getAnnotatedFile(src,name)
    thms = f.theorems
    

    for thm in thms:
        #print(f"RAW: \n\n {thm} \n\nSending to GPT:\n")
        
        #out = refinement(thm,length_metric(),5,prev_data_num=1)
        out=prompt_structured(thm,length_metric())
        print(out)
        print('\n')
        print(parseTheorem(out,context=False))
        #print(f'DECL: {d}\nPF:\n {p}')
        print('=========\n\n\n=========')

        







