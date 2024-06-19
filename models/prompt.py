from lean_dojo import *
import os
from openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
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



#REQUIRES OPENAI_API_KEY envvar to be set


#UPDATE THEOREM PARSING SO WE GET ANNOTATED IN ANNOTATEDTHEOREMS
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
                err_msg += f"[ERROR MESSAGE]\n(line {msg['pos']['line']}, col {msg['pos']['column']} - line {msg['endPos']['line']}, col {msg['endPos']['column']})\n{msg['data']}\n[END MESSAGE]"
            
        
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


def prompt_structured(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo', prev_data=[]) -> Theorem:

    model = ChatOpenAI(model=model,temperature=0)

    class Proof(BaseModel):
        contents : List[ProofStep] = Field(description= "Contents of a proof, seperated into a sequence of proof steps (i.e. tactics)")

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

    output = chain.invoke({"data_str" : parseTheorem(thm,annotation=True,prompt=True)})
    proof = output.get('contents',[])
    
    print(f'Running:\n{parseTheorem(thm,annotation=True)}\n')
    thm = Theorem(decl=thm.decl,declID=thm.declID, proof=proof, leanFile=thm.leanFile, src=thm.src, context = thm.context)
    return thm





def best_of_n(thm:AnnotatedTheorem, metric: Metric, n: int, model = 'gpt-4-turbo', max_workers=1) -> Theorem:
    thms = []
    if max_workers == 1:
        for i in range(n):
            output = prompt_structured(thm,metric,model)
            correct,_ = eval_correctness(output)
            thms.append((output,correct))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(prompt_structured, thm, metric, model) for _ in range(n)]
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



def refinement(thm:AnnotatedTheorem,metric:Metric,n:int,model='gpt-4-turbo', prev_data_num = 1, keep_best = False) -> Theorem:
    curr = thm
    prev_data = []

    for i in range(n):
        print(f'=== i: {i} ===\n curr:\n {parseTheorem(curr,context=False)}\n prev_data = {parse_prev_data(prev_data[-prev_data_num:])}\n\n========')
        output = prompt_structured(curr,metric,model,prev_data=prev_data[-prev_data_num:]) 
        correct,data = eval_correctness(output)
        #print(parseTheoremAny(output,context=False))
        if type(output) == Theorem :
            output = annotateTheorem(output)
        print(f'COERCED!!!!! \n\n{parseTheorem(output,context=False,annotation=True)}\n\n')

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
    
    for thm in [thms[0]]:
        #print(f"RAW: \n\n {thm} \n\nSending to GPT:\n")
        
        out = refinement(thm,length_metric(),5,prev_data_num=1)
        #print(out)
        print(parseTheorem(out))
        #print('\n\n')

        







