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

#Set GITHUB_ACCESS_TOKEN environment variable to a 
# github PAT for better rate limits



#REQUIRES OPENAI_API_KEY envvar to be set
'''
def prompt_raw(thm) -> str:
    client = OpenAI()
    gpt_assistant_prompt = "You are a bot that shortens Lean4 proofs while maintaining their correctness.\n"
    gpt_user_prompt = "Here is a proof in Lean 4. Your goal is to rewrite the proof so that it is shorter. To help you keep track of the state of the proof, and to help think of ways to rewrite the proof, we have provided the proof states as comments.\n"
    gpt_data = thm
    gpt_prompt = gpt_user_prompt + gpt_data

    message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": gpt_prompt}]
    temperature=0
    max_tokens=256
    frequency_penalty=0.0


    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages = message,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )
    try:
        return response.choices[0].message
    except:
        return f"ERROR: {response.__dict__()}"

'''




def prompt_structured(thm:AnnotatedTheorem, metric:Metric, model = 'gpt-4-turbo') -> List[ProofStep]:

    model = ChatOpenAI(model=model,temperature=0)

    class Proof(BaseModel):
        contents : List[ProofStep] = Field(description= "Contents of a proof, seperated into a sequence of proof steps (i.e. tactics)")

    # Define the output parser
    parser = JsonOutputParser(pydantic_object= Proof)
    
    #gpt_assistant_prompt = "You are a bot that shortens Lean4 proofs while maintaining their correctness.\n"
    #gpt_user_prompt = "Here is a proof in Lean 4. Your goal is to rewrite the proof so that it is shorter. To help you keep track of the state of the proof, and to help think of ways to rewrite the proof, we have provided the proof states as comments.\n"


    # Define the prompt template
    prompt = PromptTemplate(
        template=metric.prompt + "{data_str}\n\n"+"{format_instructions}",
        input_variables=["data_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the chain
    chain = prompt | model | parser


    output = chain.invoke({"data_str" : thm})
    proof = output.get('contents',[])
    #TODO FORCING WITH RETRY PARSER
    
    thm = Theorem(decl=thm.decl,declID=thm.declID, proof=proof, leanFile=thm.leanFile, src=thm.src, context = thm.context)
    #ANNOTATION TIME!
    annotated = annotateTheorem(thm)
    return thm

    

if __name__ == '__main__':
    src = 'Tests'
    name = 'Tests/Basic.lean'

    f = getAnnotatedFile(src,name)
    thms = f.theorems
    
    for thm in thms:
        print(f"RAW: \n\n {thm} \n\nSending to GPT:\n")
        
        out = prompt_structured(thm,length_metric())
        print(out)
        print(parseTheorem(out))
        print('\n\n')








