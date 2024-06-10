from lean_dojo import *
import os
from openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import sys
from structures import *
print(sys.path)
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


#Set GITHUB_ACCESS_TOKEN environment variable to a 
# github PAT for better rate limits



#REQUIRES OPENAI_API_KEY envvar to be set
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






def prompt_structured(thm:AnnotatedTheorem, force = False) -> List[ProofStep]:

    model = ChatOpenAI(model='gpt-4-turbo',temperature=0)

    # Define the output parser
    parser = JsonOutputParser(pydantic_object=List[ProofStep])
    gpt_assistant_prompt = "You are a bot that shortens Lean4 proofs while maintaining their correctness.\n"
    gpt_user_prompt = "Here is a proof in Lean 4. Your goal is to rewrite the proof so that it is shorter. To help you keep track of the state of the proof, and to help think of ways to rewrite the proof, we have provided the proof states as comments.\n"


    # Define the prompt template
    prompt = PromptTemplate(
        template=gpt_assistant_prompt + "\n" + gpt_user_prompt + "\n" + "{data_str}\n\n"+"{format_instructions}",
        input_variables=["data_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the chain
    chain = prompt | model | parser


    output = chain.invoke({"data_str" : thm})
    
    #TODO FORCING WITH RETRY PARSER
    return output
    

if __name__ == '__main__':
    filepath = 'test/Test/file.lean'
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),os.path.join('annotation/cache',filepath))
    with open(path,'r') as f:
        content = f.read()
    thms = extract_theorems(content)

    for thm in thms:
        print(f"RAW: \n\n {thm} \n\nSending to GPT:\n")

        out = prompt_structured(thm)
        print(out)
        print('\n\n')








