import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from repl.repl import *
from models.structures import *
from models.prompt import prompt_structured
from evaluate.metrics import *

def eval_correctness(thm):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = thm.src
    lean_file = thm.leanFile
    pardir = os.path.dirname(lean_file)

    og_path = os.path.join(root_path,'.lake/packages',src)

    thm_text = parseTheorem(thm)
    print(thm_text)
    #print(og_path)
    '''
    TODO: FIX
    try:
        output = run_text(thm_text,og_path)
    except Exception as e:
        print(f'Error {e} on original path: {og_path}\n retrying from root')
        output = run_text(thm_text,root_path)
    '''
    output = run_text(thm_text,root_path)


    if 'messages' in output.keys():
        if any([msg['severity'] == 'error' for msg in output['messages']]):
            return (False,output)
    return (True,output)



if __name__ == '__main__':
    src = 'Tests'
    name = '«Tests»/Basic.lean'

    f = getAnnotatedFile(src,name)
    thms = f.theorems
    
    for thm in thms:
        print(f"RAW: \n\n {thm} \n\nSending to GPT:\n")

        out = prompt_structured(thm,modularity_metric())
        correct = eval_correctness(out)
        print(out)
        if correct is None:
            print(f"VALID!!! {correct}")
        else:
            print(f"INVALID!!! {correct}")

        print('\n\n')

