from metrics import *
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from repl.repl import *
from models.structures import *
from models.prompt import prompt_structured


def eval_correctness(thm):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    thm_text = parseTheorem(thm)
    print(thm_text)
    print(root_path)
    output = run_text(thm_text,root_path)
    if 'messages' in output.keys():
        if any([msg['severity'] == 'error' for msg in output['messages']]):
            return output
    return None



if __name__ == '__main__':
    src = 'Tests'
    name = '«Tests»/Basic.lean'

    f = getAnnotatedFile(src,name)
    thms = f.theorems
    
    for thm in thms:
        print(f"RAW: \n\n {thm} \n\nSending to GPT:\n")

        out = prompt_structured(thm,length_metric())
        correct = eval_correctness(out)
        print(out)
        if correct is None:
            print(f"VALID!!! {correct}")
        else:
            print(f"INVALID!!! {correct}")

        print('\n\n')

