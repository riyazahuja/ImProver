import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from repl.repl import *
from models.structures import *
from evaluate.metrics import *

def eval_correctness(thm):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = thm.src
    lean_file = thm.leanFile
    pardir = os.path.dirname(lean_file)

    og_path = os.path.join(root_path,'.lake/packages',src)

    if (type(thm) is AnnotatedTheorem):
        thm_text = parseAnnotatedTheorem(thm)
    else:
        thm_text = parseTheorem(thm)
    #print(thm_text)
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