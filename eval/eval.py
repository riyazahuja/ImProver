from metrics import *
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from repl.repl import *
from models.structures import *


def eval_correctness(thm):
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    thm_text = parseTheorem(thm)
    output = run_text(thm_text,root_path)
    if 'messages' in output.keys():
        if any([msg['severity'] == 'error' for msg in output['messages']]):
            return output
    return None

def eval_metric(thm, metric):
    return metric(thm)

def score_metric(old_thm,new_thm,metric):
    old = eval_metric(old_thm,metric)
    new = eval_metric(new_thm,metric)
    return percent_change(old,new)

