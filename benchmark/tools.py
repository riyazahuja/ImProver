import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from evaluate.eval import *
from repl.repl import *
from models.structures import *
from models.prompt import prompt_structured




def benchmark_file(src,name,metric=None,model='gpt-4-turbo'):
    f = getAnnotatedFile(src,name)
    thms = f.theorems
    
    for thm in thms:
        print(f"RAW: \n\n {thm} \n\nSending to GPT:\n")

        out = prompt_structured(thm,model=model)
        original_correct = eval_correctness(thm)
        correct = eval_correctness(out)
        old_m = eval_metric(thm,metric)
        new_m = eval_metric(out,metric)
        pcent = score_metric(thm,out,length_metric)

        data = {
            'src': src,
            'file_name' : name,
            'theorem' : thm.decl,
            'metric' : metric,
            'original_correct' : original_correct,
            'new_correct' : correct,
            'original_score' : old_m,
            'new_score': new_m,
            'delta' : pcent,
            'original_raw' : thm,
            'new_raw' : out
        }