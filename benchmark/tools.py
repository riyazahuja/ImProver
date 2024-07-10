import os
import sys
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.append(str(Path(__file__).parent.parent))
from evaluate.eval import *
from evaluate.metrics import *
from models.structures import *
from models.prompt import *
from evaluate.build_prooftree import *
import shutil
import pandas as pd 
import seaborn as sns
import time
from models.rag import *
import itertools


def process_instance(thm:AnnotatedTheorem,method):
    start_time = time.time()
    fn, metric, kwargs = method
    
    og_correct,og_messages,_ = eval_correctness(thm)
    og_score = None
    if og_correct:
        og_score=metric.metric(thm)
    
    output_thm = fn(thm,metric,**kwargs)

    new_correct,new_messages,output_anno_thm = eval_correctness(output_thm)
    processing_time = time.time()-start_time
    new_score = None
    if new_correct:
        new_score=metric.metric(output_anno_thm)
    
    delta = None
    if new_score is not None and og_score is not None:
        delta = ((new_score-og_score)/og_score) * 100

    og_raw = parseTheorem(thm,context=False)
    new_raw = parseTheorem(output_thm,context=False)
    #print(parseTheorem(output_thm,context=False))

    def parse_msg(message):
        return f'{message.content}\n\tat: {message.message_src}'
    
    og_errors = '\n'.join([parse_msg(msg) for msg in og_messages if msg.severity=='error'])
    new_errors = '\n'.join([parse_msg(msg) for msg in new_messages if msg.severity=='error'])

    return {
        'repo': thm.src,
        'file': thm.leanFile,
        'decl': thm.decl,
        'method': fn.__name__,
        'n' : kwargs.get('n',None) if fn.__name__ !='prompt_structured' else None,
        'metric': metric.name,
        'model': kwargs.get('model','gpt-4-turbo'),
        'annotation': kwargs.get('annotation',True),
        'syntax_search': kwargs.get('syntax_search',False),
        'mathlib_search': kwargs.get('mathlib_search',False),
        'og_correct': og_correct,
        'og_errors': og_errors,
        'og_score': og_score,
        'new_correct': new_correct,
        'new_errors': new_errors,
        'new_score': new_score,
        'delta':delta,
        'og_raw':og_raw,
        'new_raw':new_raw,
        'time': processing_time
    }

def benchmark_theorem(thm:AnnotatedTheorem, methods, method_workers=None):
    if method_workers is None:
        method_workers=len(methods)
    with ThreadPoolExecutor(max_workers=min(method_workers,len(methods))) as executor:
        futures = [executor.submit(process_instance,thm,method) for method in methods]
    data = [future.result() for future in futures]
    return data

def benchmark_file(file:AnnotatedFile, methods, theorem_workers=None, method_workers=None):
    thms = file.theorems
    if theorem_workers is None:
        theorem_workers=len(thms)

    with ThreadPoolExecutor(max_workers=min(theorem_workers,len(thms))) as executor:
        futures = [executor.submit(benchmark_theorem,thm,methods,method_workers=method_workers) for thm in thms]
    data = [x for future in futures for x in future.result()]
    return data


def benchmark_repo(repo:Repo,methods,file_workers = None, theorem_workers=None, method_workers=None):
    anno_files = [f for f in repo.files if type(f)==AnnotatedFile]
    if file_workers is None:
        file_workers = len(anno_files)

    with ThreadPoolExecutor(max_workers=min(file_workers,len(anno_files))) as executor:
        futures = [executor.submit(benchmark_file,f,methods,theorem_workers=theorem_workers,method_workers=method_workers) for f in anno_files]
    data = [x for future in futures for x in future.result()]
    return data



def save_to_csv(data,path='data.csv'):
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path,index=False)

def get_methods(fn=[prompt_structured],
                metric=[length_metric()],
                annotation=[True],
                model=['gpt-4-turbo'],
                n=[1],
                syntax_search=[False],
                mathlib_search=[False]):
    dl = [fn,annotation,model,n,syntax_search,mathlib_search,metric]
    prod = list(itertools.product(*dl))
    return [
        (i[0],i[6],{'annotation':i[1],
                'model':i[2],
                'n':i[3],
                'syntax_search':i[4],
                'mathlib_search':i[5]
                })
        for i in prod
    ]


#Methods look like (fn:callable, kwargs : dict)
if __name__ == "__main__":

    #methods = [(prompt_structured,length_metric(),{})]
    methods = get_methods(model=['gpt-4o'],
                          fn=[best_of_n,refinement],n=[3],
                          syntax_search=[True],
                          mathlib_search=[True])
    repo = getRepo('Tests','configs/config_test.json')
    files = {file.file_name:file for file in repo.files}
    f = files['Basic.lean']
    f2 = files['Basic2.lean']


    data = benchmark_file(f,methods)
    data.extend(benchmark_file(f2,methods))
    save_to_csv(data)

    

