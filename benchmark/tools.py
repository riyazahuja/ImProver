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
from tqdm import tqdm


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
    if new_score is not None and og_score is not None and og_score != 0:
        delta = ((new_score-og_score)/og_score) * 100

    og_raw = parseTheorem(thm,context=False)
    new_raw = parseTheorem(output_thm,context=False)
    #print(parseTheorem(output_thm,context=False))

    def parse_msg(message):
        return f'{message.content}\n\tat: {message.message_src}'
    
    og_errors = '\n'.join([parse_msg(msg) for msg in og_messages if msg.severity=='error' or (msg.severity=='warning' and 'sorry' in msg.content)])
    new_errors = '\n'.join([parse_msg(msg) for msg in new_messages if msg.severity=='error'or (msg.severity=='warning' and 'sorry' in msg.content)])

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

def benchmark_theorem(thm:AnnotatedTheorem, methods, method_workers=None,show_method_progress=False):
    if method_workers is None:
        method_workers=len(methods)

    if show_method_progress:
        with tqdm(total=len(methods),desc='Methods: ') as pbar:
            with ThreadPoolExecutor(max_workers=min(method_workers,len(methods))) as executor:
                futures = [executor.submit(process_instance,thm,method) for method in methods]

                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=min(method_workers,len(methods))) as executor:
            futures = [executor.submit(process_instance,thm,method) for method in methods]



    data = [future.result() for future in futures]
    return data

def benchmark_file(file:AnnotatedFile, methods, theorem_workers=None, method_workers=None,show_theorem_progress=False,show_method_progress=False):

    thms = file.theorems
    if theorem_workers is None:
        theorem_workers=len(thms)

    if show_theorem_progress:
        with tqdm(total=len(thms),desc='Theorems: ') as pbar:
            with ThreadPoolExecutor(max_workers=min(theorem_workers,len(thms))) as executor:
                futures = [executor.submit(benchmark_theorem,thm,methods,method_workers=method_workers,show_method_progress=show_method_progress) for thm in thms]

                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=min(theorem_workers,len(thms))) as executor:
                futures = [executor.submit(benchmark_theorem,thm,methods,method_workers=method_workers,show_method_progress=show_method_progress) for thm in thms]


    data = [x for future in futures for x in future.result()]

    return data


def benchmark_repo(repo:Repo,methods,file_workers = None, theorem_workers=None, method_workers=None,show_file_progress=False,show_theorem_progress=False):
    anno_files = [f for f in repo.files if type(f)==AnnotatedFile]
    if file_workers is None:
        file_workers = len(anno_files)

    if show_file_progress:
        with tqdm(total=len(anno_files),desc='Files: ') as pbar:
            with ThreadPoolExecutor(max_workers=min(file_workers,len(anno_files))) as executor:
                futures = [executor.submit(benchmark_file,f,methods,theorem_workers=theorem_workers,method_workers=method_workers,show_theorem_progress=show_theorem_progress) for f in anno_files]

                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=min(file_workers,len(anno_files))) as executor:
            futures = [executor.submit(benchmark_file,f,methods,theorem_workers=theorem_workers,method_workers=method_workers,show_theorem_progress=show_theorem_progress) for f in anno_files]

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

# def parse_informal_thm(data,src='',leanFile='',project_path='',context_preparsing=False):
#     context = '/-<INFORMAL PROOF>-/\n'+data.get('context','')
#     statement = data.get('statement','')
#     proof = data.get('proof','')

#     if context_preparsing:
#         print('==== Preparsing context ====')
#         def temp_metric ():  
#             def num_errors(thm):
#                 if type(thm) == Theorem:
#                     thm = annotateTheorem(thm)
#                 errors = sum(1 for msg in thm.messages if msg.severity=='error')
#                 return errors

#             sys_prompt = ('system','You are an AI assistant who automatically formalizes LaTeX context, proofs, and definitions into Lean 4 context, proofs, and definitions and ensures their correctness. You will recieve a string form of this latex document and you will have to return a correct lean 4 parsed version.')
#             user_prompt = ('human','Return a correct lean4 formatted version of the current input.')

#             return Metric('TEMP', [sys_prompt,user_prompt], num_errors, 'MIN')
        
#         context = prompt_basic(context,temp_metric(),model='gpt-4o')
#         print(f'==== DONE Preparsing context ====\n{context}\n=================')
    


#     return AnnotatedTheorem(decl=statement,
#                             declID='',
#                             src=src,
#                             leanFile=leanFile,
#                             context=context,
#                             proof=[AnnotatedProofStep(prevState=[],tactic=proof,nextState=[],srcUpToTactic='',declUpToTactic='',start=(None,None),end=(None,None))],
#                             project_path=project_path,
#                             messages=[],
#                             pretty_print=context+'\n'+statement+'\n'+proof)




#Methods look like (fn:callable, kwargs : dict)
if __name__ == "__main__":

    
    #DEMO methods:
    # files -> MIL Set theory
    # Baseline performance (1 shot, gpt 4o, gpt 4 turbo, no annotation, no rag, no fn, both metrics,flat) - 14:24
    # Annotation+structure performance (1 shot, gpt-4o, annotation, flat/structured, both metrics) - ~10
    # Method Performance (gpt 4o, annotation, flat + refinement + best of n, n=5,10, (keep best) both metrics?) - 4 hours???
    # RAG Performance
    # Completion performance (bonus)
    # (Fake) autoformalization example





    methods = get_methods(model=['gpt-4o'],
                          fn=[refinement(prompt_flat)],
                          n=[5],
                          metric=[completion_metric()],
                          mathlib_search=[True],
                          syntax_search=[True]
                          )
    repo = getRepo('Tests','configs/config_test.json')
    files = {file.file_name:file for file in repo.files}
    #file = files['Basic.lean']
    #keys= [k for k in files.keys() if 'Solutions_' in k]
    keys = ['Solutions_S06_Sequences_and_Convergence.lean', 'Solutions_S03_Negation.lean', 'Solutions_S01_Implication_and_the_Universal_Quantifier.lean', 'Solutions_S04_Conjunction_and_Iff.lean', 'Solutions_S05_Disjunction.lean', 'Solutions_S02_The_Existential_Quantifier.lean']
    #files = [files[k] for k in keys]
    print([f'{k} : {len(files[k].theorems)}\n' for k in keys])
    f = files['S01_Implication_and_the_Universal_Quantifier.lean']
    
    data = []
    #with tqdm(total=len(files),desc='Files: ') as pbar:
        #for f in files:
            #data.extend(benchmark_file(f,methods,theorem_workers=4,method_workers=None,show_theorem_progress=True,show_method_progress=True))
            #pbar.update(1)
    data.extend(benchmark_file(f,methods,theorem_workers=6,method_workers=None,show_theorem_progress=True,show_method_progress=True))
    
    #data.extend(benchmark_file(f2,methods,show_theorem_progress=True))
    #data = benchmark_repo(repo,methods,file_workers=1,theorem_workers=6,show_file_progress=True)
    save_to_csv(data,path='completion_data.csv')

    

