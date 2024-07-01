import os
import sys
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.append(str(Path(__file__).parent.parent))
from evaluate.eval import *
from evaluate.metrics import *
from repl.repl import *
from models.structures import *
from models.prompt import *
from evaluate.build_prooftree import *
import shutil
import pandas as pd 
import seaborn as sns
import time
import numpy as np
from tqdm import tqdm
from models.rag import *

metrics = {'LENGTH': length_metric(), 'MODULARITY': modularity_metric()}

def process_theorem(thm, metric, model,method,annotation,rag):
    start_time = time.time()
    promptfn = prompt_rag if rag else prompt_structured
    if method[0] == 'BASIC' or method == 'BASIC':
        out = promptfn(thm,metric,model=model,annotation=annotation)
    elif method[0] == 'REFINEMENT':
        n = method[1]
        kwargs = {}
        if len(method) == 3:
            kwargs = method[2]
        out = refinement(thm,metric,n,model=model,promptfn=promptfn, annotation=annotation,**kwargs)
    elif method[0] == 'BEST':
        n = method[1]
        kwargs = {}
        if len(method) == 3:
            kwargs = method[2]
        out = best_of_n(thm,metric,n,model=model,annotation=annotation,**kwargs)
    else:
        raise ValueError(f"Invalid Method: {method}")
    

    #print(out)
    original_correct,msg = eval_correctness(thm)
    if not original_correct:
        raise ValueError(f'===CTX:===\n {thm.context}\n===DECL:===\n{thm.decl}\n\n ===msg:=== {msg.get("messages",msg)}\n\n===TEST===\n{parseTheorem(thm)}')
    correct,msg = eval_correctness(out)
    #print(f'EVAL: {correct}\n {msg.get("messages",msg)}')
    old_m = metric.metric(thm)
    if correct and original_correct:
        if metric.name == 'MODULARITY' and type(out) == Theorem:
            out = annotateTheorem(out)
        new_m = metric.metric(out)
        delta = metric.delta(thm, out)
        
        if metric.name == 'MODULARITY':
            #print('PRINTING PROOF TREES!!!')
            if os.path.isdir(f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}'):
                shutil.rmtree(f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}')
            G, p, l, _ = getProofTree(thm)
            save_tree(G,p,l,f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}/OG.png')
            G, p, l, _ = getProofTree(out)
            save_tree(G,p,l,f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}/GPT.png')
            #print('DONE PRINTING PROOF TREES!!!')
    else:
        new_m = None
        delta = None

    total_time = time.time()-start_time

    return {
        'decl': thm.decl,
        'original_correct': original_correct,
        'new_correct': correct,
        'original_score': old_m,
        'new_score': new_m,
        'delta': delta,
        'original_raw': parseTheorem(thm,context=False),
        'new_raw': parseTheorem(out,context=False),
        'total_time': total_time
    }


def benchmark_file(src, name, project_path, metric_name, annotation=True, rag=True,model='gpt-4-turbo', method_workers = None,theorem_workers=None, methods=[('BASIC')]):
    #print(f'{src}|{name}')
    start_time = time.time()
    f = getAnnotatedFile(src, name,project_path)
    thms = f.theorems
    metric = metrics[metric_name]
    file_benchmark = {'src': src, 'path': name, 'metric': metric_name, 'model': model, 'num_theorems' : len(thms)}
    if os.path.isdir(f'.trees/{src}/{name}') and metric_name=="MODULARITY":
        shutil.rmtree(f'.trees/{src}/{name}')
    theorem_data=[]
    future_data={}
    max_workers = 1
    if method_workers is None:
        method_workers=len(methods)
    else:
        method_workers=1
    if theorem_workers is None:
        theorem_workers = len(thms)
    else:
        theorem_workers=1
    max_workers = method_workers * theorem_workers
    
    with ThreadPoolExecutor(max_workers=max(1,min(len(thms)*len(methods),max_workers))) as executor:
        
        #with tqdm(total=len(thms)*len(methods)) as pbar:
        future_to_thm = {(thm.declID,method[0]) : executor.submit(process_theorem, thm, metric, model, method, annotation,rag) for thm in thms for method in methods}
            # for pair,future in concurrent.futures.as_completed(future_to_thm):
            #     pbar.update(1)
            #     result = future.result()
            #     ID, method = pair
            #     if ID not in future_data.keys():
            #         future_data[ID]={}
            #     future_data[ID][method] = result
        #print(f'Completed threads: {sum(1 for k in future_to_thm.keys() if future_to_thm[k].done())} / {len(future_to_thm.keys())}')
        for pair,future in future_to_thm.items():
            result = future.result()
            ID, method = pair
            if ID not in future_data.keys():
                future_data[ID]={}
            future_data[ID][method] = result
        
        
    #print(future_data)
    for ID,val in future_data.items():
        data = {}

        for method in methods:
            curr = val[method[0]]
            data.update({
                'decl': curr['decl'],
                'original_correct': curr['original_correct'], 
                f'new_correct_{method}': curr['new_correct'],
                'original_score': curr['original_score'],
                f'new_score_{method}': curr['new_score'],
                f'delta_{method}': curr['delta'],
                'original_raw': curr['original_raw'],
                f'new_raw_{method}':curr['new_raw'],
                f'total_time_{method}': curr['total_time']
                })
        theorem_data.append(data)


    

    file_benchmark['theorems']=theorem_data

    total_time = time.time()-start_time
    # Aggregate results at the file level
    og_corr = sum(1 for v in file_benchmark['theorems'] if v['original_correct'])


    file_benchmark['total_time'] = total_time
    file_benchmark['original_correct'] = og_corr
    file_benchmark['percent_original_correct'] = og_corr/len(thms)*100 if len(thms)!=0 else None

    for method in methods:
        new_corr = sum(1 for v in file_benchmark['theorems'] if v[f'new_correct_{method}'])
        deltas = [v[f'delta_{method}'] for v in file_benchmark['theorems'] if v[f'delta_{method}'] is not None]

        file_benchmark[f'new_correct_{method}'] = new_corr
        file_benchmark[f'percent_new_correct_{method}'] = new_corr/len(thms)*100 if len(thms)!=0 else None
        file_benchmark[f'mean_delta_{method}'] = np.mean(deltas) if deltas != [] else None
        file_benchmark[f'median_delta_{method}'] = np.median(deltas) if deltas != [] else None
        file_benchmark[f'stdev_delta_{method}'] = np.std(deltas) if deltas != [] else None

    return file_benchmark

def correct_print(data):
    correct,output = data
    if correct:
        return 'True'
    else:
        return f'False : {output["messages"]}'
def pretty_print(result,printAll=False):
    out = f'''
src: {result['src']}
path: {result['path']}
metric: {result['metric']}
model: {result['model']}

# original correct: {result['original_correct']}/{len(result['theorems'].keys())} = {result['original_correct']/len(result['theorems'].keys()) * 100}%
# new correct : {result['new_correct']}/{len(result['theorems'].keys())} = {result['new_correct']/len(result['theorems'].keys()) * 100}%
avg delta = {result['avg_delta']}%

Theorems:
'''
    thms = result['theorems']
    thm_txt = ''
    for _,thm in thms.items():
        thm_txt += f'''
===  {thm['decl']}  ===
original correct? {correct_print(thm['original_correct'])} | new correct? {correct_print(thm['new_correct'])}
original score: {thm['original_score']} | new score: {thm['new_score']}
delta = {thm['delta']}%

og: {parseTheorem(thm['original_raw'],False) if printAll else ''}

new: {parseTheorem(thm['new_raw'],False) if printAll else ''}

'''
    out += thm_txt
    return out

#METHODS: 
# ('BASIC') is a standard, basic prompt.
# ('REFINEMENT', n, {prev_num=1, keep_best=False}) is an n-shot iterative refinement
# ('BEST', n, {max_workers=n}) is best of n with n workers
def benchmark_repo(src, metric_name, proj_path, annotation = True, rag=True, model='gpt-4-turbo',start = '',ignore=[], methods = [('BASIC')],method_workers = None,theorem_workers=None):
    start_time = time.time()

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #proj_path=os.path.join(root_path,'.lake','packages','Tests3')

    cache_path = os.path.join(root_path, '.cache')
    src_path = os.path.join(cache_path, src)
    start_path = os.path.join(src_path,start)
    if os.path.isdir(f'.trees/{src}') and metric_name=='MODULARITY':
        shutil.rmtree(f'.trees/{src}')
    file_names = []
    abs_ignore = [os.path.join(src_path,ig) for ig in ignore]
    for root, dirs, files in os.walk(start_path):
        rel_ignore = [os.path.relpath(ig,start=start_path) for ig in abs_ignore]
        dirs[:] = [d for d in dirs if d not in rel_ignore]
        for file in files:
            path = os.path.relpath(os.path.join(root, file), start=src_path)
            if get_stem(path) not in ignore and path.endswith('.json'):
                file_names.append(path)
    if os.path.isfile(get_stem(start_path)+'.json'):
        file_names=[get_stem(os.path.relpath(start_path, start=src_path))+'.json']
    if any([start.startswith(ig) for ig in ignore]):
        file_names=[]
    #print(file_names)

    file_data = []
    for name in file_names: #TODO PARALLELIZE
        result = benchmark_file(src, get_stem(name)+'.lean', proj_path, metric_name, annotation=annotation,rag=rag, model=model, methods=methods,method_workers=method_workers,theorem_workers=theorem_workers)
        file_data.append(result)
    #print(results)

    total_time = time.time()-start_time

    repo_data = {
        'name': src,
        'metric': metric_name,
        'model': model,
        'total_time': total_time,
        'number_of_files': len(file_names),
        'number_of_theorems': sum(file['num_theorems'] for file in file_data),
        'num_original_correct': sum(file['original_correct'] for file in file_data),
        'percent_original_correct': sum(file['original_correct'] for file in file_data)/sum(x['num_theorems'] for x in file_data)*100,
        'files': file_data
    }

    for method in methods:
        all_deltas = [thm[f'delta_{method}'] for file in file_data for thm in file['theorems'] if thm[f'delta_{method}'] is not None]

        repo_data[f'num_new_correct_{method}']= sum(file[f'new_correct_{method}'] for file in file_data)
        repo_data[f'percent_new_correct_{method}']= sum(file[f'new_correct_{method}'] for file in file_data)/sum(x['num_theorems'] for x in file_data)*100
        repo_data[f'mean_delta_{method}']= np.mean(all_deltas) if all_deltas else None
        repo_data[f'median_delta_{method}']= np.median(all_deltas) if all_deltas else None
        repo_data[f'stdev_delta_{method}']= np.std(all_deltas) if all_deltas else None


    return repo_data

def save_to_csv(repo_data, methods=[('BASIC')], thm_path='theorem_data.csv', file_path='file_data.csv', repo_path='repository_data.csv',raw=False):
    file_data = repo_data['files']
    thm_data = [thm for file in file_data for thm in file['theorems'] ]
    del repo_data['files']
    for file in file_data:
        del file['theorems']
    if not raw:
        for thm in thm_data:
            del thm['original_raw']
            for method in methods:
                del thm[f'new_raw_{method}']
    
    repo_df = pd.DataFrame([repo_data])
    files_df = pd.DataFrame(file_data)
    thms_df = pd.DataFrame(thm_data)

    repo_df.to_csv(repo_path,index=False)
    files_df.to_csv(file_path,index=False)
    thms_df.to_csv(thm_path,index=False)



if __name__ == "__main__":
    methods = ['BASIC']
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

   #proj_path=os.path.join(root_path,'.lake','packages','Tests3')
    proj_path = '/Users/ahuja/Desktop/LeanTestData/Tests'

    repo_data = benchmark_repo('Tests', 'LENGTH', proj_path=proj_path,start='Tests/C04_Sets_and_Functions/solutions/Solutions_S01_Sets.lean',model='gpt-4-turbo',methods = methods)
    save_to_csv(repo_data,methods=methods,raw=True)
    
    #repo_data2 = benchmark_repo('Tests3', 'LENGTH', proj_path=proj_path,start='Tests3/all.lean',model='gpt-4-turbo',methods = methods)
    #save_to_csv(repo_data2,methods=methods,raw=True)
    

