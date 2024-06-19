import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.append(str(Path(__file__).parent.parent))
from evaluate.eval import *
from evaluate.metrics import *
from repl.repl import *
from models.structures import *
from models.prompt import *
from evaluate.build_prooftree import *
import shutil
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

metrics = {'LENGTH': length_metric(), 'MODULARITY': modularity_metric()}

def process_theorem(thm, metric, model):
    start_time = time.time()
    #out = refinement(thm, metric, 5, model=model)
    out = prompt_structured(thm,metric)
    #print(out)
    original_correct,old_out = eval_correctness(thm)
    correct,new_out = eval_correctness(out)
    old_m = metric.metric(thm)
    if correct and original_correct:
        if metric.name == 'MODULARITY' and type(out) == Theorem:
            out = annotateTheorem(out)
        new_m = metric.metric(out)
        delta = metric.delta(thm, out)

        if metric.name == 'MODULARITY':
            if os.path.isdir(f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}'):
                shutil.rmtree(f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}')
            G, p, l, _ = getProofTree(thm)
            save_tree(G,p,l,f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}/OG.png')
            G, p, l, _ = getProofTree(out)
            save_tree(G,p,l,f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}/GPT.png')
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

def benchmark_file(src, name, metric_name, model='gpt-4-turbo', max_workers=3):
    start_time = time.time()
    f = getAnnotatedFile(src, name)
    thms = f.theorems
    metric = metrics[metric_name]
    file_benchmark = {'src': src, 'path': name, 'metric': metric_name, 'model': model, 'num_theorems' : len(thms)}
    if os.path.isdir(f'.trees/{src}/{name}') and metric_name=="MODULARITY":
        shutil.rmtree(f'.trees/{src}/{name}')
    theorem_data=[]
    with ThreadPoolExecutor(max_workers=min(len(thms),max_workers)) as executor:
        future_to_thm = [executor.submit(process_theorem, thm, metric, model) for thm in thms]
        for future in future_to_thm:
            result = future.result()
            theorem_data.append(result)

    file_benchmark['theorems']=theorem_data

    total_time = time.time()-start_time
    # Aggregate results at the file level
    og_corr = sum(1 for v in file_benchmark['theorems'] if v['original_correct'])
    new_corr = sum(1 for v in file_benchmark['theorems'] if v['new_correct'])
    deltas = [v['delta'] for v in file_benchmark['theorems'] if v['delta'] is not None]

    file_benchmark['total_time'] = total_time
    file_benchmark['original_correct'] = og_corr
    file_benchmark['new_correct'] = new_corr
    file_benchmark['percent_original_correct'] = og_corr/len(thms)*100
    file_benchmark['percent_new_correct'] = new_corr/len(thms)*100
    file_benchmark['mean_delta'] = np.mean(deltas) if deltas != [] else None
    file_benchmark['median_delta'] = np.median(deltas) if deltas != [] else None
    file_benchmark['stdev_delta'] = np.std(deltas) if deltas != [] else None

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


def benchmark_repo(src, metric_name, model='gpt-4-turbo'):
    start_time = time.time()
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, '.cache')
    src_path = os.path.join(cache_path, src)
    if os.path.isdir(f'.trees/{src}') and metric_name=='MODULARITY':
        shutil.rmtree(f'.trees/{src}')
    file_names = []
    for root, _, files in os.walk(src_path):
        for file in files:
            if file.endswith('.jsonl'):
                path = os.path.relpath(os.path.join(root, file), start=src_path)
                file_names.append(path)
    file_data = []
    for name in file_names: #TODO PARALLELIZE
        result = benchmark_file(src, name, metric_name, model)
        file_data.append(result)
    #print(results)

    total_time = time.time()-start_time

    all_deltas = [thm['delta'] for file in file_data for thm in file['theorems'] if thm['delta'] is not None]

    repo_data = {
        'name': src,
        'metric': metric_name,
        'model': model,
        'total_time': total_time,
        'number_of_files': len(file_names),
        'num_original_correct': sum(file['original_correct'] for file in file_data),
        'num_new_correct': sum(file['new_correct'] for file in file_data),
        'percent_original_correct': sum(file['original_correct'] for file in file_data)/sum(x['num_theorems'] for x in file_data)*100,
        'percent_new_correct': sum(file['new_correct'] for file in file_data)/sum(x['num_theorems'] for x in file_data)*100,
        'mean_delta': np.mean(all_deltas) if all_deltas else None,
        'median_delta': np.median(all_deltas) if all_deltas else None,
        'stdev_delta': np.std(all_deltas) if all_deltas else None,
        'files': file_data
    }
    return repo_data

def save_to_csv(repo_data, thm_path='theorem_data.csv', file_path='file_data.csv', repo_path='repository_data.csv'):
    file_data = repo_data['files']
    thm_data = [thm for file in file_data for thm in file['theorems'] ]
    del repo_data['files']
    for file in file_data:
        del file['theorems']
    #for thm in thm_data:
        #del thm['original_raw']
        #del thm['new_raw']
    repo_df = pd.DataFrame([repo_data])
    files_df = pd.DataFrame(file_data)
    thms_df = pd.DataFrame(thm_data)

    repo_df.to_csv(repo_path,index=False)
    files_df.to_csv(file_path,index=False)
    thms_df.to_csv(thm_path,index=False)

def save_repo_data_to_csv(repo_data):
    repo_df = pd.DataFrame([{
        'name': repo_data['name'],
        'model': repo_data['model'],
        'metric_name': repo_data['metric'],
        'total_time': repo_data['total_time'],
        'number_of_files': repo_data['number_of_files'],
        'number_original_thms_correct': repo_data['num_original_correct'],
        'number_new_thms_correct': repo_data['num_new_correct'],
        'percent_original_thms_correct': repo_data['percent_original_correct'],
        'percent_new_thms_correct': repo_data['percent_new_correct'],
        'mean_delta': repo_data['mean_delta'],
        'median_delta': repo_data['median_delta'],
        'stdev_delta': repo_data['stdev_delta']
    }])
    repo_df.to_csv('repository_data.csv', index=False)

def save_file_data_to_csv(repo_data):
    file_records = []
    for file in repo_data['files']:
        file_records.append({
            'repo': file['src'],
            'name': file['path'],
            'model': file['model'],
            'metric': file['metric'],
            'total_time': file['total_time'],
            'num_theorems': file['num_theorems'],
            'number_original_thms_correct': file['original_correct'],
            'number_new_thms_correct': file['new_correct'],
            'percent_original_thms_correct': file['percent_original_correct'],
            'percent_new_thms_correct': file['percent_new_correct'],
            'mean_delta': file['mean_delta'],
            'median_delta': file['median_delta'],
            'stdev_delta': file['stdev_delta']
        })
    file_df = pd.DataFrame(file_records)
    file_df.to_csv('file_data.csv', index=False)


def save_theorem_data_to_csv(repo_data):
    theorem_records = []
    for file in repo_data['files']:
        for thm in file['theorems']:
            theorem_records.append({
                'repo_name': repo_data['name'],
                'file_name': file['path'],
                'decl': thm['decl'],
                'model': file['model'],
                'metric': file['metric'],
                'total_time': thm['total_time'],
                'original_correct': thm['original_correct'],
                'new_correct': thm['new_correct'],
                'original_score': thm['original_score'],
                'new_score': thm['new_score'],
                'delta': thm['delta']
            })
    theorem_df = pd.DataFrame(theorem_records)
    theorem_df.to_csv('theorem_data.csv', index=False)



if __name__ == "__main__":
    repo_data = benchmark_repo('Tests3', 'LENGTH')
    #for f in repo_data:
    #    print(pretty_print(f,True))
    #print('\n\nDATAFRAMING\n')
    save_to_csv(repo_data)
    #save_repo_data_to_csv(repo_data)
    #save_file_data_to_csv(repo_data)
    #save_theorem_data_to_csv(repo_data)
    
