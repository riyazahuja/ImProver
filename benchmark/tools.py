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

metrics = {'LENGTH': length_metric(), 'MODULARITY': modularity_metric()}

def process_theorem(thm, metric, model):
    #print(f"RAW: \n\n {thm} \n\nSending to GPT:\n")
    out = best_of_n(thm, metric, 5, model=model)
    print(out)
    original_correct,old_out = eval_correctness(thm)
    correct,new_out = eval_correctness(out)
    old_m = metric.metric(thm)
    if correct and original_correct:
        if metric.name == 'MODULARITY' and type(out) == Theorem:
            out = annotateTheorem(out)
        new_m = metric.metric(out)
        delta = metric.delta(thm, out)
        if metric.name == 'MODULARITY':
            #print(thm.decl)
            #print(out.decl)
            if os.path.isdir(f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}'):
                shutil.rmtree(f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}')
            G, p, l, _ = getProofTree(thm)
            save_tree(G,p,l,f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}/OG.png')
            G, p, l, _ = getProofTree(out)
            save_tree(G,p,l,f'.trees/{thm.src}/{thm.leanFile}/{thm.decl}/GPT.png')
    else:
        new_m = None
        delta = None

    return {
        'decl': thm.decl,
        'original_correct': (original_correct,old_out),
        'new_correct': (correct,new_out),
        'original_score': old_m,
        'new_score': new_m,
        'delta': delta,
        'original_raw': thm,
        'new_raw': out
    }

def benchmark_file(src, name, metric_name, model='gpt-4-turbo', max_workers=3):
    f = getAnnotatedFile(src, name)
    thms = f.theorems
    metric = metrics[metric_name]
    file_benchmark = {'src': src, 'path': name, 'metric': metric_name, 'model': model, 'theorems': {}}
    if os.path.isdir(f'.trees/{src}/{name}') and metric_name=="MODULARITY":
        shutil.rmtree(f'.trees/{src}/{name}')
    with ThreadPoolExecutor(max_workers=min(len(thms),max_workers)) as executor:
        future_to_thm = {executor.submit(process_theorem, thm, metric, model): thm.declID for thm in thms}
        for future in future_to_thm:
            result = future.result()
            file_benchmark['theorems'][future_to_thm[future]] = result

    # Aggregate results at the file level
    og_corr = sum(1 for v in file_benchmark['theorems'].values() if v['original_correct'][0])
    new_corr = sum(1 for v in file_benchmark['theorems'].values() if v['new_correct'][0])
    deltas = [v['delta'] for v in file_benchmark['theorems'].values() if v['delta'] is not None]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0

    file_benchmark['original_correct'] = og_corr
    file_benchmark['new_correct'] = new_corr
    file_benchmark['avg_delta'] = avg_delta
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

og: {parseTheoremAny(thm['original_raw'],False) if printAll else ''}

new: {parseTheoremAny(thm['new_raw'],False) if printAll else ''}

'''
    out += thm_txt
    return out


def benchmark_repo(src, metric_name, model='gpt-4-turbo'):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, '.cache')
    src_path = os.path.join(cache_path, src)
    if os.path.isdir(f'.trees/{src}') and metric_name=='MODULARITY':
        shutil.rmtree(f'.trees/{src}')
    file_names = []
    for root, _, files in os.walk(src_path):
        for file in files:
            if file.endswith('.lean'):
                path = os.path.relpath(os.path.join(root, file), start=src_path)
                file_names.append(path)
    results = []
    for name in file_names:
        result = benchmark_file(src, name, metric_name, model)
        results.append(result)
    
     # Optionally print or handle the file benchmark results
    return results

if __name__ == "__main__":
    output = benchmark_repo('Tests3', 'LENGTH')
    for f in output:
        print(pretty_print(f,True))
