import json
import os

def fix_haves(src,file_name):
    root_path = os.path.dirname(os.path.abspath(__file__))
    content_path = os.path.join(root_path, f'.cache/{src}')


    with open(os.path.join(content_path,file_name),'r') as f:
        data = [json.loads(jline) for jline in f.read().splitlines()]

    no_incl = []
    for idx,item in list(enumerate(data)):
        tactic = item['tactic']

        if 'have' in tactic:
            cnt = item['tactic'].count('\n')
            for i in range(cnt):
                no_incl.append(idx+i+1)
    output = [data[i] for i in range(len(data)) if i not in no_incl]

    with open(os.path.join(content_path,file_name),'w') as f:
        for entry in output:
            json.dump(entry, f)
            f.write('\n')


fix_haves('Tests3','Tests3/Basic.jsonl')
    