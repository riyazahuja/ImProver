from __future__ import annotations
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Union,Tuple,Dict
import os
import json
import tempfile
import subprocess 
from textwrap import indent

class ProofStep(BaseModel):
    tactic : Union[str,Theorem] = Field(description="One line/tactic in a tactic proof (str) or a subtheorem/sublemma/subproof")

class Theorem(BaseModel):
    decl : str = Field(description="Theorem declaration")
    proof : List[ProofStep] = Field(..., description="Sequence of proofsteps for full proof of theorem.")
    declID : str = Field(description="Unique theorem declaration ID")
    src : str = Field(description="Source repo of the theorem")
    leanFile : str = Field(description="Lean file in which theorem is located")
    context : str = Field(description="Context of the theorem (i.e. file contents up to decl)")
    
class File(BaseModel):
    src : str = Field(description="File source repo")
    file_name : str = Field(description="File Name")
    file_path : str
    file_type : str
    contents : str = Field(description= "File contents")

class AnnotatedProofStep(BaseModel):
    prevState : List[str] = Field(description="Pretty printed tactic st ate before the tactic invocation")
    tactic : str = Field(description="One line/tactic in a tactic proof.")
    nextState : List[str] = Field(description="Pretty printed tactic state after the tactic invocation")
    srcUpToTactic : str = Field(description="Source code from file start to current tactic")
    declUpToTactic : str = Field(description="Source code from theorem declaration to current tactic")
    start : int = Field(description='start UTF-8 byte position of tactic invocation')
    end : int = Field(description='end UTF-8 byte position of tactic invocation')

class AnnotatedTheorem(BaseModel):
    decl : str = Field(description="Theorem declaration")
    declID : str = Field(description="Unique theorem declaration ID")
    src : str = Field(description="Source repo of the theorem")
    leanFile : str = Field(description="Lean file in which theorem is located")
    context : str = Field(description="Context of the theorem (i.e. file contents up to decl)")
    proof : List[AnnotatedProofStep] = Field(..., description="Sequence of annotated proofsteps for full proof of theorem.")

class AnnotatedFile(BaseModel):
    src : str = Field(description="File source repo")
    file_name : str = Field(description="File Name")
    contents : str = Field(description= "File contents")
    theorems : List[AnnotatedTheorem] = Field(..., description="List of all theorems in a file")

class Repo(BaseModel):
    url: str 
    commit : str 
    version : str 
    name: str 
    owner : str 
    import_file: str
    imports: List[str]
    dependencies: List[Repo]
    files: List[Union[AnnotatedFile,File]]







def getTheorems(data,src, file) -> List[AnnotatedTheorem]:
    temp = {}
    for step in data:
        ps = AnnotatedProofStep(prevState=step['prevState'],
                                        tactic = step['tactic'],
                                        nextState=step['nextState'],
                                        srcUpToTactic=step['srcUpToTactic'],
                                        declUpToTactic=step['declUpToTactic'],
                                        start = int(step['start']),
                                        end= int(step['end']))
        decl = step['decl']
        declID = step['declId']

        lines_src = step['srcUpToTactic'].split('\n')
        decl_lines = decl.split('\n')
        #print(lines_src)
        
        #lines = [line for line in lines_src if not line in decl_lines]
        maybe_context = '\n'.join(lines_src[:-len(decl_lines)-1]).strip()

        if declID not in temp.keys():
            temp[declID] = {'proof':[ps], 'decl':decl,'context' : maybe_context}
            #print(temp)
        else:
            #print(temp)
            curr_proof = temp[declID]['proof']
            curr_decl = temp[declID]['decl']
            curr_ctxt = temp[declID]['context']
            curr_proof.append(ps)
            temp[declID] = {'proof':curr_proof, 'decl':curr_decl, 'context':curr_ctxt}
            
    result = {}
    for ID,value in temp.items():
        result[ID] = AnnotatedTheorem(leanFile=file,src=src,decl=value['decl'],declID=ID,proof=value['proof'],context = value['context'])
    return [v for _,v in result.items()]
        

def get_stem(path):
    #print(f'{path} : {path[-5:]} : {path[:-5]}')
    if path[-5:]=='.lean':
        return path[:-5]
    elif path[-6:]=='.jsonl':
        return path[:-6]
    return path

def getAnnotatedFile(src, file_name):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, f'.cache/{src}')
    lake_path =  os.path.join(root_path, f'.lake/packages/{src}')

    contents = None
    data = None

    file_name = get_stem(file_name)
    print(f'{root_path}|.lake/packages/{src}|{file_name}.lean')

    contents = ''
    try:
        with open(os.path.join(lake_path,file_name+'.lean'),'r') as f:
            contents = f.read()
    except:
        pass


    with open(os.path.join(cache_path,file_name+'.jsonl'),'r') as f:
        data = [json.loads(jline) for jline in f.read().splitlines()]
    
    theorems = getTheorems(data,src, file_name)
    #print(theorems)

    return AnnotatedFile(src=src,file_name=file_name,contents = contents,theorems=theorems)
    



#TODO IMPLEMENT LOCAL CONFIGS TO ADD LOCAL CONFIGS TO GETREPO


def getRepo(src,config=None,annotate=True,force=False):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config is None:
        config_files = []
        for root,_,files in os.walk(os.path.join(root_path,'configs')):
            for file in files:
                path = os.path.join(root, file)
                if path.endswith('.json'):
                    config_files.append(path)
        for path in config_files:
            with open(path,'r') as f:
                data = json.load(f)
            if src in [item.get('name','') for item in data]:
                config = os.path.relpath(path,start=root_path)
                break
        if config is None:
            raise ValueError(f'{src} config not found')
    
    config_path = os.path.join(root_path,config)
    with open(config_path,'r') as f:
        data = json.load(f)
    data = [item for item in data if src==item.get('name','')]
    if len(data) == 0:
        raise ValueError(f'{src} not in config file {config}')
    repo_data = data[0]
    url = repo_data.get('repo','')
    commit = repo_data.get('commit','')
    version = repo_data.get('lean','')
    name = repo_data.get('name','')
    import_file = repo_data.get('import_file','')
    imports = repo_data.get('imports',[])

    #depedencies - assume in env (ie lake build has been run):
    package_path = os.path.join(root_path,'.lake','packages')
    dependency_names = [item for item in os.listdir(package_path) if os.path.isdir(os.path.join(package_path, item)) and item!=name]
    
















def parseAnnotatedTheorem(thm,context=True,annotation=False):
    last_pstep = thm.proof[-1]
    if context:
        src = last_pstep.srcUpToTactic+last_pstep.tactic
    else:
        src = last_pstep.declUpToTactic+last_pstep.tactic
    return src

def elim_overlap(pf: List[AnnotatedProofStep]):
    ptr = 0
    output = []
    for step in pf:
        start = step.start
        end = step.end
        if start <= ptr and end <= ptr:
            #this is inside a have
            pass
        else:
            ptr = max(start,end)
            output.append(step)
    return output

def annotate(step : AnnotatedProofStep, prev_goal=False):
    prev = step.prevState
    tactic = step.tactic
    next = step.nextState
    def pp_state(state):
        if state == []:
            return 'No Goals Left!'
        return "\n".join(state)
    
    prev_text = f'''
/-
{pp_state(prev)}
-/
'''    
    text = f'''
{tactic}
/-
{pp_state(next)}
-/
'''
    if prev_goal:
        return indent(prev_text+text,'  ', lambda x: True)
    else:
        return indent(text,'  ', lambda x: True)
    

def parseAnnotatedTheorem2(thm,context=True,annotation=False,prompt=False):
    statement = thm.decl
    if context:
        context = thm.context
    else:
        context = ''

    psteps = elim_overlap(thm.proof)

    proof = ''
    for i in range(len(psteps)):
        if annotation:
            text = annotate(psteps[i],i==0)
        else:
            text = psteps[i].tactic
        proof = proof + '  ' + text + '\n'
        
    if prompt:
        return f'CONTEXT:\n {context}\n\n THEOREM: {statement} := by\n{proof}'
    else:
        return f'{context}\n\n{statement} := by\n{proof}'


def parse_proof(thm,indent = 1,dot=False):
    output = ''
    spaces = '  '
    proof = thm.proof
    for step in proof:
        content = step.tactic
        if type(content) == str:
            output += indent*spaces + content + '\n'
            #single tactic
        else:
            output += indent*spaces + content.decl+'\n'
            output += parse_proof(content,indent=indent+1,dot=True)
            #subtheorem
    if output[:len(spaces)] == spaces and dot:
        depth = indent * len(spaces)
        output = output[:depth] + output[depth: depth + 1].replace(' ', '.') + output[depth + 1:]
    return output
            

def parseTheoremBase(thm,context=True,prompt=False):
    statement = thm.decl
    if context:
        context = thm.context
    else:
        context = ''
    proof = parse_proof(thm,dot=True)
    
    if prompt:
        return f'CONTEXT:\n {context}\n\n THEOREM: {statement} := by\n{proof}'
    else:
        return f'{context}\n\n{statement} := by\n{proof}'



def parseTheorem(thm,context=True,annotation=False,prompt=False):
    if type(thm) == AnnotatedTheorem:
        return parseAnnotatedTheorem2(thm,context,annotation,prompt)
    else:
        return parseTheoremBase(thm,context,prompt)

def run_training_data(root_path,module_name):
    os.chdir(root_path)
    cmd = f'lake exe training_data {module_name}'
    output = subprocess.run([cmd],shell=True,text=True,capture_output=True)
    data_raw = output.stdout
    data = [json.loads(item) for item in data_raw.splitlines()]
    return data_raw



def annotateTheorem(thm:Theorem, force=False) -> AnnotatedTheorem:
    src = thm.src
    path = thm.leanFile
    text = parseTheorem(thm)


    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_path = os.path.join(root_path,'.lake','packages',src,os.path.dirname(path))
    cache_path = os.path.join(root_path,'.cache',src,os.path.dirname(path))

    #print(cache_path)
    #make tempfile at package_path containing text
    #then chdir to root_path and run lake exe training_data {os.path.dirname(path).replace('/','.')+'.{file.name}'}
    temp = tempfile.NamedTemporaryFile(suffix='.lean',dir=package_path)
    with open(temp.name,'w') as f:
        f.write(text)
    
    #print(f'{src} | {path} | {os.path.dirname(path)}')
    mod_name = get_stem(os.path.dirname(path).replace('/','.') + f'.{os.path.basename(temp.name)}')
    #print(mod_name)
    output = run_training_data(root_path,mod_name)

    #json_path = get_stem(temp.name)+'.jsonl'
    json_path = os.path.join(cache_path,get_stem(os.path.basename(temp.name))+'.jsonl')
    lean_path = os.path.join(cache_path,os.path.basename(temp.name))
    with open(json_path,'w') as f:
        f.write(output)
    with open(lean_path,'w') as f:
        f.write("")
    
    path = os.path.join(get_stem(os.path.dirname(path)), os.path.basename(temp.name))
    #print(f'json_path = {json_path}\n {src}|{path}')
    file = getAnnotatedFile(src,path)
    thms = file.theorems
    os.remove(json_path)
    os.remove(lean_path)
    if len(thms)==0:
        raise NameError(f'No Theorems??\n {file}\n =|{output}|=\n\n =|{text}|=')
    output = thms[-1]
    #print([s.tactic for s in output.proof])
    output.proof = elim_overlap(output.proof)
    #print([s.tactic for s in output.proof])
    first = None
    #print('ENTERING FIRST CALC')
    for idx in range(min(len(thm.proof),len(output.proof))):
        if thm.proof[idx].tactic != output.proof[idx].tactic:
            #print(f'\n\nDIFF! {thm.proof[idx].tactic} vs {output.proof[idx].tactic}\n\n')
            first = idx-1
            break
    if first is None:
        if len(thm.proof) != len(output.proof):
            first = min(len(thm.proof),len(output.proof))-1
            #print(f'\n\nDIFF LENGTH! {thm.proof[first].tactic} vs {output.proof[first].tactic}\n\n')
    
    
    #print(f'{first}: \nthm: {thm.proof[first].tactic}\noutput: {output.proof[first].tactic}\n')
    if first is not None:
        max_pos = output.proof[first].end
        if force:
            def get_empty_annotated_proof_step(thm,i):
                proofstep=thm.proof[i]
                return AnnotatedProofStep(prevState=['ERROR'],tactic = proofstep.tactic, nextState=['ERROR'],srcUpToTactic='ERROR',declUpToTactic='ERROR',start=max_pos+i,end=max_pos+i)
            proof = [get_empty_annotated_proof_step(thm,i) if i > idx else output.proof[i] for i in range(len(thm.proof))]

            return AnnotatedTheorem(decl=output.decl,
                                    declID=output.declID,
                                    src=output.src,
                                    leanFile=output.leanFile,
                                    context=output.context,
                                    proof=proof)
        else:
            raise ValueError(f'input theorem is incorrect! \n{parseTheorem(thm,context=False)}\n{parseTheorem(output,context=False)}\nfirst={first}\n{thm.proof}\n{output.proof}')


    return output
   
    
    


if __name__ == '__main__':
    #src = 'Tests3'
    #path = 'Tests3/Basic.lean'
    #f = getAnnotatedFile(src,path)
    #thms = f.theorems
    #thm = thms[0]
    #print(thm)

    getRepo('Tests3','configs/config_test3_all.json')
    # thm = Theorem(decl='example (h : ¬ (P ∨ Q)) : ¬ P ∧ ¬ Q ', declID='Tests3.Basic.5_0.rDUmICG12jdHPcg', src='Tests3', leanFile='Tests3/Basic', context='import Mathlib.Tactic\n\nvariable (P Q R S : Prop)', proof=[ProofStep(tactic='constructor'), ProofStep(tactic='intro p'), ProofStep(tactic='have duh : P ∨ Q := by { left; exact p }'), ProofStep(tactic='exact h duh'), ProofStep(tactic='intro q fail'), ProofStep(tactic='have duh : P ∨ Q := by { right; exact q }'), ProofStep(tactic='exact h duh')])
    # print(parseTheorem(thm))
    # out = annotateTheorem(thm,force=True)
    # print(out)
    # print(parseTheorem(out,annotation=True))
    #print(parseAnnotatedTheorem2(thm,context=False,annotation=True))
