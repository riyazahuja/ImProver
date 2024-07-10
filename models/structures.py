from __future__ import annotations
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from typing import List, Union,Optional,Tuple
import json
import tempfile
import subprocess 
from textwrap import indent
import re

#NOTE position encodings have top left as line 1, column 0 
# -> we restandardize to line 1 column 1. makes more sense that way

class ProofStep(BaseModel):
    tactic : Union[str,Theorem] = Field(description="One line/tactic in a tactic proof (str) or a subtheorem/sublemma/subproof")

class Theorem(BaseModel):
    decl : Optional[str] = Field(description="Theorem declaration. Optional argument, if not provided, will default to being an implicit case statement using dot notation.",default=None)
    proof : List[ProofStep] = Field(..., description="Sequence of proofsteps for full proof of theorem.")
    declID : str = Field(description="Unique theorem declaration ID")
    src : str = Field(description="Source repo of the theorem")
    leanFile : str = Field(description="Lean file in which theorem is located")
    context : str = Field(description="Context of the theorem (i.e. file contents up to decl)")
    project_path : str = Field(description="Local path to src repo contents")
    
class File(BaseModel):
    src : str = Field(description="File source repo")
    file_name : str = Field(description="File Name")
    file_path : str = Field(description="File path (stem) relative to src repo root")
    file_type : str = Field(description="File type")
    contents : str = Field(description= "File contents")
    project_path : str = Field(description="Local path to src repo contents")

class AnnotatedProofStep(BaseModel):
    prevState : List[str] = Field(description="Pretty printed tactic st ate before the tactic invocation")
    tactic : str = Field(description="One line/tactic in a tactic proof.")
    nextState : List[str] = Field(description="Pretty printed tactic state after the tactic invocation")
    srcUpToTactic : str = Field(description="Source code from file start to current tactic")
    declUpToTactic : str = Field(description="Source code from theorem declaration to current tactic")
    start: Tuple[Optional[int],Optional[int]]=Field(description='start coordinates from source file as (row,column)')
    end: Tuple[Optional[int],Optional[int]]=Field(description='end coordinates from source file as (row,column)')
    #start : int = Field(description='start UTF-8 byte position of tactic invocation')
    #end : int = Field(description='end UTF-8 byte position of tactic invocation')

class Message(BaseModel):
    severity:str = Field(description='Message severity')
    start: Tuple[Optional[int],Optional[int]]=Field(description='start coordinates from source file as (row,column)')
    end: Tuple[Optional[int],Optional[int]]=Field(description='end coordinates from source file as (row,column)')
    message_src: Optional[str] = Field(description='equivalent to source_contents[start:end]')
    content: str = Field(description='Message contents')

class AnnotatedTheorem(BaseModel):
    decl : str = Field(description="Theorem declaration")
    declID : str = Field(description="Unique theorem declaration ID")
    src : str = Field(description="Source repo of the theorem")
    leanFile : str = Field(description="Lean file in which theorem is located")
    context : str = Field(description="Context of the theorem (i.e. file contents up to decl)")
    proof : List[AnnotatedProofStep] = Field(..., description="Sequence of annotated proofsteps for full proof of theorem.")
    project_path : str = Field(description="Local path to src repo contents")
    messages : List[Message] = Field(...,description="Messages from the lean server")

class AnnotatedFile(BaseModel):
    src : str = Field(description="File source repo")
    file_name : str = Field(description="File Name")
    file_path : str = Field(description="File path (stem) relative to src repo root")
    file_type : str = Field(description="File type")
    contents : str = Field(description= "File contents")
    theorems : List[AnnotatedTheorem] = Field(..., description="List of all theorems in a file")
    project_path : str = Field(description="Local path to src repo contents")

class Repo(BaseModel):
    type: str = Field(description="Repository type (git or local)")
    url: Optional[str] = Field(description= "if git repo, github url")#if remote
    commit : Optional[str] = Field(description= "if git repo, github commit SHA code") #if remote
    path : Optional[str] = Field(description= "if local repo, path")#if local
    version : str = Field(description= "Lean version")
    name: str = Field(description= "Repo name")
    dependencies: List[Repo] = Field(description= "Repository dependencies")
    files: List[Union[AnnotatedFile,File]]= Field(description= "files in repository")
    project_path : str = Field(description="Local path to src repo contents")

def getMessages(start,end,msgs,contents:str):
    thm_start_row = start.get('line',None)
    thm_start_col = start.get('col',None)
    if end is not None:
        thm_end_row = end.get('line',None)
        thm_end_col = end.get('col',None)
    else:
        thm_end_row=0
        thm_end_col=0

    def msg_in_thm(msg):
        start_row = msg.get('pos',{}).get('line',None) if msg.get('pos',None) is not None else None
        end_row = msg.get('endPos',{}).get('line',None) if msg.get('endPos',None) is not None else None

        if start_row is None:
            return False
        if thm_start_row <= start_row and end is not None and thm_end_row>= end_row:
            #print(f'\nCASE 1 <================= {msg}\n')
            return True
        elif thm_start_row <= start_row and end is None:
            #print(f'\nCASE 2 <================= {msg}\n')
            return True
        else:
            return False

    
    msgs2=[msg for msg in msgs if msg_in_thm(msg)]
    #s = "\n".join(map(lambda x: f'{x[0]+1} | {x[1]}',list(enumerate(contents.splitlines()))))
    #print(f'+++++++++++++++\nthm_start_row={thm_start_row}\nMSG CONTENTS:\n{s}\nMSGS:\n{msgs}\nMSGS2:\n{msgs2}\n+++++++++++++++\n')
    msgs=msgs2
    return [getMessage(msg,contents) for msg in msgs]

def getMessage(msg,contents:str):
    severity = msg['severity']

    start_row = msg.get('pos',{}).get('line',None) if msg.get('pos',None) is not None else None
    start_col = msg.get('pos',{}).get('column',None) if msg.get('pos',None) is not None else None
    end_row = msg.get('endPos',{}).get('line',None) if msg.get('endPos',None) is not None else None
    end_col = msg.get('endPos',{}).get('column',None) if msg.get('endPos',None) is not None else None

    

    message_src = None
    if start_col is None:
        start_col=1
        #message_src = ''
    if end_col is None and end_row is not None:
        end_col=len(end_row)
        #message_src = ''

    if end_row is None:
        #s = "\n".join(map(lambda x: f'{x[0]} | {x[1]}',list(enumerate(contents.splitlines()))))
        #print(f'~~~~~~~~~~~~~~~\nidx : {start_row-1}\n{s}\n~~~~~~~~~~~~~~~')
        try:
            lines = [contents.splitlines()[start_row-1]]
        except:
            lines = []

    else:
        lines = contents.splitlines()[start_row-1:end_row]
    trimmed_lines = []
    #s = "\n".join(map(lambda x: f'{x[0]+1} | {x[1]}',list(enumerate(contents.splitlines()))))
    #if severity=='error':
        #print(f'******************\nMSG: {msg}\n\n{s}\n******************')

    for line in lines:
        if line == lines[0]:
            if len(lines) == 1:
                trimmed_lines.append(line)
            else:
                trimmed_lines.append(line[start_col:])
        elif line == lines[-1]:
            if end_col is None:
                trimmed_lines.append(line)
            else:
                trimmed_lines.append(line[:end_col+1])
            
        else:
            trimmed_lines.append(line)
    content = msg['data']

    #if message_src is not None:
    message_src = '\n'.join(trimmed_lines)

    return Message(severity=severity,start=(start_row,start_col),end=(end_row,end_col),message_src=message_src, content=content)



def getTheorems(data, src, path, project_path,contents,until_end=False) -> List[AnnotatedTheorem]:
    temp = {}
    #print(data)
    msgs = data['messages']
    
    data = data['tactics']
    
    #messages = [getMessage(msg,contents) for msg in msgs]

    for step in data:
        ps = AnnotatedProofStep(prevState=step['prevState'],
                                        tactic = step['tactic'],
                                        nextState=step['nextState'],
                                        srcUpToTactic=step['srcUpToTactic'],
                                        declUpToTactic=step['declUpToTactic'],
                                        start = (step['startPos'].get('line',None),step['startPos'].get('column',None)),
                                        end= (step['endPos'].get('line',None),step['endPos'].get('column',None))
        )
        def elim_by(text):
            return re.sub(r'\s*:=\s*by\s*$','',text,flags=re.M)

        decl = elim_by(step['decl'])
        declID = step['declId']
        thm_start,thm_end = step['thm_startPos'],step['thm_endPos']
        if until_end:
            thm_end=None
        #print(f'\n###########\n{decl} : start -> {thm_start}\n################\n')
        messages = getMessages(thm_start,thm_end,msgs,contents)
        messages.reverse()
        lines_src = step['srcUpToTactic'].split('\n')
        decl_lines = decl.split('\n')
        #print(lines_src)
        
        #lines = [line for line in lines_src if not line in decl_lines]
        maybe_context = '\n'.join(lines_src[:-len(decl_lines)-1]).strip()

        if declID not in temp.keys():
            temp[declID] = {'proof':[ps], 'decl':decl,'context' : maybe_context,'messages':messages}
            #print(temp)
        else:
            #print(temp)
            curr_proof = temp[declID]['proof']
            curr_decl = temp[declID]['decl']
            curr_ctxt = temp[declID]['context']
            curr_msgs = temp[declID]['messages']
            curr_proof.append(ps)
            temp[declID] = {'proof':curr_proof, 'decl':curr_decl, 'context':curr_ctxt,'messages':curr_msgs}
            
    result = {}
    for ID,value in temp.items():
        result[ID] = AnnotatedTheorem(leanFile=path,src=src,decl=value['decl'],declID=ID,proof=value['proof'],context = value['context'],project_path=project_path,messages=value['messages'])
    return [v for _,v in result.items()]
        

def get_stem(path):
    #print(f'{path} : {path[-5:]} : {path[:-5]}')
    if path[-5:]=='.lean':
        return path[:-5]
    elif path[-5:]=='.json':
        return path[:-5]
    return path



def getAnnotatedFile(src, path, project_path,until_end=False):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root_path, f'.cache/{src}')

    #content_path = os.path.join(root_path,'.lake','packages',src)

    #print(f'{project_path}|{path}')
    with open(os.path.join(project_path,path),'r') as f:
        contents = f.read()
    

    stem,ftype = os.path.splitext(path)
    with open(os.path.join(cache_path,stem+'.json'),'r') as f:
        data = json.load(f)
    
    theorems = getTheorems(data, src, path,project_path,contents,until_end=until_end)

    return AnnotatedFile(src=src,file_name=os.path.basename(path),contents = contents,theorems=theorems,file_path=path,file_type=ftype,project_path=project_path)
    
#if annotate + force, run annotateFile and if error quit
#if annotate run annotateFile and if error, return File
#if not annotate, return File
def getFile(src,path,project_path, annotate=True,force=False):
    #print(f'{src} | {path}')
    if annotate:
        try:
            return getAnnotatedFile(src,path,project_path)
        except FileNotFoundError as e:
            if force:
                raise e
            else:
                #print(f'ERROR: \n{e}\n\n')
                pass
    stem,ftype = os.path.splitext(path)
    with open(os.path.join(project_path,path),'r') as f:
        content = f.read()
    #print(f'{os.path.basename(path)} | {stem} | {ftype}')

    return File(src=src,file_name=os.path.basename(path),file_path=path,file_type=ftype,contents=content,project_path=project_path)
# file name is all.lean
# file path is Tests3/all.lean
# file type is .lean
# project path is path to Tests3
# src is Tests3



#TODO IMPLEMENT LOCAL CONFIGS TO ADD LOCAL CONFIGS TO GETREPO
def getRepoDirect(repo_data,annotate=True,force=False,recursive=True):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_path = ''
    if 'path' in repo_data.keys():
        local = True
    elif 'repo' in repo_data.keys():
        local = False
    else:
        raise ValueError(f'Invalid Config:\n {repo_data}')
    

    version = repo_data.get('lean','')
    name = repo_data.get('name','')

    if local:
        path = repo_data.get('path','')
        url = None
        commit = None
        project_path=path
    else:
        url = repo_data.get('repo','')
        commit = repo_data.get('commit','')
        path = None
        project_path=os.path.join(root_path,'.lake','packages',name)

    
    #depedencies:
    if recursive:
        manifest_path = os.path.join(project_path,'lake-manifest.json')
        toolchain_path = os.path.join(project_path,'lean-toolchain')
        
        with open(manifest_path,'r') as f:
            manifest_data = json.load(f).get('packages',[])
        with open(toolchain_path,'r') as f:
            pack_version = f.read()
        dependencies = []
        for package in manifest_data:
            dependency_names = [item.name for item in dependencies]
            if package['name'] in dependency_names:
                continue
            if package['type'] == 'git':
                subdata = {
                    'repo': package['url'],
                    'commit': package['rev'],
                    "lean": pack_version,
                    "name": package['name']
                }
            else:
                subdata = {
                    'path': package['path'],
                    "lean": pack_version,
                    "name": package['name']
                }
            dependencies.append(getRepoDirect(subdata,annotate=False,force=False,recursive=False))
    else:
        dependencies = []
    #Files:
    repo_files = []
    ignore = ['.lake']

    for root, dirs ,files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in ignore]
        for file in files:
            fp = os.path.join(root,file)
            if fp.endswith('.lean') and name not in ignore:
                repo_files.append(getFile(name,os.path.relpath(fp,project_path),project_path,annotate=annotate,force=force))
    if local:
        return Repo(type='local',path=path,version=version,name=name,dependencies=dependencies,files=repo_files,project_path=project_path)
    else:
        return Repo(type='git',url=url,commit=commit,version=version,name=name,dependencies=dependencies,files=repo_files,project_path=project_path)
    

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

    return getRepoDirect(repo_data,annotate=annotate,force=force)



def parseAnnotatedTheorem(thm,context=True,annotation=False):
    last_pstep = thm.proof[-1]
    if context:
        src = last_pstep.srcUpToTactic+last_pstep.tactic
    else:
        src = last_pstep.declUpToTactic+last_pstep.tactic
    return src

def elim_overlap(pf: List[AnnotatedProofStep]):
    def pos_le(a,b):
        a_line,a_col = a
        b_line,b_col = b

        if a_line < b_line:
            return True
        elif a_line == b_line and a_col <= b_col:
            return True
        else:
            return False
    def pos_max(a,b):
        le = pos_le(a,b)
        if le:
            return b
        else:
            return a


    ptr = (0,0)
    output = []
    for step in pf:
        start = step.start
        end = step.end
        if pos_le(start, ptr) and pos_le(end, ptr):
            #this is inside a have
            pass
        else:
            ptr = pos_max(start,end)
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
            # pattern = re.compile(r'(case\s+\w+\b)(?!\s*=>)')
            #pattern = re.compile(r'^\s*case\s*\w+(?:\s*:\s*[^=\n]+)?\s*(?!=>)')
            # case=False
            # if re.match(pattern,content.decl):
            #     case = True
            # arrow = False
            # if '=>' in content.decl:
            #     arrow = True

            #if not case:
            hasDecl = content.decl is not None
            if hasDecl:
                output += indent*spaces + content.decl +'\n'#f"{' => ' if case and not arrow else ''}" +'\n'
            output += parse_proof(content,indent=indent+1,dot=(not hasDecl))
            #subtheorem
    
    depth = indent * len(spaces)
    #print (f'PARSING PROOF: req {depth}, first [{output[:depth]}], dot? {dot}')
    if output[:depth] == spaces*indent and dot:
        #print('HELOO!!!!')
        output = output[:depth-2] + '. '+ output[depth :]#output[depth: depth + 1].replace(' ', '.') + output[depth + 1:]
    return output
            

def parseTheoremBase(thm,context=True,prompt=False):
    statement = thm.decl
    if context:
        context = thm.context
    else:
        context = ''
    proof = parse_proof(thm,dot=False)
    
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
    if data_raw == '':
        raise KeyError(f'BAD DATA: {output}')
    data = json.loads(data_raw)
    return data



def annotateTheorem(thm:Theorem, force=False) -> AnnotatedTheorem:
    '''
    okurr heres the game plan yall
    we gonna take our theorem, and its full path
    (i.e. os.path.join(thm.project_path,thm.leanFile))
    and then move up one directory. Then we make a temporary lean file
    in that directory, and insert in parseTheorem(thm) into it.
    we then get the module name of this by taking the relative path and replacing with dots.
    Then we run training_data on this and then extract the outputted json and put into the cache
    We then get this AnnotatedFile and from it, get the last theorem from its theorems.
    Now we look at the messages. 
    If there is an error, if force is false, raise error
    else, we know that up until the line of the error, we good, 
    so we navigate to the tactic that has that range in its range, and then copy over
    tactics from thm for the rest of the proof.
    then return.
    '''


    src = thm.src
    path = thm.leanFile
    text = parseTheorem(thm)


    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #project_path = os.path.join(root_path,'.lake','packages',src)
    project_path = thm.project_path
    

    cache_path = os.path.join(root_path,'.cache',src)

    path_dir = os.path.join(project_path,os.path.dirname(path))

    temp = tempfile.NamedTemporaryFile(suffix='.lean',dir=path_dir)
    with open(temp.name,'w') as f:
        f.write(text)
    temp_relpath = os.path.relpath(temp.name,project_path)
    
    mod_name = get_stem(temp_relpath.replace('/','.'))
    output = run_training_data(root_path,mod_name)
    #print(f'{output} : {type(output)}')


    #json_path = os.path.join(cache_path,get_stem(temp_relpath)+'.json')
    #with open(json_path,'w') as f:
    #    json.dump(output,f)#f.write(output)

    
    #file = getAnnotatedFile(src,temp_relpath,project_path,until_end=True)
    thms = getTheorems(output,src,temp_relpath,project_path,text,until_end=True)
    
    #thms = file.theorems
    #os.remove(json_path)

    if len(thms)==0:
        raise NameError(f'No Theorems??\n =|{output}|=\n\n =|{text}|=\n\n=|{thm}|=\n\n=|{mod_name}|=')
    
    output = thms[-1]
    
    #print([s.tactic for s in output.proof])
    output.proof = elim_overlap(output.proof)
    #print([s.tactic for s in output.proof])
    first = None
    #print(f'\n----------\n{list(enumerate([thm.decl for thm in thms]))}\ntext:\n{text}\nmsgs:\n{output.messages}\n\nog:\n{parseTheorem(thm,context=False)}\nnew:\n{parseTheorem(output,context=False)}\n----------\n')
    
    def flattenProof (proof):
        new_proof=[]
        for stepraw in proof:
            step = stepraw.tactic
            if type(step) == str:
                new_proof.append(stepraw)
            else:
                decl = step.decl
                # if decl is not None:
                #     text = decl + '\n' + parse_proof(step)
                #     #new_proof.append(ProofStep(tactic=decl))
                #     new_proof.append(ProofStep(tactic=text))
                # else:
                #     new_proof.extend(flattenProof(step.proof))
                new_proof.append(ProofStep(tactic=decl))
                new_proof.extend(flattenProof(step.proof))
        return new_proof



    
    og_proof = flattenProof(thm.proof)


    #print('ENTERING FIRST CALC')
    for idx in range(min(len(og_proof),len(output.proof))):
        if og_proof[idx].tactic != output.proof[idx].tactic:
            first = idx
            break

    if first is None:
        if len(og_proof) != len(output.proof):
            first = min(len(og_proof),len(output.proof))
    
    
    if first is not None:
        if first != 0:
            max_pos = output.proof[first-1].end
        else:
            max_pos = (1,1)

        if force:
            def get_empty_annotated_proof_step(i):
                proofstep=og_proof[i]
                return AnnotatedProofStep(prevState=['ERROR'],tactic = proofstep.tactic, nextState=['ERROR'],srcUpToTactic='ERROR',declUpToTactic='ERROR',start=(max_pos[0]+i,max_pos[1]+i),end=(max_pos[0]+i,max_pos[1]+i))
            proof = [get_empty_annotated_proof_step(i) if i >= first else output.proof[i] for i in range(len(og_proof))]
            
            final = AnnotatedTheorem(decl=output.decl,
                                    declID=output.declID,
                                    src=output.src,
                                    leanFile=output.leanFile,
                                    context=output.context,
                                    proof=proof,
                                    project_path=project_path,
                                    messages=output.messages)

            if [s.tactic for s in og_proof] != [s.tactic for s in proof]:
                raise ValueError(f'=============Forcing Failed:\n{parseTheorem(thm,context=False)}\n{[s.tactic for s in og_proof]}\n--------\n{parseTheorem(final,context=False)}\n{[s.tactic for s in proof]}\n+++++++++++++++++++\n{[s.tactic for s in output.proof]}\n{output.messages}\nfirst: {first}\n=============')

            return final
        else:
            raise ValueError(f'input theorem is incorrect! \n{parseTheorem(thm,context=False)}\n{parseTheorem(output,context=False)}\nfirst={first}\n{og_proof}\n{output.proof}')

   
    return output
   
    
    


if __name__ == '__main__':


    # root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # r = getRepo('Tests','configs/config_test.json')
    # #f = getAnnotatedFile('Tests','Tests/Basic.lean','/Users/ahuja/Desktop/LeanTestData/Tests')
    # files = {file.file_name:file for file in r.files}
    # #print(files.keys())

    # #f = files['Solutions_S02_Functions.lean']
    # f = files['Basic.lean']
    # thms = f.theorems
    # for thm in thms:
    #   #print(parseTheorem(thm,context=False))  
    #   print(thm)
    #   print('')
    #   print(thm.messages)
    #   #print(thm)
    ProofStep.update_forward_refs()
    thm = Theorem(decl='theorem anddd (P Q : Prop) : P ∧ Q → Q ∧ P', proof=[ProofStep(tactic='intro hpq'), ProofStep(tactic='rcases hpq with ⟨hp, hq⟩'), ProofStep(tactic='constructor'), ProofStep(tactic='case intro.right => exact hp'), ProofStep(tactic='case intro.left => exact hq')], declID='Tests.Basic.1_0.pB8qUlrJF0ql2T1', src='Tests', leanFile='Tests/Basic.lean', context='', project_path='/Users/ahuja/Desktop/LeanTestData/Tests')
    print(f'OG:\n{parseTheorem(thm,context=False)}')
    print('')
    anno = annotateTheorem(thm)
    print(f'NEW:\n{parseTheorem(anno,context=False)}')
    