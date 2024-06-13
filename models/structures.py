from __future__ import annotations
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Union
import os
import json
import re

class ProofStep(BaseModel):
    tactic : str = Field(description="One line/tactic in a tactic proof.")

class AnnotatedProofStep(BaseModel):
    prevState : List[str] = Field(description="Pretty printed tactic st ate before the tactic invocation")
    tactic : str = Field(description="One line/tactic in a tactic proof.")
    nextState : List[str] = Field(description="Pretty printed tactic state after the tactic invocation")
    srcUpToTactic : str = Field(description="Source code from file start to current tactic")
    declUpToTactic : str = Field(description="Source code from theorem declaration to current tactic")

class AnnotatedTheorem(BaseModel):
    decl : str = Field(description="Theorem declaration")
    declID : str = Field(description="Unique theorem declaration ID")
    src : str = Field(description="Source repo of the theorem")
    leanFile : str = Field(description="Lean file in which theorem is located")
    context : str = Field(description="Context of the theorem (i.e. file contents up to decl)")
    proof : List[AnnotatedProofStep] = Field(..., description="Sequence of annotated proofsteps for full proof of theorem.")

class Theorem(BaseModel):
    decl : str = Field(description="Theorem declaration")
    declID : str = Field(description="Unique theorem declaration ID")
    src : str = Field(description="Source repo of the theorem")
    leanFile : str = Field(description="Lean file in which theorem is located")
    context : str = Field(description="Context of the theorem (i.e. file contents up to decl)")
    proof : List[ProofStep] = Field(..., description="Sequence of proofsteps for full proof of theorem.")

class File(BaseModel):
    src : str = Field(description="File source repo")
    file_name : str = Field(description="File Name")
    contents : str = Field(description= "File contents")

class AnnotatedFile(BaseModel):
    src : str = Field(description="File source repo")
    file_name : str = Field(description="File Name")
    contents : str = Field(description= "File contents")
    theorems : List[AnnotatedTheorem] = Field(..., description="List of all theorems in a file")


def getTheorems(data,src, file) -> List[AnnotatedTheorem]:
    temp = {}
    for step in data:
        ps = AnnotatedProofStep(prevState=step['prevState'],
                                        tactic = step['tactic'],
                                        nextState=step['nextState'],
                                        srcUpToTactic=step['srcUpToTactic'],
                                        declUpToTactic=step['declUpToTactic'])
        decl = step['decl']
        declID = step['declId']

        lines_src = step['srcUpToTactic'].split('\n')
        lines = [line for line in lines_src if decl not in line]
        maybe_context = '\n'.join(lines).strip()

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
    return path

def getAnnotatedFile(src, file_name):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    content_path = os.path.join(root_path, f'.cache/{src}')

    
    contents = None
    data = None

    file_name = get_stem(file_name)

    with open(os.path.join(content_path,file_name+'.lean'),'r') as f:
        contents = f.read()


    with open(os.path.join(content_path,file_name+'.jsonl'),'r') as f:
        data = [json.loads(jline) for jline in f.read().splitlines()]
    
    theorems = getTheorems(data,src, file_name)
    #print(theorems)

    return AnnotatedFile(src=src,file_name=file_name,contents = contents,theorems=theorems)
    

def parseAnnotatedTheorem(thm):
    last_pstep = thm.proof[-1]
    src = last_pstep.srcUpToTactic+last_pstep.tactic
    return src


def parseTheorem(thm):
    statement = thm.decl
    context = thm.context
    psteps = [t.tactic for t in thm.proof]
    proof = ''
    for i in psteps:
        proof = proof + '  ' + i + '\n'
    return f'{context}\n\n{statement} := by\n{proof}'
