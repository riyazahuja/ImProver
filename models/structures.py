from __future__ import annotations
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Union
import os
import json


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
    leanFile : str = Field(description="Lean file in which theorem is located")
    proof : List[AnnotatedProofStep] = Field(..., description="Sequence of annotated proofsteps for full proof of theorem.")

class Theorem(BaseModel):
    decl : str = Field(description="Theorem declaration")
    declID : str = Field(description="Unique theorem declaration ID")
    leanFile : str = Field(description="Lean file in which theorem is located")
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


def getTheorems(data,file) -> List[AnnotatedTheorem]:
    temp = {}
    for step in data:
        ps = AnnotatedProofStep(prevState=step['prevState'],
                                        tactic = step['tactic'],
                                        nextState=step['nextState'],
                                        srcUpToTactic=step['srcUpToTactic'],
                                        declUpToTactic=step['declUpToTactic'])
        decl = step['decl']
        declID = step['declId']
        if declID not in temp.keys():
            temp[declID] = {'proof':[ps], 'decl':decl}
            print(temp)
        else:
            print(temp)
            curr_proof = temp[declID]['proof']
            curr_decl = temp[declID]['decl']
            curr_proof.append(ps)
            temp[declID] = {'proof':curr_proof, 'decl':curr_decl}
            
    result = {}
    for ID,value in temp.items():
        result[ID] = AnnotatedTheorem(leanFile=file,decl=value['decl'],declID=ID,proof=value['proof'])
    return [v for _,v in result.items()]
        

def getAnnotatedFile(src, file_name):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    content_path = os.path.join(root_path, f'.cache/{src}/StateComments')
    data_path = os.path.join(root_path, f'.cache/{src}/TacticPrediction')
    
    contents = None
    data = None
    with open(os.path.join(content_path,file_name+'.lean'),'r') as f:
        contents = f.read()


    with open(os.path.join(data_path,file_name+'.jsonl'),'r') as f:
        data = [json.loads(jline) for jline in f.read().splitlines()]
    
    theorems = getTheorems(data,file_name)
    print(theorems)

    return AnnotatedFile(src=src,file_name=file_name,contents = contents,theorems=theorems)
    
    
    
    


    

def parseTheorem(thm : Theorem):
    statement = thm.statement
    psteps = [t.content for t in thm.proof]
    proof = ''
    for i in psteps:
        proof = proof + '  ' + i + '\n'
    return f'{statement} := by\n{proof}'
