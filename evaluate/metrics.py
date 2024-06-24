import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from evaluate.build_prooftree import *

class Metric():
    def __init__(self, name, prompt, metric_fn, minmax):
        self.name = name
        self.prompt = prompt
        self.metric_fn = metric_fn
        self.minmax = minmax
    
    def metric(self,thm) -> int:
        return self.metric_fn(thm)
    
    def delta(self, old_thm, new_thm):
        old = self.metric(old_thm)
        new = self.metric(new_thm)
        return ((new-old)/old) * 100
    
    def cmp(self,*thms):
        scores = [(t,self.metric(t)) for t in thms]
        if self.minmax=='MIN':
            #smaller is better
            return min(*scores,key=lambda x:x[1])[0]
        elif self.minmax=='MAX':
            return max(*scores,key=lambda x:x[1])[0]
        else:
            return None
        
        
            




def length_metric ():
    gpt_assistant_prompt = "You are a bot that shortens Lean4 proofs while ensuring their correctness. Output in a proof tree format that aligns with the pydantic output parsing object that splits off subproofs and subtheorems.\n"
    gpt_user_prompt = "Here is a proof in Lean 4. Your goal is to rewrite the proof so that it is shorter. To help you keep track of the state of the proof, and to help think of ways to rewrite the proof, we have provided the proof states as comments.\n"
    length_prompt = gpt_assistant_prompt + "\n" + gpt_user_prompt + "\n"
    return Metric('LENGTH', length_prompt, lambda thm: len(thm.proof),'MIN')


def modularity_metric ():

    def mod_fn(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm)
        _,_,_,depth = getProofTree(thm)
        return depth
    
    gpt_assistant_prompt = "You are a bot that modifies Lean4 proofs to be more modular while maintaining their correctness. We say a proof is more modular if the proof structure is moreso made up of independent subproofs rather than a sequential list of tactics. The metric we're using measures the depth of the proof tree, which strongly favors proofs that use many independent lemmas and have statements as they have smaller proof tree depths.\n"
    gpt_user_prompt = '''Here is a proof in Lean 4. Your goal is to rewrite the proof so that it is more modular. Any lemmas or independent subproofs you wish to make, put them as a have statement proofstep within the tactic proof rather than an external lemma.
     To help you keep track of the state of the proof, and to help think of ways to rewrite the proof, we have provided the proof states as comments.\n'''
    modularity_prompt = gpt_assistant_prompt + "\n" + gpt_user_prompt + "\n"

    return Metric('MODULARITY', modularity_prompt, mod_fn,'MIN')

