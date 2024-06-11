import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from evaluate.build_prooftree import *

class Metric():
    def __init__(self, name, prompt, metric_fn):
        self.name = name
        self.prompt = prompt
        self.metric_fn = metric_fn
    
    def metric(self,thm):
        return self.metric_fun(thm)
    
    def delta(self, old_thm, new_thm):
        old = self.metric(old_thm)
        new = self.metric(new_thm)

        return ((new-old)/old) * 100


def length_metric ():
    gpt_assistant_prompt = "You are a bot that shortens Lean4 proofs while maintaining their correctness.\n"
    gpt_user_prompt = "Here is a proof in Lean 4. Your goal is to rewrite the proof so that it is shorter. To help you keep track of the state of the proof, and to help think of ways to rewrite the proof, we have provided the proof states as comments.\n"
    length_prompt = gpt_assistant_prompt + "\n" + gpt_user_prompt + "\n"
    return Metric('LENGTH', length_prompt, lambda thm: len(thm.proof))


def modularity_metric ():

    def mod_fn(thm):
        _,_,_,depth = getProofTree(thm)
        return depth
    
    gpt_assistant_prompt = "You are a bot that modifies Lean4 proofs to be more modular while maintaining their correctness. We say a proof is more modular if the proof structure is moreso made up of independent subproofs rather than a sequential list of tactics.\n"
    gpt_user_prompt = "Here is a proof in Lean 4. Your goal is to rewrite the proof so that it is more modular. To help you keep track of the state of the proof, and to help think of ways to rewrite the proof, we have provided the proof states as comments.\n"
    modularity_prompt = gpt_assistant_prompt + "\n" + gpt_user_prompt + "\n"

    return Metric('MODULARITY', modularity_prompt, mod_fn)

