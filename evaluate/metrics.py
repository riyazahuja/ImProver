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
    sys_prompt = ('system','''You are an AI assistant who shortens Lean 4 proofs while ensuring their correctness. 
                  You will aim to reduce the number of lines of the tactic proof while ensuring that it properly compiles in lean 4.''')
    
    user_prompt = ('human','''Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short in length - measured in 
                   the number of lines of the proof - as possible, while also ensuring that the output is still syntactically correct.''')
    return Metric('LENGTH', [sys_prompt,user_prompt], lambda thm: len(thm.proof),'MIN')


def modularity_metric ():

    def mod_fn(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm,force=True)
        _,_,_,depth = getProofTree(thm)
        return depth
    
    sys_prompt = ('system','''You are an AI assistant who rewrites Lean 4 proofs to be more modular while ensuring their correctness. 
                  We say a proof is more modular if the proof structure is moreso made up of independent subproofs 
                  rather than a sequential list of tactics. The metric we\'re using measures the depth of the proof tree, which strongly 
                  favors proofs that use many independent subproofs as the depth of the proof tree is the maximum of the depths of these branches.''')
    
    user_prompt = ('human','''Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is more modular. Any lemmas or 
                   independent subproofs you wish to make, put them as a \"have\" statement proofstep within the tactic proof rather than an external lemma.''')
    

    return Metric('MODULARITY', [sys_prompt,user_prompt], mod_fn,'MIN')



def formalization_metric ():
    
  def num_errors(thm):
      if type(thm) == Theorem:
          thm = annotateTheorem(thm)
      errors = sum(1 for msg in thm.messages if msg.severity=='error')
      return errors

  sys_prompt = ('system','You are an AI assistant who automatically formalizes LaTeX/lean4 proofs into Lean 4 proofs and ensures their correctness. You will either recieve a human-readable LaTeX proof (denoted with a <INFORMAL PROOF> header in the context), upon which you should aim to construct a formal lean 4 proof that compiles as correct. Namely, the context, decl, and proof will all be in latex, and will each need to be converted into lean4 proofs, theorem declarations, and tactic proofs respectively. Or you may recieve a lean 4 proof you must modify to eliminate any errors so that it compiles as correct.')
  user_prompt = ('human','Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is a formal Lean 4 proof (if it is not already), and moreover, a correct formal proof. If the context has the heading <INFORMAL PROOF> then be sure to convert the decl and proof to both be valid lean 4 as they are both currently latex.')
  

  return Metric('FORMALIZATION', [sys_prompt,user_prompt], num_errors, 'MIN')



def completion_metric ():
  def num_errors(thm):
    if type(thm) == Theorem:
        thm = annotateTheorem(thm)
    errors = sum(1 for msg in thm.messages if msg.severity=='error')
    return errors

  sys_prompt = ('system','You are an AI assistant who automatically solves Lean 4 proofs (as in, generates the tactic proof) and ensures its correctness. You will recieve a Lean 4 proof you must modify to eliminate any errors so that it compiles as correct and, and elimanate any \"sorry\"s with full proofs.')
  user_prompt = ('human','Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is a formal, complete, and correct Lean 4 proof by filling in its tactic proof.')

  return Metric('COMPLETION', [sys_prompt,user_prompt], num_errors, 'MIN')
  




