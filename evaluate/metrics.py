import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from models.rag import *
from evaluate.build_prooftree import *
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import json
class Metric():
    def __init__(self, name, prompt, metric_fn, examples, minmax):
        self.name = name
        self.prompt = prompt
        self.metric_fn = metric_fn
        self.examples=examples
        self.minmax = minmax
        self.vs = self.get_example_selector()
    
    def metric(self,thm) -> int:
        return self.metric_fn(thm)
    
    def delta(self, old_thm, new_thm):
        old = self.metric(old_thm)
        new = self.metric(new_thm)
        return ((new-old)/old) * 100
    
    def get_example_selector(self):
        vs = get_metric_vs(self.examples,self.name)
        return vs
        # ex = [
        #     {
        #         "input": example['input'].replace(r'{',r'{{').replace(r'}',r'}}'),
        #         "output": example['output'].replace(r'{',r'{{').replace(r'}',r'}}')
        #     }
        #     for example in self.examples
        # ]

        # example_selector = SemanticSimilarityExampleSelector.from_examples(
        #     ex,
        #     OpenAIEmbeddings(),
        #     Chroma,
        #     k=len(self.examples),
        # )
        # return example_selector

    
    
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

    def len_fn(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm,force=True)
        num_lines = len(thm.proof)
        #dont count semicolons
        semicolons = 0
        for line in thm.proof:
            #ignore <;>'s and ;'s at end of line
            content = line.tactic.replace('<;>','')[:-1]
            semicolons += content.count(';')
        return num_lines+semicolons


    sys_prompt = ('system','''You are an AI assistant who shortens Lean 4 proofs while ensuring their correctness. 
                  You will aim to reduce the number of lines of the tactic proof while ensuring that it properly compiles in lean 4.''')
    
    user_prompt = ('human','''Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short in length - measured in 
                   the number of lines of the proof - as possible, while also ensuring that the output is still syntactically correct.''')
    

    examples = [
        {
            "input": '''theorem add_zero (n : ℕ) : n + 0 = n := by
  induction n with
  | zero =>
    rw [Nat.add_eq_right]
  | succ n ih =>
    rw [Nat.add_assoc, Nat.add_comm 1 0, ← Nat.add_assoc,ih]''',
            "output":'''theorem add_zero (n : ℕ) : n + 0 = n := by
  exact rfl
'''
        },
        {
            "input": '''theorem imp_trans {P Q R : Prop} (h1 : P → Q) (h2 : Q → R) : P → R := by
  intro hp
  apply h2
  apply h1
  exact hp''',
  "output":'''theorem imp_trans {P Q R : Prop} (h1 : P → Q) (h2 : Q → R) : P → R := by
  exact (fun p => h2 (h1 p))'''
        },
        {
            "input":'''theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
  ext x
  constructor
  . intro h
    rcases h with ⟨hs,_⟩
    exact hs
  . intro h
    constructor
    . exact h
    . exact h''',
    "output":'''theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
  exact Set.ext fun x => { mp := fun h => And.casesOn h fun hs _ => hs, mpr := fun h => ⟨h, h⟩ }'''
        },
        {
            "input":'''theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
  ext x
  unfold Function.comp
  exact rfl''',
  "output":'''theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
  exact (funext fun x => id rfl)'''
        }
    ]


    return Metric('LENGTH', [sys_prompt,user_prompt], lambda thm: len_fn,examples,'MIN')


# def length_metric ():
#     def len_fn(thm):
#         if type(thm) == Theorem:
#             thm = annotateTheorem(thm,force=True)
#         num_lines = len(thm.proof)
#         #dont count semicolons
#         semicolons = 0
#         for line in thm.proof:
#             #ignore <;>'s and ;'s at end of line
#             content = line.tactic.replace('<;>','')[:-1]
#             semicolons += content.count(';')
#         return num_lines+semicolons

#     sys_prompt = ('system','''You are an AI assistant who shortens Lean 4 proofs while ensuring their correctness. 
#                   You will aim to reduce the number of tactic invocations in the tactic proof - as in, decrease the number of lines in the proof, without simply moving everything to one line using semicolons - while ensuring that it properly compiles in lean 4. ''')
    
#     user_prompt = ('human','''Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short in length - measured in 
#                    the number of tactic invocations in the proof (lines + number of semicolons-1) - as possible, while also ensuring that the output is still syntactically correct.''')
#     return Metric('LENGTH', [sys_prompt,user_prompt], len_fn,'MIN')


def modularity_metric ():

    def mod_fn(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm,force=True)
        G,_,_ = getProofTree(thm)
        return depth(G)
    
    sys_prompt = ('system','''You are an AI assistant who rewrites Lean 4 proofs to be more modular while ensuring their correctness. 
                  We say a proof is more modular if the proof structure is moreso made up of independent subproofs 
                  rather than a sequential list of tactics. The metric we\'re using measures the depth of the proof tree, which strongly 
                  favors proofs that use many independent subproofs as the depth of the proof tree is the maximum of the depths of these branches.''')
    
    user_prompt = ('human','''Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is more modular. Any lemmas or 
                   independent subproofs you wish to make, put them as a \"have\" statement proofstep within the tactic proof rather than an external lemma.''')
    
    examples = []

    return Metric('MODULARITY', [sys_prompt,user_prompt], mod_fn,examples,'MIN')

def similarity_metric ():

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
    



    examples = [
        {
            'input':'''theorem add_zero (a : Nat) : a + 0 = a := by
  --induction
  induction a with
  | zero => rfl
  | succ a ih => rw [Nat.succ_add, ih]''',
            'output':'''theorem add_zero (a : Nat) : a + 0 = a := by
  --rewriting
  rw [Nat.add_comm, Nat.zero_add]'''
        },
        {
            'input':'''theorem double_negation (P : Prop) : P → ¬¬P := by
  --using contradiction
  intro h h1
  contradiction''',
            'output':'''theorem double_negation (P : Prop) : P → ¬¬P := by
  --directly
  intro h h1
  exact h1 h'''
        },
        {
            'input':'''theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
  --bash cases
  intro h
  by_cases hp : P
  . constructor
    . exact hp
    . intro hq
      exact h (fun _ => hq)
  . exact Classical.not_imp.mp h''',
            'output':'''theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
  --push negations inwards
  intro h
  push_neg at h
  exact h'''
        },
        {
            'input':'''theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  -- direct
  intro h nq p
  exact nq (h p)''',
            'output':'''theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  -- logical equivalences
  intro h nq
  have nhq := not_or_of_imp h
  rcases nhq with hnp | hq
  . exact hnp
  . exfalso
    contradiction'''
        }
    ]


    return Metric('SIMILARITY', [sys_prompt,user_prompt], mod_fn,examples,'MIN')

def readability_metric ():
    def readability_fn (thm):
        return 0
    
    sys_prompt = ('system','You are an AI assistant who automatically formalizes LaTeX/lean4 proofs into Lean 4 proofs and ensures their correctness. You will either recieve a human-readable LaTeX proof (denoted with a <INFORMAL PROOF> header in the context), upon which you should aim to construct a formal lean 4 proof that compiles as correct. Namely, the context, decl, and proof will all be in latex, and will each need to be converted into lean4 proofs, theorem declarations, and tactic proofs respectively. Or you may recieve a lean 4 proof you must modify to eliminate any errors so that it compiles as correct.')
    user_prompt = ('human','Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is a formal Lean 4 proof (if it is not already), and moreover, a correct formal proof. If the context has the heading <INFORMAL PROOF> then be sure to convert the decl and proof to both be valid lean 4 as they are both currently latex.')
    

    examples = [
        {
            'input':'''theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
  intro h
  by_cases hp : P
  . constructor
    . exact hp
    . intro hq
      exact h (fun _ => hq)
  . exact Classical.not_imp.mp h''',
            'output':'''theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
  intro h
  push_neg at h
  exact h'''
        },
        {
            'input':'''theorem double_negation (P : Prop) : P → ¬¬P := by
  --using contradiction
  intro h h1
  contradiction''',
            'output':'''theorem double_negation (P : Prop) : P → ¬¬P := by
  --directly
  intro h h1
  exact h1 h'''
        },
        {
            'input':'''theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  intro h nq
  have nhq := not_or_of_imp h
  rcases nhq with hnp | hq
  . exact hnp
  . exfalso
    contradiction''',
            'output':'''theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  intro h nq p
  exact nq (h p)'''
        },
        {
            'input':'''theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
  exact Set.ext fun x => { mp := fun h => And.casesOn h fun hs _ => hs, mpr := fun h => ⟨h, h⟩ }''',
            'output':'''theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
  ext x
  constructor
  . intro h
    rcases h with ⟨hs,_⟩
    exact hs
  . intro h
    constructor
    . exact h
    . exact h'''
        },
        {
            'input':'''theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
  exact (funext fun x => id rfl)''',
            'output':'''theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
  ext x
  unfold Function.comp
  exact rfl'''
        }
    ]

    return Metric('READABILITY', [sys_prompt,user_prompt], readability_fn,examples,'MAX')

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
  




if __name__=='__main__':
    metric = length_metric()
    