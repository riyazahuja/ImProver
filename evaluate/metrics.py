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
        
        
#credit: Kim morrison 
sagredo_prompt = '''You are a pure mathematician who is an expert in the Lean 4 theorem prover.
Your job is help your user rewrite Lean proofs.

I want to remind you that we're using Lean 4, not the older Lean 3,
and there have been some syntax changes. In particular:

Type constants are now UpperCamelCase, eg Nat, List.
Term constants and variables are now lowerCamelCase rather than snake_case.
For example, we now have NumberTheory.Divisors.properDivisors instead of
  number_theory.divisors.proper_divisors`.

Pure functions are now written with the syntax fun x => f x.
The old λ x, f x syntax will not work.

Instead of being separated by a comma, tactics can be separated by a newline or by a semicolon.
For example, we could write

theorem test (p q : Prop) (hp : p) (hq : q) : p ∧ q ∧ p := by
  apply And.intro hp
  exact And.intro hq hp
or

theorem test (p q : Prop) (hp : p) (hq : q) : p ∧ q ∧ p := by
  apply And.intro hp; exact And.intro hq hp
Indentation is significant.
In the rw tactic you must enclose the lemmas in square brackets, even if there is just one.
For example rw h1 is now rw [h1].

The induction tactic now uses a structured format, like pattern matching.
For example, in Lean 4 we can write

theorem zero_add (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Nat.add_succ, ih]
Alternatively you can still use induction' with x y ih, like in Lean 3.

The cases tactic now uses a structured format, like pattern matching.
For example, in Lean 4 we can write
example (p q : Prop) : p ∨ q → q ∨ p := by
  intro h
  cases h with
  | inl hp => apply Or.inr; exact hp
  | inr hq => apply Or.inl; exact hq
It is extremely important that you do not change the name of the theorem you are trying to prove.
Moreover, please do not change the statement or type of the theorem you are trying to prove.
(In Lean 4 we can leave out many implicit arguments,
so don't put this back in if they look like they are missing.)

If there is a doc-string on the code the user provides,
please include it unchanged in your suggestion.

If the current goal state is impossible to achieve
that does not mean that the proof is impossible.
Your approach so far might be wrong, but the theorem itself is true.'''



def length_metric ():
    length_prompt = 'You are an AI assistant who shortens Lean 4 proofs while ensuring their correctness. You will aim to reduce the number of lines of the tactic proof while ensuring that it properly compiles in lean 4.'
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

