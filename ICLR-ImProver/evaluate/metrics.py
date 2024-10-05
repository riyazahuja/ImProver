import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from models.rag import *
from evaluate.build_prooftree import *


"""
Metric class definition and basic metrics used for our main results:

Note that the "Readability metric" described in the paper is actually called the modularity metric in our code.
"""


class Metric:
    def __init__(
        self,
        name,
        prompt,
        examples,
        minmax,
        score_fn=None,
        metric_fn=None,
        cmp=None,
        lock_refinement_state=False,
    ):
        if score_fn is None and metric_fn is None:
            raise ValueError("Need either score or metric fn")
        if score_fn is None and cmp is None:
            raise ValueError("Need either score or cmp fn")
        self.name = name
        self.prompt = prompt
        self.examples = examples
        self.minmax = minmax
        self.vs = self.get_example_selector()
        self.score_fn = score_fn
        self.lock_refinement_state = lock_refinement_state

        if cmp is not None:
            self.cmp = cmp
        else:
            self.cmp = self.get_cmp()

        if metric_fn is None:
            self.metric_fn = self.delta
        else:
            self.metric_fn = metric_fn

    def score(self, thm):
        return self.score_fn(thm)

    def metric(self, thm1, thm2):
        return self.metric_fn(thm1, thm2)

    def delta(self, old_thm, new_thm):
        if self.score_fn != None:
            old = self.score(old_thm)
            new = self.score(new_thm)
            return ((new - old) / old) if old != 0 else None
        else:
            return None

    def get_example_selector(self):
        if len(self.examples) == 0:
            return None

        vs = get_metric_vs(self.examples, self.name)
        return vs

    def get_cmp(self):
        if self.score_fn is not None:

            def cmp(*thms):
                scores = [(x, self.score(x)) for x in thms]
                if self.minmax == "MIN":
                    # smaller is better
                    return min(*scores, key=lambda x: x[1])[0]
                elif self.minmax == "MAX":
                    return max(*scores, key=lambda x: x[1])[0]
                else:
                    return None

            return cmp

        else:
            raise ValueError("called get_cmp on fn without scorefn")


def length_metric():

    def len_fn(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm, force=True)
        G, _, _ = getProofTree(thm)
        return G.number_of_nodes()
        # # thm.proof = elim_overlap(thm.proof)
        # num_lines = len(elim_overlap(thm.proof))
        # # return num_lines
        # # dont count semicolons
        # semicolons = 0
        # for line in thm.proof:
        #     # ignore <;>'s and ;'s at end of line
        #     content = line.tactic.replace("<;>", "")[:-1]
        #     semicolons += content.count(";")
        # return num_lines + semicolons

    sys_prompt = (
        "system",
        """You are an AI assistant who shortens Lean 4 proofs while ensuring their correctness. 
                  You will aim to reduce the length of the tactic proof (measured by the number of tactics invoked) while ensuring that it is correct and properly compiles in lean 4.""",
    )

    user_prompt = (
        "human",
        """Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in 
                   the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem.""",
    )

    examples = [
        {
            "input": """theorem add_zero (n : ℕ) : n + 0 = n := by
  induction n with
  | zero =>
    rw [Nat.add_eq_right]
  | succ n ih =>
    rw [Nat.add_assoc, Nat.add_comm 1 0, ← Nat.add_assoc,ih]""",
            "output": """theorem add_zero (n : ℕ) : n + 0 = n := by
  exact rfl
""",
        },
        {
            "input": """theorem imp_trans {P Q R : Prop} (h1 : P → Q) (h2 : Q → R) : P → R := by
  intro hp
  apply h2
  apply h1
  exact hp""",
            "output": """theorem imp_trans {P Q R : Prop} (h1 : P → Q) (h2 : Q → R) : P → R := by
  exact (fun p => h2 (h1 p))""",
        },
        {
            "input": """theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
  ext x
  constructor
  . intro h
    rcases h with ⟨hs,_⟩
    exact hs
  . intro h
    constructor
    . exact h
    . exact h""",
            "output": """theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
  exact Set.ext fun x => { mp := fun h => And.casesOn h fun hs _ => hs, mpr := fun h => ⟨h, h⟩ }""",
        },
        {
            "input": """theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
  ext x
  unfold Function.comp
  exact rfl""",
            "output": """theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
  exact (funext fun x => id rfl)""",
        },
        {
            "input": """example (P Q R S : Prop) (h1 : P → Q) (h2 : Q → R) (h3 : R → S) (h4 : P)  : S := by
  apply h3
  apply h2
  apply h1
  apply h4""",
            "output": """example (P Q R S : Prop) (h1 : P → Q) (h2 : Q → R) (h3 : R → S) (h4 : P)  : S := by
    exact h3 (h2 (h1 h4))""",
        },
        {
            "input": """example (P Q :Prop): P ∨ Q → Q ∨ P := by
  intro h
  cases h with
  | inl hp => exact Or.inr hp
  | inr hq => exact Or.inl hq""",
            "output": """example (P Q :Prop): P ∨ Q → Q ∨ P := by
  rintro (hp | hq)
  . exact Or.inr hp
  . exact Or.inl hq""",
        },
        {
            "input": """example (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  intro h nq
  have nhq := by
    exact not_or_of_imp h
  rcases nhq with hnp | hq
  . exact hnp
  . exfalso
    contradiction""",
            "output": """example (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  intro h nq p
  exact nq (h p)""",
        },
        {
            "input": """example : (P → Q) ∧ (Q → R) → P → R := by
  intro h p
  rcases h with ⟨a,b⟩
  apply b
  apply a
  exact p""",
            "output": """example : (P → Q) ∧ (Q → R) → P → R := by
  rintro (⟨hpq,hqr⟩) hp
  exact hqr (hpq hp)""",
        },
        {
            "input": """example (h : P → Q) (h1 : P ∧ R) : Q ∧ R := by
  rcases h1 with ⟨p,r⟩
  constructor
  exact h p
  exact r""",
            "output": """example (h : P → Q) (h1 : P ∧ R) : Q ∧ R := by
  exact And.imp_left h h1""",
        },
        {
            "input": """example (h : ¬ (P ∧ Q)) : P → ¬ Q := by
  intro p opp
  have duh : P ∧ Q := by
    constructor
    exact p
    exact opp
  exact h duh""",
            "output": """example (h : ¬ (P ∧ Q)) : P → ¬ Q := by
  intro hp hq
  apply h
  exact ⟨hp,hq⟩""",
        },
    ]

    return Metric("LENGTH", [sys_prompt, user_prompt], examples, "MIN", score_fn=len_fn)


def modularity_metric():

    def count_haves(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm, force=True)
        pf = [tac.tactic for tac in thm.proof]
        return sum(1 for tactic in pf if tactic.strip().startswith("have")) / len(
            thm.proof
        )

    sys_prompt = (
        "system",
        """You are an AI assistant who rewrites Lean 4 proofs to be more readable while ensuring their correctness. We measure readablity by considering the ratio of the number of explicitly typed 'have' tactics against the total number of tactics in the proof, as this is proportional to whether a proof is declarative in style, and thus, readable.""",
    )

    user_prompt = (
        "human",
        """Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is more readable and declarative and modular, while also being correct.""",
    )

    examples = [
        {
            "input": """
example (P Q R S : Prop) (h1 : P → Q) (h2 : Q → R) (h3 : R → S) (h4 : P)  : S := by
  apply h3
  apply h2
  apply h1
  apply h4
""",
            "output": """
example (P Q R S : Prop) (h1 : P → Q) (h2 : Q → R) (h3 : R → S) (h4 : P) : S := by
  apply h3
  apply h2
  have bar := by
    apply h1
    apply h4
  exact bar
""",
        },
        {
            "input": """
example (P Q R S : Prop) (h1 : P → Q) (h2 : Q → R) (h3 : R → S) (h4 : P) : S := by
  have h1' := by exact h1
  have h2' := by exact h2
  have h3' := by exact h3
  have h4' := by exact h4

  apply h3'
  apply h2'
  apply h1'
  apply h4'
""",
            "output": """
example (P Q R S : Prop) (h1 : P → Q) (h2 : Q → R) (h3 : R → S) (h4 : P) : S := by
  exact h3 (h2 (h1 h4))
""",
        },
        {
            "input": """
example (P Q R S : Prop) (h1 : P → Q) (h2 : Q → R) (h3 : R → S) (h4 : P)  : S := by
  have all := by
    apply h3
    apply h2
    apply h1
    apply h4
  exact all
""",
            "output": """
example (P Q R S : Prop) (h1 : P → Q) (h2 : Q → R) (h3 : R → S) (h4 : P) : S := by
  exact h3 (h2 (h1 h4))
""",
        },
        {
            "input": """
example (P Q :Prop): P ∨ Q → Q ∨ P := by
  intro h
  cases h with
  | inl hp => exact Or.inr hp
  | inr hq => exact Or.inl hq
""",
            "output": """
example (P Q :Prop): P ∨ Q → Q ∨ P := by
  rintro (hp | hq)
  . exact Or.inr hp
  . exact Or.inl hq
""",
        },
        {
            "input": """
example (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  intro h nq
  have nhq := by
    exact not_or_of_imp h
  rcases nhq with hnp | hq
  . exact hnp
  . exfalso
    contradiction
""",
            "output": """
example (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  intro h nq p
  exact nq (h p)
""",
        },
        {
            "input": """
example : (P → Q) ∧ (Q → R) → P → R := by
  intro h p
  rcases h with ⟨a,b⟩
  apply b
  apply a
  exact p
""",
            "output": """
example : (P → Q) ∧ (Q → R) → P → R := by
  rintro (⟨hpq,hqr⟩) hp
  exact hqr (hpq hp)
""",
        },
        {
            "input": """
example (h : P → Q) (h1 : P ∧ R) : Q ∧ R := by
  rcases h1 with ⟨p,r⟩
  constructor
  exact h p
  exact r
""",
            "output": """
example (h : P → Q) (h1 : P ∧ R) : Q ∧ R := by
  exact And.imp_left h h1
""",
        },
        {
            "input": """
example (h1 : P ∨ Q → R) : (P → R) ∧ (Q → R) := by
  constructor
  intro p
  have duh : P ∨ Q := by
    left
    exact p
  exact h1 duh
  intro q
  have duh : P ∨ Q := by
    right
    exact q
  exact h1 duh
""",
            "output": """
example (h1 : P ∨ Q → R) : (P → R) ∧ (Q → R) := by
  constructor
  . intro p
    exact h1 (Or.inl p)
  . intro q
    exact h1 (Or.inr q)
""",
        },
        {
            "input": """
example (h : ¬ (P ∧ Q)) : ¬ P ∨ ¬ Q := by
  have hmm : P → ¬ Q := by
    intro p opp
    have duh : P ∧ Q := by
      constructor
      exact p
      exact opp
    exact h duh
  by_cases duh:P
  right
  exact hmm duh
  left
  exact duh
""",
            "output": """
example (h : ¬ (P ∧ Q)) : ¬ P ∨ ¬ Q := by
  push_neg at h
  exact not_or_of_imp h
""",
        },
    ]

    return Metric(
        "MODULARITY",
        [sys_prompt, user_prompt],
        examples,
        "MAX",
        score_fn=count_haves,
    )


# NOTE: NOT IN ICLR PAPER
# def similarity_metric():

#     def diff_fn(thm1, thm2):
#         if type(thm1) == Theorem:
#             thm1 = annotateTheorem(thm1, force=True)
#         if type(thm2) == Theorem:
#             thm2 = annotateTheorem(thm2, force=True)
#         G1, _, _ = getProofTree(thm1)
#         G2, _, _ = getProofTree(thm2)

#         return tree_edit_distance(G1, G2)

#     def cmp_fn(*thms):
#         if len(thms) == 0:
#             return None
#         elif len(thms) == 1:
#             return thms[0]
#         else:
#             main = thms[0]
#             rest = thms[1:]
#             scores = [(other, diff_fn(main, other)) for other in rest]
#             scores.append((main, 0))
#             return max(*scores, key=lambda x: x[1])[0]

#     sys_prompt = (
#         "system",
#         """You are an AI assistant who rewrites a given Lean 4 proof to be as different as possible while ensuring its correctness.
#                   We calculate the similarity of two proofs via the number of insertions, deletions, and modifications (string edit distance) to convert one proof tree to the other.
#                   That means that simply reordering some independent steps of a proof will not yield a vastly different proof, rather, one should focus
#                   on creating a proof that is both correct, and structurally and syntactically different than the original""",
#     )

#     user_prompt = (
#         "human",
#         """Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it both correct and the new proof is as different as possible
#                    from the original proof.""",
#     )

#     examples = [
#         {
#             "input": """theorem add_zero (a : Nat) : a + 0 = a := by
#   --induction
#   induction a with
#   | zero => rfl
#   | succ a ih => rw [Nat.succ_add, ih]""",
#             "output": """theorem add_zero (a : Nat) : a + 0 = a := by
#   --rewriting
#   rw [Nat.add_comm, Nat.zero_add]""",
#         },
#         {
#             "input": """theorem double_negation (P : Prop) : P → ¬¬P := by
#   --using contradiction
#   intro h h1
#   contradiction""",
#             "output": """theorem double_negation (P : Prop) : P → ¬¬P := by
#   --directly
#   intro h h1
#   exact h1 h""",
#         },
#         {
#             "input": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
#   --bash cases
#   intro h
#   by_cases hp : P
#   . constructor
#     . exact hp
#     . intro hq
#       exact h (fun _ => hq)
#   . exact Classical.not_imp.mp h""",
#             "output": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
#   --push negations inwards
#   intro h
#   push_neg at h
#   exact h""",
#         },
#         {
#             "input": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
#   -- direct
#   intro h nq p
#   exact nq (h p)""",
#             "output": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
#   -- logical equivalences
#   intro h nq
#   have nhq := not_or_of_imp h
#   rcases nhq with hnp | hq
#   . exact hnp
#   . exfalso
#     contradiction""",
#         },
#     ]

#     return Metric(
#         "SIMILARITY",
#         [sys_prompt, user_prompt],
#         examples,
#         "MAX",
#         metric_fn=diff_fn,
#         cmp=cmp_fn,
#         lock_refinement_state=True,
#     )

# NOTE: NOT IN ICLR PAPER
# def readability_metric():
#     def readability_fn(thm):
#         if type(thm) == Theorem:
#             thm = annotateTheorem(thm, force=True)
#         proof = elim_overlap(thm.proof)
#         num_lines = len(proof)
#         mean_line_length = (
#             sum(
#                 (
#                     len(step.tactic.splitlines()[0])
#                     if step.tactic.splitlines() != []
#                     else 1
#                 )
#                 for step in proof
#             )
#             / num_lines
#         )
#         max_line_length = max(
#             len(step.tactic.splitlines()[0]) if step.tactic.splitlines() != [] else 1
#             for step in proof
#         )
#         mod = modularity_metric()
#         mod_score = mod.score(thm)
#         norm_line_length = mean_line_length / max_line_length
#         weights = [0.33, 0.67]
#         vals = [mod_score, norm_line_length]
#         return sum(weights[i] * vals[i] for i in range(len(vals)))

#     sys_prompt = (
#         "system",
#         """You are an AI assistant who rewrites a given Lean 4 proof to be as readable as possible while ensuring its correctness.
#                   We calculate the readability of a given proof  of two metrics via considering the average length of lines/tactics (shorter is better),
#                   as well as the modularity of the proof, as given by structure of the proof tree. Additionally, a proof's readability is higher if the variable names are descriptive and complex and abstract proof terms are replaced with more understandable tactic applications.
#                     Specifically, we say a proof is modular if it has low proof tree depth, high breadth, and a relatively high number of independent subproofs that are reused often.""",
#     )

#     user_prompt = (
#         "human",
#         """Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is more readable to a human - while also ensuring it is syntactically correct.""",
#     )

#     examples = [
#         {
#             "input": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
#   intro h
#   by_cases hp : P
#   . constructor
#     . exact hp
#     . intro hq
#       exact h (fun _ => hq)
#   . exact Classical.not_imp.mp h""",
#             "output": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
#   intro h
#   push_neg at h
#   exact h""",
#         },
#         {
#             "input": """theorem double_negation (P : Prop) : P → ¬¬P := by
#   --using contradiction
#   intro h h1
#   contradiction""",
#             "output": """theorem double_negation (P : Prop) : P → ¬¬P := by
#   --directly
#   intro h h1
#   exact h1 h""",
#         },
#         {
#             "input": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
#   intro h nq
#   have nhq := not_or_of_imp h
#   rcases nhq with hnp | hq
#   . exact hnp
#   . exfalso
#     contradiction""",
#             "output": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
#   intro h nq p
#   exact nq (h p)""",
#         },
#         {
#             "input": """theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
#   exact Set.ext fun x => { mp := fun h => And.casesOn h fun hs _ => hs, mpr := fun h => ⟨h, h⟩ }""",
#             "output": """theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
#   ext x
#   constructor
#   . intro h
#     rcases h with ⟨hs,_⟩
#     exact hs
#   . intro h
#     constructor
#     . exact h
#     . exact h""",
#         },
#         {
#             "input": """theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
#   exact (funext fun x => id rfl)""",
#             "output": """theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
#   ext x
#   unfold Function.comp
#   exact rfl""",
#         },
#     ]

#     return Metric(
#         "READABILITY",
#         [sys_prompt, user_prompt],
#         examples,
#         "MAX",
#         score_fn=readability_fn,
#     )


def completion_metric():
    def num_errors(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm)
        errors = sum(1 for msg in thm.messages if msg.severity == "error")
        return errors

    sys_prompt = (
        "system",
        'You are an AI assistant who automatically solves Lean 4 proofs (as in, generates the tactic proof) and ensures its correctness. You will recieve a Lean 4 proof you must modify to eliminate any errors so that it compiles as correct and, and elimanate any "sorry"s with full proofs.',
    )
    user_prompt = (
        "human",
        "Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is a formal, complete, and correct Lean 4 proof by filling in its tactic proof.",
    )

    examples = [
        {
            "input": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
    sorry""",
            "output": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
    intro h
    push_neg at h
    exact h""",
        },
        {
            "input": """theorem double_negation (P : Prop) : P → ¬¬P := by
   sorry""",
            "output": """theorem double_negation (P : Prop) : P → ¬¬P := by
    --directly
    intro h h1
    exact h1 h""",
        },
        {
            "input": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
   sorry""",
            "output": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
    intro h nq p
    exact nq (h p)""",
        },
        {
            "input": """theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
    sorry""",
            "output": """theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
    ext x
    constructor
    . intro h
        rcases h with ⟨hs,_⟩
        exact hs
    . intro h
        constructor
        . exact h
        . exact h""",
        },
        {
            "input": """theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
    sorry""",
            "output": """theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
    ext x
    unfold Function.comp
    exact rfl""",
        },
    ]

    return Metric(
        "COMPLETION", [sys_prompt, user_prompt], examples, "MIN", score_fn=num_errors
    )


if __name__ == "__main__":

    repo = getRepo("Tests", "configs/config_test.json")
    files = {file.file_name: file for file in repo.files}

    f = files["Basic.lean"]
    thms = f.theorems
    for i, thm in enumerate(thms):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm, force=True)
        G, p, l = getProofTree(thm)
        score = calculate_modularity(G)

        save_tree(
            G,
            p,
            l,
            os.path.join(root_path, ".trees", "mod", f"thm_{i}.png"),
            show_mod=True,
        )

        print(f"{thm.decl}\n {score}\n=========")
