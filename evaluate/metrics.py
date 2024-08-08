import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from models.rag import *
from evaluate.build_prooftree import *
import re


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
        # thm.proof = elim_overlap(thm.proof)
        num_lines = len(elim_overlap(thm.proof))
        # return num_lines
        # dont count semicolons
        semicolons = 0
        for line in thm.proof:
            # ignore <;>'s and ;'s at end of line
            content = line.tactic.replace("<;>", "")[:-1]
            semicolons += content.count(";")
        return num_lines + semicolons

    sys_prompt = (
        "system",
        """You are an AI assistant who shortens Lean 4 proofs while ensuring their correctness. 
                  You will aim to reduce the number of lines of the tactic proof while ensuring that it properly compiles in lean 4.""",
    )

    user_prompt = (
        "human",
        """Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short in length - measured in 
                   the number of lines of the proof - as possible, while also ensuring that the output is still syntactically correct.""",
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
    ]

    return Metric("LENGTH", [sys_prompt, user_prompt], examples, "MIN", score_fn=len_fn)


def modularity_metric():

    def get_haves(anno_thm: AnnotatedTheorem):
        haves = [
            (idx, step)
            for idx, step in enumerate(anno_thm.proof)
            if "have" in step.tactic
        ]

        def get_name(line):
            pattern = r"have\s+(\w+)(?=\s*:)"
            match = re.search(pattern, line)
            if match:
                return match.group(1)
            return None

        def get_instances(name, ignore_def=True, start_idx=0):
            tactics = [
                step.tactic
                for idx, step in enumerate(elim_overlap(anno_thm.proof))
                if idx >= start_idx
            ]
            cnt = sum(1 for tac in tactics if name in tac)
            if ignore_def:
                return cnt - 1
            else:
                return cnt

        haves_with_name = {
            step.tactic: (idx, get_name(step.tactic))
            for idx, step in haves
            if get_name(step.tactic) is not None
        }

        num_haves = len(haves)
        avg_reuse = (
            (
                sum(
                    get_instances(
                        haves_with_name[tac][1], start_idx=haves_with_name[tac][0]
                    )
                    for tac in haves_with_name.keys()
                )
            )
            / num_haves
            if num_haves != 0
            else 0
        )
        return (num_haves, avg_reuse)

    def mod_fn(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm, force=True)
        G, _, _ = getProofTree(thm)
        tree_depth = depth(G)
        tree_breadth = breadth(G)
        num_haves, avg_reuse_cnt = get_haves(thm)
        normalized_depth = 1 - tree_depth / len(thm.proof)  # shorter is better
        normalized_breadth = tree_breadth / len(thm.proof)  # bigger is better
        normalized_haves = num_haves / len(thm.proof)  # bigger is better
        normalized_reuse = avg_reuse_cnt / len(thm.proof)  # bigger is better
        vals = [
            normalized_depth,
            normalized_breadth,
            normalized_haves,
            normalized_reuse,
        ]
        weights = [0.25, 0.25, 0.25, 0.25]

        return sum(weights[i] * vals[i] for i in range(len(weights)))

    sys_prompt = (
        "system",
        """You are an AI assistant who rewrites Lean 4 proofs to be more modular while ensuring their correctness. 
                  We measure modularity by considering the depth of the proof tree (less is better), the breadth of the proof tree (greater is better),
                  the number of have statements (more is better), and the avg number of times these have statements are reused (more is better).""",
    )

    user_prompt = (
        "human",
        """Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is more modular. Any lemmas or 
                   independent subproofs you wish to make, put them as a \"have\" statement proofstep within the tactic proof rather than an external lemma.
                   Rewrite the current theorem to be as modular as possible, in that it has as many have statements as possible that are reused often, and the proof tree is broad and not deep - while ensuring the output is still syntactically correct.""",
    )

    examples = []

    return Metric(
        "MODULARITY", [sys_prompt, user_prompt], examples, "MAX", score_fn=mod_fn
    )


def modularity_metric2():

    def mod_fn(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm, force=True)
        G, _, _ = getProofTree(thm)
        partition = nx.algorithms.community.greedy_modularity_communities(G)
        score = nx.algorithms.community.modularity(G, partition)
        return score

    sys_prompt = (
        "system",
        """You are an AI assistant who rewrites Lean 4 proofs to be more modular while ensuring their correctness. 
                  We measure modularity by considering the depth of the proof tree (less is better), the breadth of the proof tree (greater is better),
                  the number of have statements (more is better), and the avg number of times these have statements are reused (more is better).""",
    )

    user_prompt = (
        "human",
        """Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is more modular. Any lemmas or 
                   independent subproofs you wish to make, put them as a \"have\" statement proofstep within the tactic proof rather than an external lemma.
                   Rewrite the current theorem to be as modular as possible, in that it has as many have statements as possible that are reused often, and the proof tree is broad and not deep - while ensuring the output is still syntactically correct.""",
    )

    examples = []

    return Metric(
        "MODULARITY", [sys_prompt, user_prompt], examples, "MAX", score_fn=mod_fn
    )


# I Dont think this satisfies the triangle inequality, so refinement state is locked to always compare with original thm.
# Therefore, it is essentially just best of n, with some error message forwarding
def similarity_metric():

    def diff_fn(thm1, thm2):
        if type(thm1) == Theorem:
            thm1 = annotateTheorem(thm1, force=True)
        if type(thm2) == Theorem:
            thm2 = annotateTheorem(thm2, force=True)
        G1, _, _ = getProofTree(thm1)
        G2, _, _ = getProofTree(thm2)

        return tree_edit_distance(G1, G2)

    def cmp_fn(*thms):
        if len(thms) == 0:
            return None
        elif len(thms) == 1:
            return thms[0]
        else:
            main = thms[0]
            rest = thms[1:]
            scores = [(other, diff_fn(main, other)) for other in rest]
            scores.append((main, 0))
            return max(*scores, key=lambda x: x[1])[0]

    sys_prompt = (
        "system",
        """You are an AI assistant who rewrites a given Lean 4 proof to be as different as possible while ensuring its correctness. 
                  We calculate the similarity of two proofs via the number of insertions, deletions, and modifications (string edit distance) to convert one proof tree to the other.
                  That means that simply reordering some independent steps of a proof will not yield a vastly different proof, rather, one should focus 
                  on creating a proof that is both correct, and structurally and syntactically different than the original""",
    )

    user_prompt = (
        "human",
        """Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it both correct and the new proof is as different as possible 
                   from the original proof.""",
    )

    examples = [
        {
            "input": """theorem add_zero (a : Nat) : a + 0 = a := by
  --induction
  induction a with
  | zero => rfl
  | succ a ih => rw [Nat.succ_add, ih]""",
            "output": """theorem add_zero (a : Nat) : a + 0 = a := by
  --rewriting
  rw [Nat.add_comm, Nat.zero_add]""",
        },
        {
            "input": """theorem double_negation (P : Prop) : P → ¬¬P := by
  --using contradiction
  intro h h1
  contradiction""",
            "output": """theorem double_negation (P : Prop) : P → ¬¬P := by
  --directly
  intro h h1
  exact h1 h""",
        },
        {
            "input": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
  --bash cases
  intro h
  by_cases hp : P
  . constructor
    . exact hp
    . intro hq
      exact h (fun _ => hq)
  . exact Classical.not_imp.mp h""",
            "output": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
  --push negations inwards
  intro h
  push_neg at h
  exact h""",
        },
        {
            "input": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  -- direct
  intro h nq p
  exact nq (h p)""",
            "output": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  -- logical equivalences
  intro h nq
  have nhq := not_or_of_imp h
  rcases nhq with hnp | hq
  . exact hnp
  . exfalso
    contradiction""",
        },
    ]

    return Metric(
        "SIMILARITY",
        [sys_prompt, user_prompt],
        examples,
        "MAX",
        metric_fn=diff_fn,
        cmp=cmp_fn,
        lock_refinement_state=True,
    )


def readability_metric():
    def readability_fn(thm):
        if type(thm) == Theorem:
            thm = annotateTheorem(thm, force=True)
        proof = elim_overlap(thm.proof)
        num_lines = len(proof)
        mean_line_length = (
            sum(
                (
                    len(step.tactic.splitlines()[0])
                    if step.tactic.splitlines() != []
                    else 1
                )
                for step in proof
            )
            / num_lines
        )
        max_line_length = max(
            len(step.tactic.splitlines()[0]) if step.tactic.splitlines() != [] else 1
            for step in proof
        )
        mod = modularity_metric()
        mod_score = mod.score(thm)
        norm_line_length = mean_line_length / max_line_length
        weights = [0.33, 0.67]
        vals = [mod_score, norm_line_length]
        return sum(weights[i] * vals[i] for i in range(len(vals)))

    sys_prompt = (
        "system",
        """You are an AI assistant who rewrites a given Lean 4 proof to be as readable as possible while ensuring its correctness. 
                  We calculate the readability of a given proof  of two metrics via considering the average length of lines/tactics (shorter is better),
                  as well as the modularity of the proof, as given by structure of the proof tree. Additionally, a proof's readability is higher if the variable names are descriptive and complex and abstract proof terms are replaced with more understandable tactic applications.
                    Specifically, we say a proof is modular if it has low proof tree depth, high breadth, and a relatively high number of independent subproofs that are reused often.""",
    )

    user_prompt = (
        "human",
        """Rewrite the current theorem (wrapped in <CURRENT>...</CURRENT>) so it is more readable to a human - while also ensuring it is syntactically correct.""",
    )

    examples = [
        {
            "input": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
  intro h
  by_cases hp : P
  . constructor
    . exact hp
    . intro hq
      exact h (fun _ => hq)
  . exact Classical.not_imp.mp h""",
            "output": """theorem not_imp (P Q : Prop) : ¬ (P → Q) → P ∧ ¬ Q := by
  intro h
  push_neg at h
  exact h""",
        },
        {
            "input": """theorem double_negation (P : Prop) : P → ¬¬P := by
  --using contradiction
  intro h h1
  contradiction""",
            "output": """theorem double_negation (P : Prop) : P → ¬¬P := by
  --directly
  intro h h1
  exact h1 h""",
        },
        {
            "input": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  intro h nq
  have nhq := not_or_of_imp h
  rcases nhq with hnp | hq
  . exact hnp
  . exfalso
    contradiction""",
            "output": """theorem contraposition (P Q : Prop) : (P → Q) → (¬ Q → ¬ P) := by
  intro h nq p
  exact nq (h p)""",
        },
        {
            "input": """theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
  exact Set.ext fun x => { mp := fun h => And.casesOn h fun hs _ => hs, mpr := fun h => ⟨h, h⟩ }""",
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
  exact (funext fun x => id rfl)""",
            "output": """theorem comp_assoc {α β γ δ : Type*} (f : γ → δ) (g : β → γ) (h : α → β) : (f ∘ g) ∘ h = f ∘ (g ∘ h) := by
  ext x
  unfold Function.comp
  exact rfl""",
        },
    ]

    return Metric(
        "READABILITY",
        [sys_prompt, user_prompt],
        examples,
        "MAX",
        score_fn=readability_fn,
    )


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
        partition = nx.algorithms.community.greedy_modularity_communities(G)
        score = nx.algorithms.community.modularity(G, partition)
        save_tree2(
            G,
            p,
            l,
            os.path.join(root_path, ".trees", "mod", f"thm_{i}.png"),
            partition=partition,
        )

        print(f"{thm.decl}:{score}\n=========")
