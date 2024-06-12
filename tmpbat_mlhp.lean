import Mathlib.Tactic

variable (P Q R S : Prop)

/-
Replace the following sorry's by proofs.
-/

example : (P → Q) ∧ (Q → R) → P → R := by
  intro h p
  rcases h with ⟨a,b⟩
  apply b
  apply a
  exact p

example (h : P → Q) (h1 : P ∧ R) : Q ∧ R := by
  rcases h1 with ⟨p,r⟩
  constructor
  exact h p
  exact r

example (h : ¬ (P ∧ Q)) : P → ¬ Q := by
  intro p opp
  have duh : P ∧ Q := by
    constructor
    exact p
    exact opp
  exact h duh

example (h : ¬ (P → Q)) : ¬ Q := by
  intro opp
  have duh : P → Q := by
    intro _
    exact opp
  exact h duh

example (h : P ∧ ¬ Q) : ¬ (P → Q) := by
  rcases h with ⟨p,nq⟩
  intro huh
  have duh := huh p
  contradiction




example (h1 : P ∨ Q) (h2 : P → R) : R ∨ Q := by
  rcases h1 with a|b
  left
  exact h2 a
  right
  exact b


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


example (h1 : P → R) (h2 : Q → R) : P ∨ Q → R := by
  intro pq
  rcases pq with a|b
  exact h1 a
  exact h2 b
example (h : ¬ (P ∨ Q)) : ¬ P ∧ ¬ Q  := by
  constructor
  intro p
  have duh : P∨ Q := by
    left
    exact p
  left
  exact p
  exact h duh
  intro q
  have duh : P∨ Q := by
    right
    exact q
  right
  exact q
  exact h duh
