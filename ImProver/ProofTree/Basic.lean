import Mathlib.Data.Nat.Basic
import Mathlib.Tactic


theorem duh (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  constructor
  . rcases h with ⟨h1, h2⟩
    exact h2
  . exact h.1


theorem duh2 (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  constructor
  . exact h.2
  . exact h.1

theorem duh3 (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  have right : p := by
    exact h.1
  constructor
  . exact h.2
  . exact right



theorem example_with_simp_all (p q r : Prop) :
  p → (p → q) → (q → r) → p ∧ r := by
  intro hp hpq hqr
  constructor
  · exact hp
  . exact hqr (hpq hp)



theorem duh' : ∃n:ℕ, 2+3 = n := by
  constructor
  simp
  case w => use 5
  rfl
