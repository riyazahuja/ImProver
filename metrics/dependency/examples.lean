import ImProver.metrics.tagger
import Mathlib.Data.Set.Lattice
import Mathlib.Data.Set.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Data.Nat.Prime.Basic

import ImProver.metrics.tagger

@[improver_example test, version unoptimized]
example : True := by
  sorry

@[improver_example test, version optimized]
example : True := by
  trivial


@[improver_example strong_tactics, version unoptimized]
theorem qux {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  apply Left.add_nonneg
  . exact ha
  . exact hb

@[improver_example strong_tactics, version optimized]
theorem qux' {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  linarith



@[improver_example inlining, version unoptimized]
theorem foo {x y : ℝ} : x ≤ y ∧ ¬y ≤ x ↔ x ≤ y ∧ x ≠ y := by
  constructor
  · rintro ⟨h0, h1⟩
    constructor
    · exact h0
    intro h2
    apply h1
    rw [h2]
  rintro ⟨h0, h1⟩
  constructor
  · exact h0
  intro h2
  apply h1
  apply le_antisymm h0 h2

@[improver_example inlining, version optimized]
theorem foo' {x y : ℝ} : x ≤ y ∧ ¬y ≤ x ↔ x ≤ y ∧ x ≠ y  := by
  constructor
  · rintro ⟨h0, h1⟩
    exact ⟨h0, fun h2 => h1 (by rw [h2])⟩
  · rintro ⟨h0, h1⟩
    exact ⟨h0, fun h2 => h1 (by linarith)⟩



@[improver_example have_reuse, version optimized]
theorem foo' {a b c d : ℝ} :
    max a b + max c d = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by
  have lemma_add_distrib : ∀ (x y z : ℝ), z + max x y = max (z + x) (z + y) := by
    intro x y z
    rcases le_total x y with h | h
    · rw [max_eq_right h, max_eq_right (add_le_add_left h z)]
    · rw [max_eq_left h, max_eq_left (add_le_add_left h z)]

  calc max a b + max c d
    _ = max c d + max a b := by rw [add_comm]
    _ = max (max c d + a) (max c d + b) := by rw [lemma_add_distrib]
    _ = max (a + max c d) (b + max c d) := by rw [add_comm (max c d) a, add_comm (max c d) b]
    _ = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by rw [lemma_add_distrib, lemma_add_distrib]

@[improver_example have_reuse, version optimized]
theorem foo'' {a b c d : ℝ} :
    max a b + max c d = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by
  have lemma_add_distrib : ∀ (x y z : ℝ), z + max x y = max (z + x) (z + y) := by
    intro x y z
    rcases le_total x y with h | h
    · simp_all only [sup_of_le_right, add_le_add_iff_left]
    · simp_all only [sup_of_le_left, add_le_add_iff_left]

  calc max a b + max c d
    _ = max c d + max a b := by linarith
    _ = max (max c d + a) (max c d + b) := by simp [lemma_add_distrib]
    _ = max (a + max c d) (b + max c d) := by simp [add_comm]
    _ = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by simp [lemma_add_distrib]



theorem Nat.choose_eq_one_iff {n p : ℕ} : n.choose p = 1 ↔ p = 0 ∨ n = p := by
  induction n generalizing p with
  | zero => cases p <;> simp
  | succ n ih =>
    cases p with
    | zero => simp
    | succ p =>
      simp only [Nat.choose_succ_succ, Nat.choose_eq_zero_iff, Nat.add_eq_one_iff, ih]
      omega


theorem Nat.choose_eq_one_iff' {n p : ℕ} : n.choose p = 1 ↔ p = 0 ∨ n = p := by
  induction n generalizing p with
  | zero => cases p with
    | zero =>
      rw [choose_self, or_self]
      constructor <;> intro _; rfl
      rfl
    | succ p =>
      rw [choose_zero_succ, AddLeftCancelMonoid.add_eq_zero,
         self_eq_add_left]
      constructor
      . intro h
        exact False.elim (one_ne_zero h)
      . intro h
        rcases h with ⟨h0, h1⟩ | h'
        . exact h1
        . exact Nat.eq_zero_of_add_eq_zero_left (id (Eq.symm h'))

  | succ n ih =>
    cases p with
    | zero =>
      rw [choose_zero_right, AddLeftCancelMonoid.add_eq_zero]
      constructor
      . intro h
        exact Or.symm (Or.inr rfl)
      . intro _; rfl
    | succ p =>
      simp only [Nat.choose_succ_succ, Nat.choose_eq_zero_iff, Nat.add_eq_one_iff, ih]
      omega


example (a b c : ℝ) (h : a ≤ b) : c - Real.exp b ≤ c - Real.exp a := by gcongr

example (a b c : ℝ) (h : a ≤ b) : c - Real.exp b ≤ c - Real.exp a := by
  apply sub_le_sub_left
  apply Real.exp_le_exp.mpr
  exact h


section


-- maybe don't do this?
theorem two_le {m : ℕ} (h0 : m ≠ 0) (h1 : m ≠ 1) : 2 ≤ m := by
  cases m; contradiction
  case succ m =>
    cases m; contradiction
    repeat apply Nat.succ_le_succ
    apply zero_le

theorem exists_prime_factor {n : Nat} (h : 2 ≤ n) : ∃ p : Nat, p.Prime ∧ p ∣ n := by
  by_cases np : n.Prime
  · use n, np
  induction' n using Nat.strong_induction_on with n ih
  rw [Nat.prime_def_lt] at np
  push_neg at np
  rcases np h with ⟨m, mltn, mdvdn, mne1⟩
  have : m ≠ 0 := by
    intro mz
    rw [mz, zero_dvd_iff] at mdvdn
    linarith
  have mgt2 : 2 ≤ m := two_le this mne1
  by_cases mp : m.Prime
  · use m, mp
  · rcases ih m mltn mgt2 mp with ⟨p, pp, pdvd⟩
    use p, pp
    apply pdvd.trans mdvdn


theorem exists_prime_factor2 {n : Nat} (h : 2 ≤ n) : ∃ p : Nat, p.Prime ∧ p ∣ n := by
  by_cases np : n.Prime
  · use n, np
  induction' n using Nat.strong_induction_on with n ih
  rw [Nat.prime_def_lt] at np
  push_neg at np
  rcases np h with ⟨m, mltn, mdvdn, mne1⟩
  have : m ≠ 0 := by
    intro mz
    rw [mz, zero_dvd_iff] at mdvdn
    linarith
  have mgt2 : 2 ≤ m := by
    cases m; contradiction
    case succ m =>
      cases m; contradiction
      repeat apply Nat.succ_le_succ
      apply zero_le
  by_cases mp : m.Prime
  · use m, mp
  · rcases ih m mltn mgt2 mp with ⟨p, pp, pdvd⟩
    use p, pp
    apply pdvd.trans mdvdn
