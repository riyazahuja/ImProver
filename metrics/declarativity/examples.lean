import ImProver.metrics.tagger
import Mathlib.Data.Set.Lattice
import Mathlib.Data.Set.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Data.Nat.Prime.Basic

namespace declarativityExamples


@[improver_example have_reuse, version unoptimized]
theorem foo {a b c d : ℝ} :
    max a b + max c d = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by
  rcases le_total a b with h_ab | h_ba
  · rcases le_total c d with h_cd | h_dc
    -- Case 1: a ≤ b and c ≤ d
    · calc max a b + max c d
        _ = b + d := by rw [max_eq_right h_ab, max_eq_right h_cd]
        _ = max (b+c) (b+d) := by rw [max_eq_right (add_le_add_left h_cd b)]
        _ = max (max (a+d) (b+c)) (b+d) := by
          rw [max_eq_right (add_le_add_left h_cd b)]
          apply symm
          apply @max_eq_right _ _ (max (a+d) (b+c)) (b+d)
          rw [max_le_iff]
          constructor
          . linarith
          . exact add_le_add_left h_cd b
        _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ab, h_cd]
    -- Case 2: a ≤ b and d ≤ c
    · calc max a b + max c d
        _ = b + c := by rw [max_eq_right h_ab, max_eq_left h_dc]
        _ = max (b+c) (b+d) := by rw [max_eq_left (add_le_add_left h_dc b)]
        _ = max (max (a+d) (b+c)) (b+d) := by
          rw [max_eq_left (add_le_add_left h_dc b)]
          apply symm
          rw [max_assoc (a+d) (b+c) (b+d), max_comm (b+c) (b+d), ← max_assoc (a+d) (b+d) (b+c)]
          apply @max_eq_right _ _ (max (a+d) (b+d)) (b+c)
          rw [max_le_iff]
          constructor
          . linarith
          . exact add_le_add_left h_dc b
        _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ab, h_dc]; linarith
  · rcases le_total c d with h_cd | h_dc
    -- Case 3: b ≤ a and c ≤ d
    · calc max a b + max c d
        _ = a + d := by rw [max_eq_left h_ba, max_eq_right h_cd]
        _ = max (a+c) (a+d) := by rw [max_eq_right (add_le_add_left h_cd a)]
        _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ba, h_cd]
    -- Case 4: b ≤ a and d ≤ c
    · calc max a b + max c d
          _ = a + c := by rw [max_eq_left h_ba, max_eq_left h_dc]
          _ = max (a+c) (a+d) := by rw [max_eq_left (add_le_add_left h_dc a)]
          _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ba, h_dc]



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






section

theorem even_of_even_sqr {m : ℕ} (h : 2 ∣ m ^ 2) : 2 ∣ m := by
  rw [pow_two, Nat.prime_two.dvd_mul] at h
  cases h <;> assumption

@[improver_example add_declarativization, version unoptimized]
theorem bar {m n : ℕ} (coprime_mn : m.Coprime n) : m ^ 2 ≠ 2 * n ^ 2 := by
  intro sqr_eq
  rcases dvd_iff_exists_eq_mul_left.mp (even_of_even_sqr (Dvd.intro (n ^ 2) (id (Eq.symm sqr_eq)))) with ⟨k, meq⟩
  have : 2 * (2 * k ^ 2) = 2 * n ^ 2 := by
    rw [← sqr_eq, meq]
    ring
  have : 2 ∣ m.gcd n := by
    apply Nat.dvd_gcd
    . exact even_of_even_sqr (Dvd.intro (n ^ 2) (id (Eq.symm sqr_eq)))
    . apply even_of_even_sqr
      rw [← (mul_right_inj' (by norm_num)).mp this]
      apply dvd_mul_right
  have : 2 ∣ 1 := by
    convert this
    symm
    exact coprime_mn
  norm_num at this


@[improver_example add_declarativization, version optimized]
theorem bar'{m n : ℕ} (coprime_mn : m.Coprime n) : m ^ 2 ≠ 2 * n ^ 2 := by
  intro sqr_eq
  have : 2 ∣ m := by
    apply even_of_even_sqr
    rw [sqr_eq]
    apply dvd_mul_right
  obtain ⟨k, meq⟩ := dvd_iff_exists_eq_mul_left.mp this
  have : 2 * (2 * k ^ 2) = 2 * n ^ 2 := by
    rw [← sqr_eq, meq]
    ring
  have : 2 * k ^ 2 = n ^ 2 :=
    (mul_right_inj' (by norm_num)).mp this
  have : 2 ∣ n := by
    apply even_of_even_sqr
    rw [← this]
    apply dvd_mul_right
  have : 2 ∣ m.gcd n := by
    apply Nat.dvd_gcd <;>
    assumption
  have : 2 ∣ 1 := by
    convert this
    symm
    exact coprime_mn
  norm_num at this


end

end declarativityExamples
