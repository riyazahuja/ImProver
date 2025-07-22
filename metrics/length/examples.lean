import ImProver.metrics.tagger
import Mathlib.Data.Set.Lattice
import Mathlib.Data.Set.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic


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




open Function
open Set

variable {α β : Type*} [Inhabited α]
variable (f : α → β)

noncomputable section
open Classical

def inverse (f : α → β) : β → α := fun y : β ↦
  if h : ∃ x, f x = y then Classical.choose h else default

theorem inverse_spec {f : α → β} (y : β) (h : ∃ x, f x = y) : f (inverse f y) = y := by
  rw [inverse, dif_pos h]
  exact Classical.choose_spec h


@[improver_example inlining2, version unoptimized]
theorem bar : Injective f ↔ LeftInverse (inverse f) f := by
  constructor
  · intro h y
    apply h
    apply inverse_spec
    use y
  intro h x1 x2 e
  rw [← h x1, ← h x2, e]


@[improver_example inlining2, version optimized]
theorem bar' : Injective f ↔ LeftInverse (inverse f) f  := by
  constructor
  · exact fun h y ↦ h (inverse_spec _ ⟨y, rfl⟩)
  · exact fun h x1 x2 e ↦ by rw [←h x1, e, h x2]






theorem baz {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  apply Left.add_nonneg
  . exact ha
  . exact hb


theorem baz' {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  linarith

-- @[improver_example have_reuse, version unoptimized]
-- theorem max_add_max_eq {a b c d : ℝ} :
--     max a b + max c d = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by
--   rcases le_total a b with h_ab | h_ab
--   · rcases le_total c d with h_cd | h_cd
--     -- Case 1: a ≤ b and c ≤ d
--     · calc max a b + max c d
--         _ = b + d := by rw [max_eq_right h_ab, max_eq_right h_cd]
--         _ = max (b+c) (b+d) := by rw [max_eq_right (add_le_add_left h_cd b)]
--         _ = max (max (a+d) (b+c)) (b+d) := by
--           rw [max_eq_right (le_trans (add_le_add_right h_ab d) (@max_le_iff _ _ a b d |>.mp (by simp [h_ab, h_cd, le_refl])))]
--         _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ab, h_cd]

--     · calc max a b + max c d
--         _ = b + c := by rw [max_eq_right h_ab, max_eq_left h_cd]
--         _ = max (b+c) (b+d) := by rw [max_eq_left (add_le_add_left h_cd b)]
--         _ = max (max (a+d) (b+c)) (b+d) := by rw [max_eq_right (le_trans (add_le_add_right h_ab d) (max_le_iff.mp (by simp [h_ab, h_cd, le_refl])))]
--         _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ab, h_cd]; linarith
--   · rcases le_total c d with h_cd | h_cd
--     -- Case 3: b < a and c ≤ d
--     · calc max a b + max c d
--         _ = a + d := by rw [max_eq_left h_ab, max_eq_right h_cd]
--         _ = max (a+c) (a+d) := by rw [max_eq_right (add_le_add_left h_cd a)]
--         _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ab, h_cd]
--     · calc max a b + max c d
--           _ = a + c := by rw [max_eq_left h_ab, max_eq_left h_cd]
--           _ = max (a+c) (a+d) := by rw [max_eq_left (add_le_add_left h_cd a)]
--           _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ab, h_cd]





@[improver_example have_reuse, version optimized]
theorem max_add_max_eq' {a b c d : ℝ} :
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
