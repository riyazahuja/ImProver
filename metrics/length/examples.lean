import ImProver.metrics.tagger
import Mathlib.Data.Set.Lattice
import Mathlib.Data.Set.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic

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
