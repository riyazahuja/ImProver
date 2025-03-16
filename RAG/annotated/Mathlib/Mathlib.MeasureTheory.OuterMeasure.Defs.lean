/-- An outer measure is a countably subadditive monotone function that sends `∅` to `0`. -/
structure OuterMeasure (α : Type*) where
  /-- Outer measure function. Use automatic coercion instead. -/
  protected measureOf : Set α → ℝ≥0∞
  protected empty : measureOf ∅ = 0
  protected mono : ∀ {s₁ s₂}, s₁ ⊆ s₂ → measureOf s₁ ≤ measureOf s₂
  protected iUnion_nat : ∀ s : ℕ → Set α, Pairwise (Disjoint on s) →
    measureOf (⋃ i, s i) ≤ ∑' i, measureOf (s i)


/-- A mixin class saying that elements `μ : F` are outer measures on `α`.

This typeclass is used to unify some API for outer measures and measures. -/
class OuterMeasureClass (F : Type*) (α : outParam Type*) [FunLike F (Set α) ℝ≥0∞] : Prop where
  protected measure_empty (f : F) : f ∅ = 0
  protected measure_mono (f : F) {s t} : s ⊆ t → f s ≤ f t
  protected measure_iUnion_nat_le (f : F) (s : ℕ → Set α) : Pairwise (Disjoint on s) →
    f (⋃ i, s i) ≤ ∑' i, f (s i)


instance : FunLike (OuterMeasure α) (Set α) ℝ≥0∞ where
  coe m := m.measureOf
  coe_injective' | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩, rfl => rfl


@[simp] theorem measureOf_eq_coe (m : OuterMeasure α) : m.measureOf = m := rfl


instance : OuterMeasureClass (OuterMeasure α) α where
  measure_empty f := f.empty
  measure_mono f := f.mono
  measure_iUnion_nat_le f := f.iUnion_nat


