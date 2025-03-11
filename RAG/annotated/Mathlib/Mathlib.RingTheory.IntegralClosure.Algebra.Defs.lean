/-- An algebra is integral if every element of the extension is integral over the base ring. -/
protected class Algebra.IsIntegral : Prop where
  isIntegral : ∀ x : A, IsIntegral R x


lemma Algebra.isIntegral_def : Algebra.IsIntegral R A ↔ ∀ x : A, IsIntegral R x :=
  ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩


