/-- There are finitely many subsets of a given finite set -/
theorem Finite.finite_subsets {α : Type u} {a : Set α} (h : a.Finite) : { b | b ⊆ a }.Finite := by
  /-
    α : Type u
    a : Set α
    h : a.Finite
    ⊢ (setOf fun b => HasSubset.Subset b a).Finite
  -/
  convert ((Finset.powerset h.toFinset).map Finset.coeEmb.1).finite_toSet
  /-
    case h.e'_2
    α : Type u
    a : Set α
    h : a.Finite
    ⊢ Eq (setOf fun b => HasSubset.Subset b a) ↑(Finset.map Finset.coeEmb.toEmbedd …
  -/
  ext s
  simpa [← @exists_finite_iff_finset α fun t => t ⊆ a ∧ t = s, Finite.subset_toFinset,
    ← and_assoc, Finset.coeEmb] using h.subset


protected theorem Finite.powerset {s : Set α} (h : s.Finite) : (𝒫 s).Finite :=
  h.finite_subsets


