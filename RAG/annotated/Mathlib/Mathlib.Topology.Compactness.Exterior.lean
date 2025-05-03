theorem IsCompact.exterior_iff : IsCompact (exterior s) ↔ IsCompact s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsCompact (exterior s)) (IsCompact s)
  -/
  simp only [isCompact_iff_finite_subcover]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (∀ {ι : Type u_1} (U : ι → Set X), (∀ (i : ι), IsOpen (U i)) → HasSubset …
  -/
  peel with ι U hUo
  simp only [(isOpen_iUnion hUo).exterior_subset,
    (isOpen_iUnion fun i ↦ isOpen_iUnion fun _ ↦ hUo i).exterior_subset]


protected alias ⟨IsCompact.of_exterior, IsCompact.exterior⟩ := IsCompact.exterior_iff


@[deprecated IsCompact.exterior (since := "2024-09-18")]
lemma Set.Finite.isCompact_exterior (hs : s.Finite) : IsCompact (exterior s) :=
  hs.isCompact.exterior

