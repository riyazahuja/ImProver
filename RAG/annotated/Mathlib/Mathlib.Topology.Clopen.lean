protected theorem IsClopen.isOpen (hs : IsClopen s) : IsOpen s := hs.2


protected theorem IsClopen.isClosed (hs : IsClopen s) : IsClosed s := hs.1


theorem isClopen_iff_frontier_eq_empty : IsClopen s ↔ frontier s = ∅ := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsClopen s) (Eq (frontier s) EmptyCollection.emptyCollection)
  -/
  rw [IsClopen, ← closure_eq_iff_isClosed, ← interior_eq_iff_isOpen, frontier, diff_eq_empty]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (And (Eq (closure s) s) (Eq (interior s) s)) (HasSubset.Subset (closure  …
  -/
  refine ⟨fun h => (h.1.trans h.2.symm).subset, fun h => ?_⟩
  exact ⟨(h.trans interior_subset).antisymm subset_closure,
    interior_subset.antisymm (subset_closure.trans h)⟩


@[simp] alias ⟨IsClopen.frontier_eq, _⟩ := isClopen_iff_frontier_eq_empty


theorem IsClopen.union (hs : IsClopen s) (ht : IsClopen t) : IsClopen (s ∪ t) :=
  ⟨hs.1.union ht.1, hs.2.union ht.2⟩


theorem IsClopen.inter (hs : IsClopen s) (ht : IsClopen t) : IsClopen (s ∩ t) :=
  ⟨hs.1.inter ht.1, hs.2.inter ht.2⟩


theorem isClopen_empty : IsClopen (∅ : Set X) := ⟨isClosed_empty, isOpen_empty⟩


theorem isClopen_univ : IsClopen (univ : Set X) := ⟨isClosed_univ, isOpen_univ⟩


theorem IsClopen.compl (hs : IsClopen s) : IsClopen sᶜ :=
  ⟨hs.2.isClosed_compl, hs.1.isOpen_compl⟩


@[simp]
theorem isClopen_compl_iff : IsClopen sᶜ ↔ IsClopen s :=
  ⟨fun h => compl_compl s ▸ IsClopen.compl h, IsClopen.compl⟩


theorem IsClopen.diff (hs : IsClopen s) (ht : IsClopen t) : IsClopen (s \ t) :=
  hs.inter ht.compl


lemma IsClopen.himp (hs : IsClopen s) (ht : IsClopen t) : IsClopen (s ⇨ t) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsClopen s
    ht : IsClopen t
    ⊢ IsClopen (HImp.himp s t)
  -/
  simpa [himp_eq] using ht.union hs.compl
  /-
    🎉 no goals
  -/


theorem IsClopen.prod {t : Set Y} (hs : IsClopen s) (ht : IsClopen t) : IsClopen (s ×ˢ t) :=
  ⟨hs.1.prod ht.1, hs.2.prod ht.2⟩


theorem isClopen_iUnion_of_finite {Y} [Finite Y] {s : Y → Set X} (h : ∀ i, IsClopen (s i)) :
    IsClopen (⋃ i, s i) :=
  ⟨isClosed_iUnion_of_finite (forall_and.1 h).1, isOpen_iUnion (forall_and.1 h).2⟩


theorem Set.Finite.isClopen_biUnion {Y} {s : Set Y} {f : Y → Set X} (hs : s.Finite)
    (h : ∀ i ∈ s, IsClopen <| f i) : IsClopen (⋃ i ∈ s, f i) :=
  ⟨hs.isClosed_biUnion fun i hi => (h i hi).1, isOpen_biUnion fun i hi => (h i hi).2⟩


theorem isClopen_biUnion_finset {Y} {s : Finset Y} {f : Y → Set X}
    (h : ∀ i ∈ s, IsClopen <| f i) : IsClopen (⋃ i ∈ s, f i) :=
 s.finite_toSet.isClopen_biUnion h


theorem isClopen_iInter_of_finite {Y} [Finite Y] {s : Y → Set X} (h : ∀ i, IsClopen (s i)) :
    IsClopen (⋂ i, s i) :=
  ⟨isClosed_iInter (forall_and.1 h).1, isOpen_iInter_of_finite (forall_and.1 h).2⟩


theorem Set.Finite.isClopen_biInter {Y} {s : Set Y} (hs : s.Finite) {f : Y → Set X}
    (h : ∀ i ∈ s, IsClopen (f i)) : IsClopen (⋂ i ∈ s, f i) :=
  ⟨isClosed_biInter fun i hi => (h i hi).1, hs.isOpen_biInter fun i hi => (h i hi).2⟩


theorem isClopen_biInter_finset {Y} {s : Finset Y} {f : Y → Set X}
    (h : ∀ i ∈ s, IsClopen (f i)) : IsClopen (⋂ i ∈ s, f i) :=
  s.finite_toSet.isClopen_biInter h


theorem IsClopen.preimage {s : Set Y} (h : IsClopen s) {f : X → Y} (hf : Continuous f) :
    IsClopen (f ⁻¹' s) :=
  ⟨h.1.preimage hf, h.2.preimage hf⟩


theorem ContinuousOn.preimage_isClopen_of_isClopen {f : X → Y} {s : Set X} {t : Set Y}
    (hf : ContinuousOn f s) (hs : IsClopen s) (ht : IsClopen t) : IsClopen (s ∩ f ⁻¹' t) :=
  ⟨ContinuousOn.preimage_isClosed_of_isClosed hf hs.1 ht.1,
    ContinuousOn.isOpen_inter_preimage hf hs.2 ht.2⟩


/-- The intersection of a disjoint covering by two open sets of a clopen set will be clopen. -/
theorem isClopen_inter_of_disjoint_cover_clopen {s a b : Set X} (h : IsClopen s) (cover : s ⊆ a ∪ b)
    (ha : IsOpen a) (hb : IsOpen b) (hab : Disjoint a b) : IsClopen (s ∩ a) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s a b : Set X
    h : IsClopen s
    cover : HasSubset.Subset s (Union.union a b)
    ha : IsOpen a
    hb : IsOpen b
    hab : Disjoint a b
    ⊢ IsClopen (Inter.inter s a)
  -/
  refine ⟨?_, IsOpen.inter h.2 ha⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s a b : Set X
    h : IsClopen s
    cover : HasSubset.Subset s (Union.union a b)
    ha : IsOpen a
    hb : IsOpen b
    hab : Disjoint a b
    ⊢ IsClosed (Inter.inter s a)
  -/
  have : IsClosed (s ∩ bᶜ) := IsClosed.inter h.1 (isClosed_compl_iff.2 hb)
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s a b : Set X
    h : IsClopen s
    cover : HasSubset.Subset s (Union.union a b)
    ha : IsOpen a
    hb : IsOpen b
    hab : Disjoint a b
    this : IsClosed (Inter.inter s (HasCompl.compl b))
    ⊢ IsClosed (Inter.inter s a)
  -/
  convert this using 1
  /-
    case h.e'_3
    X : Type u
    inst✝ : TopologicalSpace X
    s a b : Set X
    h : IsClopen s
    cover : HasSubset.Subset s (Union.union a b)
    ha : IsOpen a
    hb : IsOpen b
    hab : Disjoint a b
    this : IsClosed (Inter.inter s (HasCompl.compl b))
    ⊢ Eq (Inter.inter s a) (Inter.inter s (HasCompl.compl b))
  -/
  refine (inter_subset_inter_right s hab.subset_compl_right).antisymm ?_
  /-
    case h.e'_3
    X : Type u
    inst✝ : TopologicalSpace X
    s a b : Set X
    h : IsClopen s
    cover : HasSubset.Subset s (Union.union a b)
    ha : IsOpen a
    hb : IsOpen b
    hab : Disjoint a b
    this : IsClosed (Inter.inter s (HasCompl.compl b))
    ⊢ HasSubset.Subset (Inter.inter s (HasCompl.compl b)) (Inter.inter s a)
  -/
  rintro x ⟨hx₁, hx₂⟩
  /-
    case h.e'_3.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s a b : Set X
    h : IsClopen s
    cover : HasSubset.Subset s (Union.union a b)
    ha : IsOpen a
    hb : IsOpen b
    hab : Disjoint a b
    this : IsClosed (Inter.inter s (HasCompl.compl b))
    x : X
    hx₁ : Membership.mem s x
    hx₂ : Membership.mem (HasCompl.compl b) x
    ⊢ Membership.mem (Inter.inter s a) x
  -/
  exact ⟨hx₁, by simpa [not_mem_of_mem_compl hx₂] using cover hx₁⟩
  /-
    🎉 no goals
  -/


theorem isClopen_of_disjoint_cover_open {a b : Set X} (cover : univ ⊆ a ∪ b)
    (ha : IsOpen a) (hb : IsOpen b) (hab : Disjoint a b) : IsClopen a :=
  univ_inter a ▸ isClopen_inter_of_disjoint_cover_clopen isClopen_univ cover ha hb hab


@[simp]
theorem isClopen_discrete [DiscreteTopology X] (s : Set X) : IsClopen s :=
  ⟨isClosed_discrete _, isOpen_discrete _⟩


theorem isClopen_range_inl : IsClopen (range (Sum.inl : X → X ⊕ Y)) :=
  ⟨isClosed_range_inl, isOpen_range_inl⟩


theorem isClopen_range_inr : IsClopen (range (Sum.inr : Y → X ⊕ Y)) :=
  ⟨isClosed_range_inr, isOpen_range_inr⟩


theorem isClopen_range_sigmaMk {X : ι → Type*} [∀ i, TopologicalSpace (X i)] {i : ι} :
    IsClopen (Set.range (@Sigma.mk ι X i)) :=
  ⟨IsClosedEmbedding.sigmaMk.isClosed_range, IsOpenEmbedding.sigmaMk.isOpen_range⟩


protected theorem Topology.IsQuotientMap.isClopen_preimage {f : X → Y} (hf : IsQuotientMap f)
    {s : Set Y} : IsClopen (f ⁻¹' s) ↔ IsClopen s :=
  and_congr hf.isClosed_preimage hf.isOpen_preimage


@[deprecated (since := "2024-10-22")]
alias QuotientMap.isClopen_preimage := IsQuotientMap.isClopen_preimage


theorem continuous_boolIndicator_iff_isClopen (U : Set X) :
    Continuous U.boolIndicator ↔ IsClopen U := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    U : Set X
    ⊢ Iff (Continuous U.boolIndicator) (IsClopen U)
  -/
  rw [continuous_bool_rng true, preimage_boolIndicator_true]
  /-
    🎉 no goals
  -/


theorem continuousOn_boolIndicator_iff_isClopen (s U : Set X) :
    ContinuousOn U.boolIndicator s ↔ IsClopen (((↑) : s → X) ⁻¹' U) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s U : Set X
    ⊢ Iff (ContinuousOn U.boolIndicator s) (IsClopen (Set.preimage Subtype.val U))
  -/
  rw [continuousOn_iff_continuous_restrict, ← continuous_boolIndicator_iff_isClopen]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s U : Set X
    ⊢ Iff (Continuous (s.restrict U.boolIndicator)) (Continuous (Set.preimage Subt …
  -/
  rfl
  /-
    🎉 no goals
  -/


