/-- A semi-ring of sets `C` is a family of sets containing `∅`, stable by intersection and such that
for all `s, t ∈ C`, `s \ t` is equal to a disjoint union of finitely many sets in `C`. -/
structure IsSetSemiring (C : Set (Set α)) : Prop where
  empty_mem : ∅ ∈ C
  inter_mem : ∀ s ∈ C, ∀ t ∈ C, s ∩ t ∈ C
  diff_eq_sUnion' : ∀ s ∈ C, ∀ t ∈ C,
    ∃ I : Finset (Set α), ↑I ⊆ C ∧ PairwiseDisjoint (I : Set (Set α)) id ∧ s \ t = ⋃₀ I


lemma isPiSystem (hC : IsSetSemiring C) : IsPiSystem C := fun s hs t ht _ ↦ hC.inter_mem s hs t ht


open scoped Classical in
/-- In a semi-ring of sets `C`, for all sets `s, t ∈ C`, `s \ t` is equal to a disjoint union of
finitely many sets in `C`. The finite set of sets in the union is not unique, but this definition
gives an arbitrary `Finset (Set α)` that satisfies the equality.

We remove the empty set to ensure that `t ∉ hC.diffFinset hs ht` even if `t = ∅`. -/
noncomputable def diffFinset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) :
    Finset (Set α) :=
  (hC.diff_eq_sUnion' s hs t ht).choose \ {∅}


lemma empty_not_mem_diffFinset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) :
    ∅ ∉ hC.diffFinset hs ht := by
  classical
  simp only [diffFinset, mem_sdiff, Finset.mem_singleton, eq_self_iff_true, not_true,
    and_false, not_false_iff]


lemma diffFinset_subset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) :
    ↑(hC.diffFinset hs ht) ⊆ C := by
  classical
  simp only [diffFinset, coe_sdiff, coe_singleton, diff_singleton_subset_iff]
  exact (hC.diff_eq_sUnion' s hs t ht).choose_spec.1.trans (Set.subset_insert _ _)


lemma pairwiseDisjoint_diffFinset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) :
    PairwiseDisjoint (hC.diffFinset hs ht : Set (Set α)) id := by
  classical
  simp only [diffFinset, coe_sdiff, coe_singleton]
  exact Set.PairwiseDisjoint.subset (hC.diff_eq_sUnion' s hs t ht).choose_spec.2.1
      diff_subset


lemma sUnion_diffFinset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) :
    ⋃₀ hC.diffFinset hs ht = s \ t := by
  classical
  rw [(hC.diff_eq_sUnion' s hs t ht).choose_spec.2.2]
  simp only [diffFinset, coe_sdiff, coe_singleton, diff_singleton_subset_iff]
  rw [sUnion_diff_singleton_empty]


lemma not_mem_diffFinset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) :
    t ∉ hC.diffFinset hs ht := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ Not (Membership.mem (hC.diffFinset hs ht) t)
  -/
  intro hs_mem
  suffices t ⊆ s \ t by
    have h := @disjoint_sdiff_self_right _ t s _
    specialize h le_rfl this
    simp only [Set.bot_eq_empty, Set.le_eq_subset, subset_empty_iff] at h
    refine hC.empty_not_mem_diffFinset hs ht ?_
    rwa [← h]
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    hs_mem : Membership.mem (hC.diffFinset hs ht) t
    ⊢ HasSubset.Subset t (SDiff.sdiff s t)
  -/
  rw [← hC.sUnion_diffFinset hs ht]
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    hs_mem : Membership.mem (hC.diffFinset hs ht) t
    ⊢ HasSubset.Subset t (↑(hC.diffFinset hs ht)).sUnion
  -/
  exact subset_sUnion_of_mem hs_mem
  /-
    🎉 no goals
  -/


lemma sUnion_insert_diffFinset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) (hst : t ⊆ s) :
    ⋃₀ insert t (hC.diffFinset hs ht) = s := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    hst : HasSubset.Subset t s
    ⊢ Eq (Insert.insert t ↑(hC.diffFinset hs ht)).sUnion s
  -/
  conv_rhs => rw [← union_diff_cancel hst, ← hC.sUnion_diffFinset hs ht]
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    hst : HasSubset.Subset t s
    ⊢ Eq (Insert.insert t ↑(hC.diffFinset hs ht)).sUnion (Union.union t (↑(hC.diff …
  -/
  simp only [mem_coe, sUnion_insert]
  /-
    🎉 no goals
  -/


lemma disjoint_sUnion_diffFinset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) :
    Disjoint t (⋃₀ hC.diffFinset hs ht) := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ Disjoint t (↑(hC.diffFinset hs ht)).sUnion
  -/
  rw [hC.sUnion_diffFinset]
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ Disjoint t (SDiff.sdiff s t)
  -/
  exact disjoint_sdiff_right
  /-
    🎉 no goals
  -/


lemma pairwiseDisjoint_insert_diffFinset (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C) :
    PairwiseDisjoint (insert t (hC.diffFinset hs ht) : Set (Set α)) id := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ (Insert.insert t ↑(hC.diffFinset hs ht)).PairwiseDisjoint id
  -/
  have h := hC.pairwiseDisjoint_diffFinset hs ht
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    h : (↑(hC.diffFinset hs ht)).PairwiseDisjoint id
    ⊢ (Insert.insert t ↑(hC.diffFinset hs ht)).PairwiseDisjoint id
  -/
  refine PairwiseDisjoint.insert_of_not_mem h (hC.not_mem_diffFinset hs ht) fun u hu ↦ ?_
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    h : (↑(hC.diffFinset hs ht)).PairwiseDisjoint id
    u : Set α
    hu : Membership.mem (↑(hC.diffFinset hs ht)) u
    ⊢ Disjoint (id t) (id u)
  -/
  simp_rw [id]
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    h : (↑(hC.diffFinset hs ht)).PairwiseDisjoint id
    u : Set α
    hu : Membership.mem (↑(hC.diffFinset hs ht)) u
    ⊢ Disjoint t u
  -/
  refine Disjoint.mono_right ?_ (hC.disjoint_sUnion_diffFinset hs ht)
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    h : (↑(hC.diffFinset hs ht)).PairwiseDisjoint id
    u : Set α
    hu : Membership.mem (↑(hC.diffFinset hs ht)) u
    ⊢ LE.le u (↑(hC.diffFinset hs ht)).sUnion
  -/
  simp only [Set.le_eq_subset]
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    h : (↑(hC.diffFinset hs ht)).PairwiseDisjoint id
    u : Set α
    hu : Membership.mem (↑(hC.diffFinset hs ht)) u
    ⊢ HasSubset.Subset u (↑(hC.diffFinset hs ht)).sUnion
  -/
  exact subset_sUnion_of_mem hu
  /-
    🎉 no goals
  -/


/-- In a semiring of sets `C`, for all set `s ∈ C` and finite set of sets `I ⊆ C`, there is a
finite set of sets in `C` whose union is `s \ ⋃₀ I`.
See `IsSetSemiring.diffFinset₀` for a definition that gives such a set. -/
lemma exists_disjoint_finset_diff_eq (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) :
    ∃ J : Finset (Set α), ↑J ⊆ C ∧ PairwiseDisjoint (J : Set (Set α)) id ∧
      s \ ⋃₀ I = ⋃₀ J := by
  classical
  induction I using Finset.induction with
  | empty =>
    simp only [coe_empty, sUnion_empty, diff_empty, exists_prop]
    refine ⟨{s}, singleton_subset_set_iff.mpr hs, ?_⟩
    simp only [coe_singleton, pairwiseDisjoint_singleton, sUnion_singleton, eq_self_iff_true,
      and_self_iff]
  | @insert t I' _ h => ?_

  rw [coe_insert] at hI
  have ht : t ∈ C := hI (Set.mem_insert _ _)
  obtain ⟨J, h_ss, h_dis, h_eq⟩ := h ((Set.subset_insert _ _).trans hI)
  let Ju : ∀ u ∈ C, Finset (Set α) := fun u hu ↦ hC.diffFinset hu ht
  have hJu_subset : ∀ (u) (hu : u ∈ C), ↑(Ju u hu) ⊆ C := by
    intro u hu x hx
    exact hC.diffFinset_subset hu ht hx
  have hJu_disj : ∀ (u) (hu : u ∈ C), (Ju u hu : Set (Set α)).PairwiseDisjoint id := fun u hu ↦
    hC.pairwiseDisjoint_diffFinset hu ht
  have hJu_sUnion : ∀ (u) (hu : u ∈ C), ⋃₀ (Ju u hu : Set (Set α)) = u \ t :=
    fun u hu ↦ hC.sUnion_diffFinset hu ht
  have hJu_disj' : ∀ (u) (hu : u ∈ C) (v) (hv : v ∈ C) (_h_dis : Disjoint u v),
      Disjoint (⋃₀ (Ju u hu : Set (Set α))) (⋃₀ ↑(Ju v hv)) := by
    intro u hu v hv huv_disj
    rw [hJu_sUnion, hJu_sUnion]
    exact disjoint_of_subset Set.diff_subset Set.diff_subset huv_disj
  let J' : Finset (Set α) := Finset.biUnion (Finset.univ : Finset J) fun u ↦ Ju u (h_ss u.prop)
  have hJ'_subset : ↑J' ⊆ C := by
    intro u
    simp only [J' ,Subtype.coe_mk, univ_eq_attach, coe_biUnion, mem_coe, mem_attach, iUnion_true,
      mem_iUnion, Finset.exists_coe, exists₂_imp]
    intro v hv huvt
    exact hJu_subset v (h_ss hv) huvt
  refine ⟨J', hJ'_subset, ?_, ?_⟩
  · rw [Finset.coe_biUnion]
    refine PairwiseDisjoint.biUnion ?_ ?_
    · simp only [univ_eq_attach, mem_coe, id, iSup_eq_iUnion]
      simp_rw [PairwiseDisjoint, Set.Pairwise]
      intro x _ y _ hxy
      have hxy_disj : Disjoint (x : Set α) y := by
        by_contra h_contra
        refine hxy ?_
        refine Subtype.ext ?_
        exact h_dis.elim x.prop y.prop h_contra
      convert hJu_disj' (x : Set α) (h_ss x.prop) y (h_ss y.prop) hxy_disj
      · rw [sUnion_eq_biUnion]
        congr
      · rw [sUnion_eq_biUnion]
        congr
    · exact fun u _ ↦ hJu_disj _ _
  · rw [coe_insert, sUnion_insert, Set.union_comm, ← Set.diff_diff, h_eq]
    simp_rw [J', sUnion_eq_biUnion, Set.iUnion_diff]
    simp only [Subtype.coe_mk, mem_coe, Finset.mem_biUnion, Finset.mem_univ, exists_true_left,
      Finset.exists_coe, iUnion_exists, true_and]
    rw [iUnion_comm]
    refine iUnion_congr fun i ↦ ?_
    by_cases hi : i ∈ J
    · simp only [hi, iUnion_true, exists_prop]
      rw [← hJu_sUnion i (h_ss hi), sUnion_eq_biUnion]
      simp only [mem_coe]
    · simp only [hi, iUnion_of_empty, iUnion_empty]


open scoped Classical in
/-- In a semiring of sets `C`, for all set `s ∈ C` and finite set of sets `I ⊆ C`,
`diffFinset₀` is a finite set of sets in `C` such that `s \ ⋃₀ I = ⋃₀ (hC.diffFinset₀ hs I hI)`.
`diffFinset` is a special case of `diffFinset₀` where `I` is a singleton. -/
noncomputable def diffFinset₀ (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) : Finset (Set α) :=
  (hC.exists_disjoint_finset_diff_eq hs hI).choose \ {∅}


lemma empty_not_mem_diffFinset₀ (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) :
    ∅ ∉ hC.diffFinset₀ hs hI := by
  classical
  simp only [diffFinset₀, mem_sdiff, Finset.mem_singleton, eq_self_iff_true, not_true,
    and_false, not_false_iff]


lemma diffFinset₀_subset (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) :
    ↑(hC.diffFinset₀ hs hI) ⊆ C := by
  classical
  simp only [diffFinset₀, coe_sdiff, coe_singleton, diff_singleton_subset_iff]
  exact (hC.exists_disjoint_finset_diff_eq hs hI).choose_spec.1.trans (Set.subset_insert _ _)


lemma pairwiseDisjoint_diffFinset₀ (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) :
    PairwiseDisjoint (hC.diffFinset₀ hs hI : Set (Set α)) id := by
  classical
  simp only [diffFinset₀, coe_sdiff, coe_singleton]
  exact Set.PairwiseDisjoint.subset
    (hC.exists_disjoint_finset_diff_eq hs hI).choose_spec.2.1 diff_subset


lemma diff_sUnion_eq_sUnion_diffFinset₀ (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) :
    s \ ⋃₀ I = ⋃₀ hC.diffFinset₀ hs hI := by
  classical
  rw [(hC.exists_disjoint_finset_diff_eq hs hI).choose_spec.2.2]
  simp only [diffFinset₀, coe_sdiff, coe_singleton, diff_singleton_subset_iff]
  rw [sUnion_diff_singleton_empty]


lemma sUnion_diffFinset₀_subset (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) :
    ⋃₀ (hC.diffFinset₀ hs hI : Set (Set α)) ⊆ s := by
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    ⊢ HasSubset.Subset (↑(hC.diffFinset₀ hs hI)).sUnion s
  -/
  rw [← hC.diff_sUnion_eq_sUnion_diffFinset₀]
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    ⊢ HasSubset.Subset (SDiff.sdiff s (↑I).sUnion) s
  -/
  exact diff_subset
  /-
    🎉 no goals
  -/


lemma disjoint_sUnion_diffFinset₀ (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) :
    Disjoint (⋃₀ (I : Set (Set α))) (⋃₀ hC.diffFinset₀ hs hI) := by
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    ⊢ Disjoint (↑I).sUnion (↑(hC.diffFinset₀ hs hI)).sUnion
  -/
  rw [← hC.diff_sUnion_eq_sUnion_diffFinset₀]; exact Set.disjoint_sdiff_right
                                               /-
                                                 🎉 no goals
                                               -/


lemma disjoint_diffFinset₀ (hC : IsSetSemiring C) (hs : s ∈ C) (hI : ↑I ⊆ C) :
    Disjoint I (hC.diffFinset₀ hs hI) := by
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    ⊢ Disjoint I (hC.diffFinset₀ hs hI)
  -/
  by_contra h
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    h : Not (Disjoint I (hC.diffFinset₀ hs hI))
    ⊢ False
  -/
  rw [Finset.not_disjoint_iff] at h
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    h : Exists fun a => And (Membership.mem I a) (Membership.mem (hC.diffFinset₀ h …
    ⊢ False
  -/
  obtain ⟨u, huI, hu_diffFinset₀⟩ := h
  have h_disj : u ≤ ⊥ := hC.disjoint_sUnion_diffFinset₀ hs hI (subset_sUnion_of_mem huI)
    (subset_sUnion_of_mem hu_diffFinset₀)
  /-
    case intro.intro
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    u : Set α
    huI : Membership.mem I u
    hu_diffFinset₀ : Membership.mem (hC.diffFinset₀ hs hI) u
    h_disj : LE.le u Bot.bot
    ⊢ False
  -/
  simp only [Set.bot_eq_empty, Set.le_eq_subset, subset_empty_iff] at h_disj
  /-
    case intro.intro
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    u : Set α
    huI : Membership.mem I u
    hu_diffFinset₀ : Membership.mem (hC.diffFinset₀ hs hI) u
    h_disj : Eq u EmptyCollection.emptyCollection
    ⊢ False
  -/
  refine hC.empty_not_mem_diffFinset₀ hs hI ?_
  /-
    case intro.intro
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    u : Set α
    huI : Membership.mem I u
    hu_diffFinset₀ : Membership.mem (hC.diffFinset₀ hs hI) u
    h_disj : Eq u EmptyCollection.emptyCollection
    ⊢ Membership.mem (hC.diffFinset₀ hs hI) EmptyCollection.emptyCollection
  -/
  rwa [h_disj] at hu_diffFinset₀
  /-
    🎉 no goals
  -/


lemma pairwiseDisjoint_union_diffFinset₀ (hC : IsSetSemiring C) (hs : s ∈ C)
    (hI : ↑I ⊆ C) (h_dis : PairwiseDisjoint (I : Set (Set α)) id) :
    PairwiseDisjoint (I ∪ hC.diffFinset₀ hs hI : Set (Set α)) id := by
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    h_dis : (↑I).PairwiseDisjoint id
    ⊢ (Union.union ↑I ↑(hC.diffFinset₀ hs hI)).PairwiseDisjoint id
  -/
  rw [pairwiseDisjoint_union]
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    h_dis : (↑I).PairwiseDisjoint id
    ⊢ And ((↑I).PairwiseDisjoint id) (And ((↑(hC.diffFinset₀ hs hI)).PairwiseDisjo …
  -/
  refine ⟨h_dis, hC.pairwiseDisjoint_diffFinset₀ hs hI, fun u hu v hv _ ↦ ?_⟩
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    h_dis : (↑I).PairwiseDisjoint id
    u : Set α
    hu : Membership.mem (↑I) u
    v : Set α
    hv : Membership.mem (↑(hC.diffFinset₀ hs hI)) v
    x✝ : Ne u v
    ⊢ Disjoint (id u) (id v)
  -/
  simp_rw [id]
  exact disjoint_of_subset (subset_sUnion_of_mem hu) (subset_sUnion_of_mem hv)
    (hC.disjoint_sUnion_diffFinset₀ hs hI)


lemma sUnion_union_sUnion_diffFinset₀_of_subset (hC : IsSetSemiring C) (hs : s ∈ C)
    (hI : ↑I ⊆ C) (hI_ss : ∀ t ∈ I, t ⊆ s) :
    ⋃₀ I ∪ ⋃₀ hC.diffFinset₀ hs hI = s := by
  conv_rhs => rw [← union_diff_cancel (Set.sUnion_subset hI_ss : ⋃₀ ↑I ⊆ s),
    hC.diff_sUnion_eq_sUnion_diffFinset₀ hs hI]


lemma sUnion_union_diffFinset₀_of_subset (hC : IsSetSemiring C) (hs : s ∈ C)
    (hI : ↑I ⊆ C) (hI_ss : ∀ t ∈ I, t ⊆ s) [DecidableEq (Set α)] :
    ⋃₀ ↑(I ∪ hC.diffFinset₀ hs hI) = s := by
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    hI_ss : ∀ (t : Set α), Membership.mem I t → HasSubset.Subset t s
    inst✝ : DecidableEq (Set α)
    ⊢ Eq (↑(Union.union I (hC.diffFinset₀ hs hI))).sUnion s
  -/
  conv_rhs => rw [← sUnion_union_sUnion_diffFinset₀_of_subset hC hs hI hI_ss]
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    hI_ss : ∀ (t : Set α), Membership.mem I t → HasSubset.Subset t s
    inst✝ : DecidableEq (Set α)
    ⊢ Eq (↑(Union.union I (hC.diffFinset₀ hs hI))).sUnion (Union.union (↑I).sUnion …
  -/
  simp_rw [coe_union]
  /-
    α : Type u_1
    C : Set (Set α)
    s : Set α
    I : Finset (Set α)
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    hI : HasSubset.Subset (↑I) C
    hI_ss : ∀ (t : Set α), Membership.mem I t → HasSubset.Subset t s
    inst✝ : DecidableEq (Set α)
    ⊢ Eq (Union.union ↑I ↑(hC.diffFinset₀ hs hI)).sUnion (Union.union (↑I).sUnion  …
  -/
  rw [sUnion_union]
  /-
    🎉 no goals
  -/


/-- A ring of sets `C` is a family of sets containing `∅`, stable by union and set difference.
It is then also stable by intersection (see `IsSetRing.inter_mem`). -/
structure IsSetRing (C : Set (Set α)) : Prop where
  empty_mem : ∅ ∈ C
  union_mem ⦃s t⦄ : s ∈ C → t ∈ C → s ∪ t ∈ C
  diff_mem ⦃s t⦄ : s ∈ C → t ∈ C → s \ t ∈ C


lemma inter_mem (hC : IsSetRing C) (hs : s ∈ C) (ht : t ∈ C) : s ∩ t ∈ C := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    hC : MeasureTheory.IsSetRing C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ Membership.mem C (Inter.inter s t)
  -/
  rw [← diff_diff_right_self]; exact hC.diff_mem hs (hC.diff_mem hs ht)
                               /-
                                 🎉 no goals
                               -/


lemma isSetSemiring (hC : IsSetRing C) : IsSetSemiring C where
  empty_mem := hC.empty_mem
  inter_mem := fun _ hs _ ht => hC.inter_mem hs ht
  diff_eq_sUnion' := by
    /-
      α : Type u_1
      C : Set (Set α)
      hC : MeasureTheory.IsSetRing C
      ⊢ ∀ (s : Set α), Membership.mem C s → ∀ (t : Set α), Membership.mem C t → Exis …
    -/
    refine fun s hs t ht => ⟨{s \ t}, ?_, ?_, ?_⟩
      /-
        case refine_1
        α : Type u_1
        C : Set (Set α)
        hC : MeasureTheory.IsSetRing C
        s : Set α
        hs : Membership.mem C s
        t : Set α
        ht : Membership.mem C t
        ⊢ HasSubset.Subset (↑(Singleton.singleton (SDiff.sdiff s t))) C
      -/
    · simp only [coe_singleton, Set.singleton_subset_iff]
      /-
        case refine_1
        α : Type u_1
        C : Set (Set α)
        hC : MeasureTheory.IsSetRing C
        s : Set α
        hs : Membership.mem C s
        t : Set α
        ht : Membership.mem C t
        ⊢ Membership.mem C (SDiff.sdiff s t)
      -/
      exact hC.diff_mem hs ht
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        C : Set (Set α)
        hC : MeasureTheory.IsSetRing C
        s : Set α
        hs : Membership.mem C s
        t : Set α
        ht : Membership.mem C t
        ⊢ (↑(Singleton.singleton (SDiff.sdiff s t))).PairwiseDisjoint id
      -/
    · simp only [coe_singleton, pairwiseDisjoint_singleton]
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        α : Type u_1
        C : Set (Set α)
        hC : MeasureTheory.IsSetRing C
        s : Set α
        hs : Membership.mem C s
        t : Set α
        ht : Membership.mem C t
        ⊢ Eq (SDiff.sdiff s t) (↑(Singleton.singleton (SDiff.sdiff s t))).sUnion
      -/
    · simp only [coe_singleton, sUnion_singleton]
      /-
        🎉 no goals
      -/


lemma biUnion_mem {ι : Type*} (hC : IsSetRing C) {s : ι → Set α}
    (S : Finset ι) (hs : ∀ n ∈ S, s n ∈ C) :
    ⋃ i ∈ S, s i ∈ C := by
  classical
  induction' S using Finset.induction with i S _ h hs
  · simp [hC.empty_mem]
  · simp_rw [← Finset.mem_coe, Finset.coe_insert, Set.biUnion_insert]
    refine hC.union_mem (hs i (mem_insert_self i S)) ?_
    exact h (fun n hnS ↦ hs n (mem_insert_of_mem hnS))


lemma biInter_mem {ι : Type*} (hC : IsSetRing C) {s : ι → Set α}
    (S : Finset ι) (hS : S.Nonempty) (hs : ∀ n ∈ S, s n ∈ C) :
    ⋂ i ∈ S, s i ∈ C := by
  classical
  induction hS using Finset.Nonempty.cons_induction with
  | singleton => simpa using hs
  | cons i S hiS _ h =>
    simp_rw [← Finset.mem_coe, Finset.coe_cons, Set.biInter_insert]
    simp only [cons_eq_insert, Finset.mem_insert, forall_eq_or_imp] at hs
    refine hC.inter_mem hs.1 ?_
    exact h (fun n hnS ↦ hs.2 n hnS)


lemma partialSups_mem (hC : IsSetRing C) {s : ℕ → Set α} (hs : ∀ n, s n ∈ C) (n : ℕ) :
    partialSups s n ∈ C := by
  /-
    α : Type u_1
    C : Set (Set α)
    hC : MeasureTheory.IsSetRing C
    s : Nat → Set α
    hs : ∀ (n : Nat), Membership.mem C (s n)
    n : Nat
    ⊢ Membership.mem C ((partialSups s) n)
  -/
  rw [partialSups_eq_biUnion_range]
  /-
    α : Type u_1
    C : Set (Set α)
    hC : MeasureTheory.IsSetRing C
    s : Nat → Set α
    hs : ∀ (n : Nat), Membership.mem C (s n)
    n : Nat
    ⊢ Membership.mem C (Set.iUnion fun i => Set.iUnion fun h => s i)
  -/
  exact hC.biUnion_mem _ (fun n _ ↦ hs n)
  /-
    🎉 no goals
  -/


lemma disjointed_mem (hC : IsSetRing C) {s : ℕ → Set α} (hs : ∀ n, s n ∈ C) (n : ℕ) :
    disjointed s n ∈ C := by
  cases n with
  | zero => exact hs 0
  | succ n => exact hC.diff_mem (hs n.succ) (hC.partialSups_mem hs n)


