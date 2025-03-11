/-- An additive content is a set function with value 0 at the empty set which is finitely additive
on a given set of sets. -/
structure AddContent (C : Set (Set α)) where
  /-- The value of the content on a set. -/
  toFun : Set α → ℝ≥0∞
  empty' : toFun ∅ = 0
  sUnion' (I : Finset (Set α)) (_h_ss : ↑I ⊆ C)
      (_h_dis : PairwiseDisjoint (I : Set (Set α)) id) (_h_mem : ⋃₀ ↑I ∈ C) :
    toFun (⋃₀ I) = ∑ u ∈ I, toFun u


instance : Inhabited (AddContent C) :=
  ⟨{toFun := fun _ => 0
                 /-
                   α : Type u_1
                   C : Set (Set α)
                   s t : Set α
                   I : Finset (Set α)
                   ⊢ Eq ((fun x => 0) EmptyCollection.emptyCollection) 0
                 -/
    empty' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    α : Type u_1
                    C : Set (Set α)
                    s t : Set α
                    I : Finset (Set α)
                    ⊢ ∀ (I : Finset (Set α)), HasSubset.Subset (↑I) C → (↑I).PairwiseDisjoint id → …
                  -/
    sUnion' := by simp }⟩
                  /-
                    🎉 no goals
                  -/


instance : DFunLike (AddContent C) (Set α) (fun _ ↦ ℝ≥0∞) where
  coe m s := m.toFun s
  coe_injective' m m' _ := by
    /-
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      I : Finset (Set α)
      m m' : MeasureTheory.AddContent C
      x✝ : Eq ((fun m s => m.toFun s) m) ((fun m s => m.toFun s) m')
      ⊢ Eq m m'
    -/
    cases m
    /-
      case mk
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      I : Finset (Set α)
      m' : MeasureTheory.AddContent C
      toFun✝ : Set α → ENNReal
      empty'✝ : Eq (toFun✝ EmptyCollection.emptyCollection) 0
      sUnion'✝ : ∀ (I : Finset (Set α)), HasSubset.Subset (↑I) C → (↑I).PairwiseDisj …
      x✝ : Eq ((fun m s => m.toFun s) { toFun := toFun✝, empty' := empty'✝, sUnion'  …
      ⊢ Eq { toFun := toFun✝, empty' := empty'✝, sUnion' := sUnion'✝ } m'
    -/
    cases m'
    /-
      case mk.mk
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      I : Finset (Set α)
      toFun✝¹ : Set α → ENNReal
      empty'✝¹ : Eq (toFun✝¹ EmptyCollection.emptyCollection) 0
      sUnion'✝¹ : ∀ (I : Finset (Set α)), HasSubset.Subset (↑I) C → (↑I).PairwiseDis …
      toFun✝ : Set α → ENNReal
      empty'✝ : Eq (toFun✝ EmptyCollection.emptyCollection) 0
      sUnion'✝ : ∀ (I : Finset (Set α)), HasSubset.Subset (↑I) C → (↑I).PairwiseDisj …
      x✝ : Eq ((fun m s => m.toFun s) { toFun := toFun✝¹, empty' := empty'✝¹, sUnion …
      ⊢ Eq { toFun := toFun✝¹, empty' := empty'✝¹, sUnion' := sUnion'✝¹ } { toFun := …
    -/
    congr
    /-
      🎉 no goals
    -/


@[ext] protected lemma AddContent.ext (h : ∀ s, m s = m' s) : m = m' :=
  DFunLike.ext _ _ h


@[simp] lemma addContent_empty : m ∅ = 0 := m.empty'


lemma addContent_sUnion (h_ss : ↑I ⊆ C)
    (h_dis : PairwiseDisjoint (I : Set (Set α)) id) (h_mem : ⋃₀ ↑I ∈ C) :
    m (⋃₀ I) = ∑ u ∈ I, m u :=
  m.sUnion' I h_ss h_dis h_mem


lemma addContent_union' (hs : s ∈ C) (ht : t ∈ C) (hst : s ∪ t ∈ C) (h_dis : Disjoint s t) :
    m (s ∪ t) = m s + m t := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    m : MeasureTheory.AddContent C
    hs : Membership.mem C s
    ht : Membership.mem C t
    hst : Membership.mem C (Union.union s t)
    h_dis : Disjoint s t
    ⊢ Eq (m (Union.union s t)) (HAdd.hAdd (m s) (m t))
  -/
  by_cases hs_empty : s = ∅
    /-
      case pos
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      m : MeasureTheory.AddContent C
      hs : Membership.mem C s
      ht : Membership.mem C t
      hst : Membership.mem C (Union.union s t)
      h_dis : Disjoint s t
      hs_empty : Eq s EmptyCollection.emptyCollection
      ⊢ Eq (m (Union.union s t)) (HAdd.hAdd (m s) (m t))
    -/
  · simp only [hs_empty, Set.empty_union, addContent_empty, zero_add]
    /-
      🎉 no goals
    -/
  classical
  have h := addContent_sUnion (m := m) (I := {s, t}) ?_ ?_ ?_
  rotate_left
  · simp only [coe_pair, Set.insert_subset_iff, hs, ht, Set.singleton_subset_iff, and_self_iff]
  · simp only [coe_pair, Set.pairwiseDisjoint_insert, pairwiseDisjoint_singleton,
      mem_singleton_iff, Ne, id, forall_eq, true_and]
    exact fun _ => h_dis
  · simp only [coe_pair, sUnion_insert, sUnion_singleton]
    exact hst
  convert h
  · simp only [coe_pair, sUnion_insert, sUnion_singleton]
  · rw [sum_insert, sum_singleton]
    simp only [Finset.mem_singleton]
    refine fun hs_eq_t => hs_empty ?_
    rw [← hs_eq_t] at h_dis
    exact Disjoint.eq_bot_of_self h_dis


lemma addContent_eq_add_diffFinset₀_of_subset (hC : IsSetSemiring C)
    (hs : s ∈ C) (hI : ↑I ⊆ C) (hI_ss : ∀ t ∈ I, t ⊆ s)
    (h_dis : PairwiseDisjoint (I : Set (Set α)) id) :
    m s = ∑ i ∈ I, m i + ∑ i ∈ hC.diffFinset₀ hs hI, m i := by
  classical
  conv_lhs => rw [← hC.sUnion_union_diffFinset₀_of_subset hs hI hI_ss]
  rw [addContent_sUnion]
  · rw [sum_union]
    exact hC.disjoint_diffFinset₀ hs hI
  · rw [coe_union]
    exact Set.union_subset hI (hC.diffFinset₀_subset hs hI)
  · rw [coe_union]
    exact hC.pairwiseDisjoint_union_diffFinset₀ hs hI h_dis
  · rwa [hC.sUnion_union_diffFinset₀_of_subset hs hI hI_ss]


lemma sum_addContent_le_of_subset (hC : IsSetSemiring C)
    (h_ss : ↑I ⊆ C) (h_dis : PairwiseDisjoint (I : Set (Set α)) id)
    (ht : t ∈ C) (hJt : ∀ s ∈ I, s ⊆ t) :
    ∑ u ∈ I, m u ≤ m t := by
  classical
  rw [addContent_eq_add_diffFinset₀_of_subset hC ht h_ss hJt h_dis]
  exact le_add_right le_rfl


lemma addContent_mono (hC : IsSetSemiring C) (hs : s ∈ C) (ht : t ∈ C)
    (hst : s ⊆ t) :
    m s ≤ m t := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    m : MeasureTheory.AddContent C
    hC : MeasureTheory.IsSetSemiring C
    hs : Membership.mem C s
    ht : Membership.mem C t
    hst : HasSubset.Subset s t
    ⊢ LE.le (m s) (m t)
  -/
  have h := sum_addContent_le_of_subset (m := m) hC (I := {s}) ?_ ?_ ht ?_
    /-
      case refine_4
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      m : MeasureTheory.AddContent C
      hC : MeasureTheory.IsSetSemiring C
      hs : Membership.mem C s
      ht : Membership.mem C t
      hst : HasSubset.Subset s t
      h : LE.le ((Singleton.singleton s).sum fun u => m u) (m t)
      ⊢ LE.le (m s) (m t)
    -/
  · simpa only [sum_singleton] using h
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      m : MeasureTheory.AddContent C
      hC : MeasureTheory.IsSetSemiring C
      hs : Membership.mem C s
      ht : Membership.mem C t
      hst : HasSubset.Subset s t
      ⊢ HasSubset.Subset (↑(Singleton.singleton s)) C
    -/
  · rwa [singleton_subset_set_iff]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      m : MeasureTheory.AddContent C
      hC : MeasureTheory.IsSetSemiring C
      hs : Membership.mem C s
      ht : Membership.mem C t
      hst : HasSubset.Subset s t
      ⊢ (↑(Singleton.singleton s)).PairwiseDisjoint id
    -/
  · simp only [coe_singleton, pairwiseDisjoint_singleton]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      m : MeasureTheory.AddContent C
      hC : MeasureTheory.IsSetSemiring C
      hs : Membership.mem C s
      ht : Membership.mem C t
      hst : HasSubset.Subset s t
      ⊢ ∀ (s_1 : Set α), Membership.mem (Singleton.singleton s) s_1 → HasSubset.Subs …
    -/
  · simp [hst]
    /-
      🎉 no goals
    -/


lemma addContent_union (hC : IsSetRing C) (hs : s ∈ C) (ht : t ∈ C)
    (h_dis : Disjoint s t) :
    m (s ∪ t) = m s + m t :=
  addContent_union' hs ht (hC.union_mem hs ht) h_dis


lemma addContent_union_le (hC : IsSetRing C) (hs : s ∈ C) (ht : t ∈ C) :
    m (s ∪ t) ≤ m s + m t := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    m : MeasureTheory.AddContent C
    hC : MeasureTheory.IsSetRing C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ LE.le (m (Union.union s t)) (HAdd.hAdd (m s) (m t))
  -/
  rw [← union_diff_self, addContent_union hC hs (hC.diff_mem ht hs)]
  · exact add_le_add le_rfl
      (addContent_mono hC.isSetSemiring (hC.diff_mem ht hs) ht diff_subset)
    /-
      α : Type u_1
      C : Set (Set α)
      s t : Set α
      m : MeasureTheory.AddContent C
      hC : MeasureTheory.IsSetRing C
      hs : Membership.mem C s
      ht : Membership.mem C t
      ⊢ Disjoint s (SDiff.sdiff t s)
    -/
  · rw [Set.disjoint_iff_inter_eq_empty, inter_diff_self]
    /-
      🎉 no goals
    -/


lemma addContent_biUnion_le {ι : Type*} (hC : IsSetRing C) {s : ι → Set α}
    {S : Finset ι} (hs : ∀ n ∈ S, s n ∈ C) :
    m (⋃ i ∈ S, s i) ≤ ∑ i ∈ S, m (s i) := by
  classical
  induction' S using Finset.induction with i S hiS h hs
  · simp
  · rw [Finset.sum_insert hiS]
    simp_rw [← Finset.mem_coe, Finset.coe_insert, Set.biUnion_insert]
    simp only [Finset.mem_insert, forall_eq_or_imp] at hs
    refine (addContent_union_le hC hs.1 (hC.biUnion_mem S hs.2)).trans ?_
    exact add_le_add le_rfl (h hs.2)


lemma le_addContent_diff (m : AddContent C) (hC : IsSetRing C) (hs : s ∈ C) (ht : t ∈ C) :
    m s - m t ≤ m (s \ t) := by
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    m : MeasureTheory.AddContent C
    hC : MeasureTheory.IsSetRing C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ LE.le (HSub.hSub (m s) (m t)) (m (SDiff.sdiff s t))
  -/
  conv_lhs => rw [← inter_union_diff s t]
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    m : MeasureTheory.AddContent C
    hC : MeasureTheory.IsSetRing C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ LE.le (HSub.hSub (m (Union.union (Inter.inter s t) (SDiff.sdiff s t))) (m t) …
  -/
  rw [addContent_union hC (hC.inter_mem hs ht) (hC.diff_mem hs ht) disjoint_inf_sdiff, add_comm]
  /-
    α : Type u_1
    C : Set (Set α)
    s t : Set α
    m : MeasureTheory.AddContent C
    hC : MeasureTheory.IsSetRing C
    hs : Membership.mem C s
    ht : Membership.mem C t
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (m (SDiff.sdiff s t)) (m (Inter.inter s t))) (m  …
  -/
  refine add_tsub_le_assoc.trans_eq ?_
  rw [tsub_eq_zero_of_le
    (addContent_mono hC.isSetSemiring (hC.inter_mem hs ht) ht inter_subset_right), add_zero]


