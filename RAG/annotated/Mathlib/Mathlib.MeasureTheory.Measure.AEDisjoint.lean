/-- Two sets are said to be `μ`-a.e. disjoint if their intersection has measure zero. -/
def AEDisjoint (s t : Set α) :=
  μ (s ∩ t) = 0


/-- If `s : ι → Set α` is a countable family of pairwise a.e. disjoint sets, then there exists a
family of measurable null sets `t i` such that `s i \ t i` are pairwise disjoint. -/
theorem exists_null_pairwise_disjoint_diff [Countable ι] {s : ι → Set α}
    (hd : Pairwise (AEDisjoint μ on s)) : ∃ t : ι → Set α, (∀ i, MeasurableSet (t i)) ∧
    (∀ i, μ (t i) = 0) ∧ Pairwise (Disjoint on fun i => s i \ t i) := by
  refine ⟨fun i => toMeasurable μ (s i ∩ ⋃ j ∈ ({i}ᶜ : Set ι), s j), fun i =>
    measurableSet_toMeasurable _ _, fun i => ?_, ?_⟩
    /-
      case refine_1
      ι : Type u_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
      i : ι
      ⊢ Eq (μ ((fun i => MeasureTheory.toMeasurable μ (Inter.inter (s i) (Set.iUnion …
    -/
  · simp only [measure_toMeasurable, inter_iUnion]
    /-
      case refine_1
      ι : Type u_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
      i : ι
      ⊢ Eq (μ (Set.iUnion fun i_1 => Set.iUnion fun i_2 => Inter.inter (s i) (s i_1) …
    -/
    exact (measure_biUnion_null_iff <| to_countable _).2 fun j hj => hd (Ne.symm hj)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
      ⊢ Pairwise (Function.onFun Disjoint fun i => SDiff.sdiff (s i) ((fun i => Meas …
    -/
  · simp only [Pairwise, disjoint_left, onFun, mem_diff, not_and, and_imp, Classical.not_not]
    /-
      case refine_2
      ι : Type u_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
      ⊢ ∀ ⦃i j : ι⦄, Ne i j → ∀ ⦃a : α⦄, Membership.mem (s i) a → Not (Membership.me …
    -/
    intro i j hne x hi hU hj
    replace hU : x ∉ s i ∩ iUnion fun j ↦ iUnion fun _ ↦ s j :=
      fun h ↦ hU (subset_toMeasurable _ _ h)
    /-
      case refine_2
      ι : Type u_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
      i j : ι
      hne : Ne i j
      x : α
      hi : Membership.mem (s i) x
      hj : Membership.mem (s j) x
      hU : Not (Membership.mem (Inter.inter (s i) (Set.iUnion fun j => Set.iUnion fu …
      ⊢ Membership.mem (MeasureTheory.toMeasurable μ (Inter.inter (s j) (Set.iUnion  …
    -/
    simp only [mem_inter_iff, mem_iUnion, not_and, not_exists] at hU
    /-
      case refine_2
      ι : Type u_1
      α : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
      i j : ι
      hne : Ne i j
      x : α
      hi : Membership.mem (s i) x
      hj : Membership.mem (s j) x
      hU : Membership.mem (s i) x → ∀ (x_1 : ι), Membership.mem (HasCompl.compl (Sin …
      ⊢ Membership.mem (MeasureTheory.toMeasurable μ (Inter.inter (s j) (Set.iUnion  …
    -/
    exact (hU hi j hne.symm hj).elim
    /-
      🎉 no goals
    -/


protected theorem eq (h : AEDisjoint μ s t) : μ (s ∩ t) = 0 :=
  h


@[symm]
                                                                       /-
                                                                         α : Type u_2
                                                                         m : MeasurableSpace α
                                                                         μ : MeasureTheory.Measure α
                                                                         s t : Set α
                                                                         h : MeasureTheory.AEDisjoint μ s t
                                                                         ⊢ MeasureTheory.AEDisjoint μ t s
                                                                       -/
protected theorem symm (h : AEDisjoint μ s t) : AEDisjoint μ t s := by rwa [AEDisjoint, inter_comm]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


protected theorem symmetric : Symmetric (AEDisjoint μ) := fun _ _ => AEDisjoint.symm


protected theorem comm : AEDisjoint μ s t ↔ AEDisjoint μ t s :=
  ⟨AEDisjoint.symm, AEDisjoint.symm⟩


protected theorem _root_.Disjoint.aedisjoint (h : Disjoint s t) : AEDisjoint μ s t := by
  /-
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h : Disjoint s t
    ⊢ MeasureTheory.AEDisjoint μ s t
  -/
  rw [AEDisjoint, disjoint_iff_inter_eq_empty.1 h, measure_empty]
  /-
    🎉 no goals
  -/


protected theorem _root_.Pairwise.aedisjoint {f : ι → Set α} (hf : Pairwise (Disjoint on f)) :
    Pairwise (AEDisjoint μ on f) :=
  hf.mono fun _i _j h => h.aedisjoint


protected theorem _root_.Set.PairwiseDisjoint.aedisjoint {f : ι → Set α} {s : Set ι}
    (hf : s.PairwiseDisjoint f) : s.Pairwise (AEDisjoint μ on f) :=
  hf.mono' fun _i _j h => h.aedisjoint


theorem mono_ae (h : AEDisjoint μ s t) (hu : u ≤ᵐ[μ] s) (hv : v ≤ᵐ[μ] t) : AEDisjoint μ u v :=
  measure_mono_null_ae (hu.inter hv) h


protected theorem mono (h : AEDisjoint μ s t) (hu : u ⊆ s) (hv : v ⊆ t) : AEDisjoint μ u v :=
  mono_ae h (HasSubset.Subset.eventuallyLE hu) (HasSubset.Subset.eventuallyLE hv)


protected theorem congr (h : AEDisjoint μ s t) (hu : u =ᵐ[μ] s) (hv : v =ᵐ[μ] t) :
    AEDisjoint μ u v :=
  mono_ae h (Filter.EventuallyEq.le hu) (Filter.EventuallyEq.le hv)


@[simp]
theorem iUnion_left_iff {ι : Sort*} [Countable ι] {s : ι → Set α} :
    AEDisjoint μ (⋃ i, s i) t ↔ ∀ i, AEDisjoint μ (s i) t := by
  /-
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Set α
    ι : Sort u_3
    inst✝ : Countable ι
    s : ι → Set α
    ⊢ Iff (MeasureTheory.AEDisjoint μ (Set.iUnion fun i => s i) t) (∀ (i : ι), Mea …
  -/
  simp only [AEDisjoint, iUnion_inter, measure_iUnion_null_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_right_iff {ι : Sort*} [Countable ι] {t : ι → Set α} :
    AEDisjoint μ s (⋃ i, t i) ↔ ∀ i, AEDisjoint μ s (t i) := by
  /-
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ι : Sort u_3
    inst✝ : Countable ι
    t : ι → Set α
    ⊢ Iff (MeasureTheory.AEDisjoint μ s (Set.iUnion fun i => t i)) (∀ (i : ι), Mea …
  -/
  simp only [AEDisjoint, inter_iUnion, measure_iUnion_null_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem union_left_iff : AEDisjoint μ (s ∪ t) u ↔ AEDisjoint μ s u ∧ AEDisjoint μ t u := by
  /-
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    ⊢ Iff (MeasureTheory.AEDisjoint μ (Union.union s t) u) (And (MeasureTheory.AED …
  -/
  simp [union_eq_iUnion, and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem union_right_iff : AEDisjoint μ s (t ∪ u) ↔ AEDisjoint μ s t ∧ AEDisjoint μ s u := by
  /-
    α : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    ⊢ Iff (MeasureTheory.AEDisjoint μ s (Union.union t u)) (And (MeasureTheory.AED …
  -/
  simp [union_eq_iUnion, and_comm]
  /-
    🎉 no goals
  -/


theorem union_left (hs : AEDisjoint μ s u) (ht : AEDisjoint μ t u) : AEDisjoint μ (s ∪ t) u :=
  union_left_iff.mpr ⟨hs, ht⟩


theorem union_right (ht : AEDisjoint μ s t) (hu : AEDisjoint μ s u) : AEDisjoint μ s (t ∪ u) :=
  union_right_iff.2 ⟨ht, hu⟩


theorem diff_ae_eq_left (h : AEDisjoint μ s t) : (s \ t : Set α) =ᵐ[μ] s :=
  @diff_self_inter _ s t ▸ diff_null_ae_eq_self h


theorem diff_ae_eq_right (h : AEDisjoint μ s t) : (t \ s : Set α) =ᵐ[μ] t :=
  diff_ae_eq_left <| AEDisjoint.symm h


theorem measure_diff_left (h : AEDisjoint μ s t) : μ (s \ t) = μ s :=
  measure_congr <| AEDisjoint.diff_ae_eq_left h


theorem measure_diff_right (h : AEDisjoint μ s t) : μ (t \ s) = μ t :=
  measure_congr <| AEDisjoint.diff_ae_eq_right h


/-- If `s` and `t` are `μ`-a.e. disjoint, then `s \ u` and `t` are disjoint for some measurable null
set `u`. -/
theorem exists_disjoint_diff (h : AEDisjoint μ s t) :
    ∃ u, MeasurableSet u ∧ μ u = 0 ∧ Disjoint (s \ u) t :=
  ⟨toMeasurable μ (s ∩ t), measurableSet_toMeasurable _ _, (measure_toMeasurable _).trans h,
    disjoint_sdiff_self_left.mono_left (b := s \ t) fun x hx => by
      /-
        α : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s t : Set α
        h : MeasureTheory.AEDisjoint μ s t
        x : α
        hx : Membership.mem (SDiff.sdiff s (MeasureTheory.toMeasurable μ (Inter.inter  …
        ⊢ Membership.mem (SDiff.sdiff s t) x
      -/
      simpa using ⟨hx.1, fun hxt => hx.2 <| subset_toMeasurable _ _ ⟨hx.1, hxt⟩⟩⟩
      /-
        🎉 no goals
      -/


theorem of_null_right (h : μ t = 0) : AEDisjoint μ s t :=
  measure_mono_null inter_subset_right h


theorem of_null_left (h : μ s = 0) : AEDisjoint μ s t :=
  AEDisjoint.symm (of_null_right h)


theorem aedisjoint_compl_left : AEDisjoint μ sᶜ s :=
  (@disjoint_compl_left _ _ s).aedisjoint


theorem aedisjoint_compl_right : AEDisjoint μ s sᶜ :=
  (@disjoint_compl_right _ _ s).aedisjoint


