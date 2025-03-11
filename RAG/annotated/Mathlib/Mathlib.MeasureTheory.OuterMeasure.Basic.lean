@[simp]
theorem measure_empty : μ ∅ = 0 := OuterMeasureClass.measure_empty μ


@[mono, gcongr]
theorem measure_mono (h : s ⊆ t) : μ s ≤ μ t :=
  OuterMeasureClass.measure_mono μ h


theorem measure_mono_null (h : s ⊆ t) (ht : μ t = 0) : μ s = 0 :=
  eq_bot_mono (measure_mono h) ht


theorem measure_pos_of_superset (h : s ⊆ t) (hs : μ s ≠ 0) : 0 < μ t :=
  hs.bot_lt.trans_le (measure_mono h)


theorem measure_iUnion_le [Countable ι] (s : ι → Set α) : μ (⋃ i, s i) ≤ ∑' i, μ (s i) := by
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝ : Countable ι
    s : ι → Set α
    ⊢ LE.le (μ (Set.iUnion fun i => s i)) (tsum fun i => μ (s i))
  -/
  refine rel_iSup_tsum μ measure_empty (· ≤ ·) (fun t ↦ ?_) _
  calc
    μ (⋃ i, t i) = μ (⋃ i, disjointed t i) := by rw [iUnion_disjointed]
    _ ≤ ∑' i, μ (disjointed t i) :=
      OuterMeasureClass.measure_iUnion_nat_le _ _ (disjoint_disjointed _)
    _ ≤ ∑' i, μ (t i) := by gcongr; exact disjointed_subset ..


theorem measure_biUnion_le {I : Set ι} (μ : F) (hI : I.Countable) (s : ι → Set α) :
    μ (⋃ i ∈ I, s i) ≤ ∑' i : I, μ (s i) := by
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    I : Set ι
    μ : F
    hI : I.Countable
    s : ι → Set α
    ⊢ LE.le (μ (Set.iUnion fun i => Set.iUnion fun h => s i)) (tsum fun i => μ (s  …
  -/
  have := hI.to_subtype
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    I : Set ι
    μ : F
    hI : I.Countable
    s : ι → Set α
    this : Countable ↑I
    ⊢ LE.le (μ (Set.iUnion fun i => Set.iUnion fun h => s i)) (tsum fun i => μ (s  …
  -/
  rw [biUnion_eq_iUnion]
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    I : Set ι
    μ : F
    hI : I.Countable
    s : ι → Set α
    this : Countable ↑I
    ⊢ LE.le (μ (Set.iUnion fun x => s ↑x)) (tsum fun i => μ (s ↑i))
  -/
  apply measure_iUnion_le
  /-
    🎉 no goals
  -/


theorem measure_biUnion_finset_le (I : Finset ι) (s : ι → Set α) :
    μ (⋃ i ∈ I, s i) ≤ ∑ i ∈ I, μ (s i) :=
  (measure_biUnion_le μ I.countable_toSet s).trans_eq <| I.tsum_subtype (μ <| s ·)


theorem measure_iUnion_fintype_le [Fintype ι] (μ : F) (s : ι → Set α) :
    μ (⋃ i, s i) ≤ ∑ i, μ (s i) := by
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    inst✝ : Fintype ι
    μ : F
    s : ι → Set α
    ⊢ LE.le (μ (Set.iUnion fun i => s i)) (Finset.univ.sum fun i => μ (s i))
  -/
  simpa using measure_biUnion_finset_le Finset.univ s
  /-
    🎉 no goals
  -/


theorem measure_union_le (s t : Set α) : μ (s ∪ t) ≤ μ s + μ t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    ⊢ LE.le (μ (Union.union s t)) (HAdd.hAdd (μ s) (μ t))
  -/
  simpa [union_eq_iUnion] using measure_iUnion_fintype_le μ (cond · s t)
  /-
    🎉 no goals
  -/


lemma measure_univ_le_add_compl (s : Set α) : μ univ ≤ μ s + μ sᶜ :=
  s.union_compl_self ▸ measure_union_le s sᶜ


theorem measure_le_inter_add_diff (μ : F) (s t : Set α) : μ s ≤ μ (s ∩ t) + μ (s \ t) := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    ⊢ LE.le (μ s) (HAdd.hAdd (μ (Inter.inter s t)) (μ (SDiff.sdiff s t)))
  -/
  simpa using measure_union_le (s ∩ t) (s \ t)
  /-
    🎉 no goals
  -/


theorem measure_diff_null (ht : μ t = 0) : μ (s \ t) = μ s :=
  (measure_mono diff_subset).antisymm <| calc
    μ s ≤ μ (s ∩ t) + μ (s \ t) := measure_le_inter_add_diff _ _ _
                              /-
                                α : Type u_1
                                F : Type u_3
                                inst✝¹ : FunLike F (Set α) ENNReal
                                inst✝ : MeasureTheory.OuterMeasureClass F α
                                μ : F
                                s t : Set α
                                ht : Eq (μ t) 0
                                ⊢ LE.le (HAdd.hAdd (μ (Inter.inter s t)) (μ (SDiff.sdiff s t))) (HAdd.hAdd (μ  …
                              -/
    _ ≤ μ t + μ (s \ t) := by gcongr; apply inter_subset_right
                                      /-
                                        🎉 no goals
                                      -/
                        /-
                          α : Type u_1
                          F : Type u_3
                          inst✝¹ : FunLike F (Set α) ENNReal
                          inst✝ : MeasureTheory.OuterMeasureClass F α
                          μ : F
                          s t : Set α
                          ht : Eq (μ t) 0
                          ⊢ Eq (HAdd.hAdd (μ t) (μ (SDiff.sdiff s t))) (μ (SDiff.sdiff s t))
                        -/
    _ = μ (s \ t) := by simp [ht]
                        /-
                          🎉 no goals
                        -/


theorem measure_biUnion_null_iff {I : Set ι} (hI : I.Countable) {s : ι → Set α} :
    μ (⋃ i ∈ I, s i) = 0 ↔ ∀ i ∈ I, μ (s i) = 0 := by
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    I : Set ι
    hI : I.Countable
    s : ι → Set α
    ⊢ Iff (Eq (μ (Set.iUnion fun i => Set.iUnion fun h => s i)) 0) (∀ (i : ι), Mem …
  -/
  refine ⟨fun h i hi ↦ measure_mono_null (subset_biUnion_of_mem hi) h, fun h ↦ ?_⟩
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    I : Set ι
    hI : I.Countable
    s : ι → Set α
    h : ∀ (i : ι), Membership.mem I i → Eq (μ (s i)) 0
    ⊢ Eq (μ (Set.iUnion fun i => Set.iUnion fun h => s i)) 0
  -/
  have _ := hI.to_subtype
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    I : Set ι
    hI : I.Countable
    s : ι → Set α
    h : ∀ (i : ι), Membership.mem I i → Eq (μ (s i)) 0
    x✝ : Countable ↑I
    ⊢ Eq (μ (Set.iUnion fun i => Set.iUnion fun h => s i)) 0
  -/
  simpa [h] using measure_iUnion_le (μ := μ) fun x : I ↦ s x
  /-
    🎉 no goals
  -/


theorem measure_sUnion_null_iff {S : Set (Set α)} (hS : S.Countable) :
    μ (⋃₀ S) = 0 ↔ ∀ s ∈ S, μ s = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    S : Set (Set α)
    hS : S.Countable
    ⊢ Iff (Eq (μ S.sUnion) 0) (∀ (s : Set α), Membership.mem S s → Eq (μ s) 0)
  -/
  rw [sUnion_eq_biUnion, measure_biUnion_null_iff hS]
  /-
    🎉 no goals
  -/


@[simp]
theorem measure_iUnion_null_iff {ι : Sort*} [Countable ι] {s : ι → Set α} :
    μ (⋃ i, s i) = 0 ↔ ∀ i, μ (s i) = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    μ : F
    ι : Sort u_4
    inst✝ : Countable ι
    s : ι → Set α
    ⊢ Iff (Eq (μ (Set.iUnion fun i => s i)) 0) (∀ (i : ι), Eq (μ (s i)) 0)
  -/
  rw [← sUnion_range, measure_sUnion_null_iff (countable_range s), forall_mem_range]
  /-
    🎉 no goals
  -/


alias ⟨_, measure_iUnion_null⟩ := measure_iUnion_null_iff


@[simp]
theorem measure_union_null_iff : μ (s ∪ t) = 0 ↔ μ s = 0 ∧ μ t = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    ⊢ Iff (Eq (μ (Union.union s t)) 0) (And (Eq (μ s) 0) (Eq (μ t) 0))
  -/
  simp [union_eq_iUnion, and_comm]
  /-
    🎉 no goals
  -/


                                                                               /-
                                                                                 α : Type u_1
                                                                                 F : Type u_3
                                                                                 inst✝¹ : FunLike F (Set α) ENNReal
                                                                                 inst✝ : MeasureTheory.OuterMeasureClass F α
                                                                                 μ : F
                                                                                 s t : Set α
                                                                                 hs : Eq (μ s) 0
                                                                                 ht : Eq (μ t) 0
                                                                                 ⊢ Eq (μ (Union.union s t)) 0
                                                                               -/
theorem measure_union_null (hs : μ s = 0) (ht : μ t = 0) : μ (s ∪ t) = 0 := by simp [*]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma measure_null_iff_singleton (hs : s.Countable) : μ s = 0 ↔ ∀ x ∈ s, μ {x} = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Set α
    hs : s.Countable
    ⊢ Iff (Eq (μ s) 0) (∀ (x : α), Membership.mem s x → Eq (μ (Singleton.singleton …
  -/
  rw [← measure_biUnion_null_iff hs, biUnion_of_singleton]
  /-
    🎉 no goals
  -/


/-- Let `μ` be an (outer) measure; let `s : ι → Set α` be a sequence of sets, `S = ⋃ n, s n`.
If `μ (S \ s n)` tends to zero along some nontrivial filter (usually `Filter.atTop` on `ι = ℕ`),
then `μ S = ⨆ n, μ (s n)`. -/
theorem measure_iUnion_of_tendsto_zero {ι} (μ : F) {s : ι → Set α} (l : Filter ι) [NeBot l]
    (h0 : Tendsto (fun k => μ ((⋃ n, s n) \ s k)) l (𝓝 0)) : μ (⋃ n, s n) = ⨆ n, μ (s n) := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    ι : Type u_4
    μ : F
    s : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    h0 : Filter.Tendsto (fun k => μ (SDiff.sdiff (Set.iUnion fun n => s n) (s k))) …
    ⊢ Eq (μ (Set.iUnion fun n => s n)) (iSup fun n => μ (s n))
  -/
  refine le_antisymm ?_ <| iSup_le fun n ↦ measure_mono <| subset_iUnion _ _
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    ι : Type u_4
    μ : F
    s : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    h0 : Filter.Tendsto (fun k => μ (SDiff.sdiff (Set.iUnion fun n => s n) (s k))) …
    ⊢ LE.le (μ (Set.iUnion fun n => s n)) (iSup fun n => μ (s n))
  -/
  set S := ⋃ n, s n
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    ι : Type u_4
    μ : F
    s : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    S : Set α := Set.iUnion fun n => s n
    h0 : Filter.Tendsto (fun k => μ (SDiff.sdiff S (s k))) l (nhds 0)
    ⊢ LE.le (μ S) (iSup fun n => μ (s n))
  -/
  set M := ⨆ n, μ (s n)
  have A : ∀ k, μ S ≤ M + μ (S \ s k) := fun k ↦ calc
    μ S ≤ μ (S ∩ s k) + μ (S \ s k) := measure_le_inter_add_diff _ _ _
    _ ≤ μ (s k) + μ (S \ s k) := by gcongr; apply inter_subset_right
    _ ≤ M + μ (S \ s k) := by gcongr; exact le_iSup (μ ∘ s) k
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    ι : Type u_4
    μ : F
    s : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    S : Set α := Set.iUnion fun n => s n
    h0 : Filter.Tendsto (fun k => μ (SDiff.sdiff S (s k))) l (nhds 0)
    M : ENNReal := iSup fun n => μ (s n)
    A : ∀ (k : ι), LE.le (μ S) (HAdd.hAdd M (μ (SDiff.sdiff S (s k))))
    ⊢ LE.le (μ S) M
  -/
  have B : Tendsto (fun k ↦ M + μ (S \ s k)) l (𝓝 M) := by simpa using tendsto_const_nhds.add h0
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    ι : Type u_4
    μ : F
    s : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    S : Set α := Set.iUnion fun n => s n
    h0 : Filter.Tendsto (fun k => μ (SDiff.sdiff S (s k))) l (nhds 0)
    M : ENNReal := iSup fun n => μ (s n)
    A : ∀ (k : ι), LE.le (μ S) (HAdd.hAdd M (μ (SDiff.sdiff S (s k))))
    B : Filter.Tendsto (fun k => HAdd.hAdd M (μ (SDiff.sdiff S (s k)))) l (nhds M)
    ⊢ LE.le (μ S) M
  -/
  exact ge_of_tendsto' B A
  /-
    🎉 no goals
  -/


/-- If a set has zero measure in a neighborhood of each of its points, then it has zero measure
in a second-countable space. -/
theorem measure_null_of_locally_null [TopologicalSpace α] [SecondCountableTopology α]
    (s : Set α) (hs : ∀ x ∈ s, ∃ u ∈ 𝓝[s] x, μ u = 0) : μ s = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝¹ : TopologicalSpace α
    inst✝ : SecondCountableTopology α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → Exists fun u => And (Membership.mem (nhds …
    ⊢ Eq (μ s) 0
  -/
  choose! u hxu hu₀ using hs
  /-
    α : Type u_1
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝¹ : TopologicalSpace α
    inst✝ : SecondCountableTopology α
    s : Set α
    u : α → Set α
    hxu : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x s) (u x)
    hu₀ : ∀ (x : α), Membership.mem s x → Eq (μ (u x)) 0
    ⊢ Eq (μ s) 0
  -/
  choose t ht using TopologicalSpace.countable_cover_nhdsWithin hxu
  /-
    α : Type u_1
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝¹ : TopologicalSpace α
    inst✝ : SecondCountableTopology α
    s : Set α
    u : α → Set α
    hxu : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x s) (u x)
    hu₀ : ∀ (x : α), Membership.mem s x → Eq (μ (u x)) 0
    t : Set α
    ht : And (HasSubset.Subset t s) (And t.Countable (HasSubset.Subset s (Set.iUni …
    ⊢ Eq (μ s) 0
  -/
  rcases ht with ⟨ts, t_count, ht⟩
  /-
    case intro.intro
    α : Type u_1
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝¹ : TopologicalSpace α
    inst✝ : SecondCountableTopology α
    s : Set α
    u : α → Set α
    hxu : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x s) (u x)
    hu₀ : ∀ (x : α), Membership.mem s x → Eq (μ (u x)) 0
    t : Set α
    ts : HasSubset.Subset t s
    t_count : t.Countable
    ht : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => u x)
    ⊢ Eq (μ s) 0
  -/
  apply measure_mono_null ht
  /-
    case intro.intro
    α : Type u_1
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝¹ : TopologicalSpace α
    inst✝ : SecondCountableTopology α
    s : Set α
    u : α → Set α
    hxu : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x s) (u x)
    hu₀ : ∀ (x : α), Membership.mem s x → Eq (μ (u x)) 0
    t : Set α
    ts : HasSubset.Subset t s
    t_count : t.Countable
    ht : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => u x)
    ⊢ Eq (μ (Set.iUnion fun x => Set.iUnion fun h => u x)) 0
  -/
  exact (measure_biUnion_null_iff t_count).2 fun x hx => hu₀ x (ts hx)
  /-
    🎉 no goals
  -/


/-- If `m s ≠ 0`, then for some point `x ∈ s` and any `t ∈ 𝓝[s] x` we have `0 < m t`. -/
theorem exists_mem_forall_mem_nhdsWithin_pos_measure [TopologicalSpace α]
    [SecondCountableTopology α] {s : Set α} (hs : μ s ≠ 0) :
    ∃ x ∈ s, ∀ t ∈ 𝓝[s] x, 0 < μ t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝¹ : TopologicalSpace α
    inst✝ : SecondCountableTopology α
    s : Set α
    hs : Ne (μ s) 0
    ⊢ Exists fun x => And (Membership.mem s x) (∀ (t : Set α), Membership.mem (nhd …
  -/
  contrapose! hs
  /-
    α : Type u_1
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝¹ : TopologicalSpace α
    inst✝ : SecondCountableTopology α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    ⊢ Eq (μ s) 0
  -/
  simp only [nonpos_iff_eq_zero] at hs
  /-
    α : Type u_1
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝¹ : TopologicalSpace α
    inst✝ : SecondCountableTopology α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    ⊢ Eq (μ s) 0
  -/
  exact measure_null_of_locally_null s hs
  /-
    🎉 no goals
  -/


@[deprecated measure_empty (since := "2024-05-14")]
theorem empty' (m : OuterMeasure α) : m ∅ = 0 := measure_empty


@[deprecated measure_mono (since := "2024-05-14")]
                                                                             /-
                                                                               α : Type u_1
                                                                               m : MeasureTheory.OuterMeasure α
                                                                               s₁ s₂ : Set α
                                                                               h : HasSubset.Subset s₁ s₂
                                                                               ⊢ LE.le (m s₁) (m s₂)
                                                                             -/
theorem mono' (m : OuterMeasure α) {s₁ s₂} (h : s₁ ⊆ s₂) : m s₁ ≤ m s₂ := by gcongr
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[deprecated measure_mono_null (since := "2024-05-14")]
theorem mono_null (m : OuterMeasure α) {s t} (h : s ⊆ t) (ht : m t = 0) : m s = 0 :=
  measure_mono_null h ht


@[deprecated measure_pos_of_superset (since := "2024-05-14")]
theorem pos_of_subset_ne_zero (m : OuterMeasure α) {a b : Set α} (hs : a ⊆ b) (hnz : m a ≠ 0) :
    0 < m b :=
  measure_pos_of_superset hs hnz


@[deprecated measure_iUnion_le (since := "2024-05-14")]
protected theorem iUnion (m : OuterMeasure α) {β} [Countable β] (s : β → Set α) :
    m (⋃ i, s i) ≤ ∑' i, m (s i) :=
  measure_iUnion_le s


@[deprecated measure_biUnion_null_iff (since := "2024-05-14")]
theorem biUnion_null_iff (m : OuterMeasure α) {s : Set β} (hs : s.Countable) {t : β → Set α} :
    m (⋃ i ∈ s, t i) = 0 ↔ ∀ i ∈ s, m (t i) = 0 :=
  measure_biUnion_null_iff hs


@[deprecated measure_sUnion_null_iff (since := "2024-05-14")]
theorem sUnion_null_iff (m : OuterMeasure α) {S : Set (Set α)} (hS : S.Countable) :
    m (⋃₀ S) = 0 ↔ ∀ s ∈ S, m s = 0 := measure_sUnion_null_iff hS


@[deprecated measure_iUnion_null_iff (since := "2024-05-14")]
theorem iUnion_null_iff {ι : Sort*} [Countable ι] (m : OuterMeasure α) {s : ι → Set α} :
    m (⋃ i, s i) = 0 ↔ ∀ i, m (s i) = 0 :=
  measure_iUnion_null_iff


@[deprecated measure_iUnion_null (since := "2024-05-14")]
alias ⟨_, iUnion_null⟩ := iUnion_null_iff


@[deprecated measure_biUnion_finset_le (since := "2024-05-14")]
protected theorem iUnion_finset (m : OuterMeasure α) (s : β → Set α) (t : Finset β) :
    m (⋃ i ∈ t, s i) ≤ ∑ i ∈ t, m (s i) :=
  measure_biUnion_finset_le t s


@[deprecated measure_union_le (since := "2024-05-14")]
protected theorem union (m : OuterMeasure α) (s₁ s₂ : Set α) : m (s₁ ∪ s₂) ≤ m s₁ + m s₂ :=
  measure_union_le s₁ s₂


/-- If a set has zero measure in a neighborhood of each of its points, then it has zero measure
in a second-countable space. -/
@[deprecated measure_null_of_locally_null (since := "2024-05-14")]
theorem null_of_locally_null [TopologicalSpace α] [SecondCountableTopology α] (m : OuterMeasure α)
    (s : Set α) (hs : ∀ x ∈ s, ∃ u ∈ 𝓝[s] x, m u = 0) : m s = 0 :=
  measure_null_of_locally_null s hs


/-- If `m s ≠ 0`, then for some point `x ∈ s` and any `t ∈ 𝓝[s] x` we have `0 < m t`. -/
@[deprecated exists_mem_forall_mem_nhdsWithin_pos_measure (since := "2024-05-14")]
theorem exists_mem_forall_mem_nhds_within_pos [TopologicalSpace α] [SecondCountableTopology α]
    (m : OuterMeasure α) {s : Set α} (hs : m s ≠ 0) : ∃ x ∈ s, ∀ t ∈ 𝓝[s] x, 0 < m t :=
  exists_mem_forall_mem_nhdsWithin_pos_measure hs


/-- If `s : ι → Set α` is a sequence of sets, `S = ⋃ n, s n`, and `m (S \ s n)` tends to zero along
some nontrivial filter (usually `atTop` on `ι = ℕ`), then `m S = ⨆ n, m (s n)`. -/
theorem iUnion_of_tendsto_zero {ι} (m : OuterMeasure α) {s : ι → Set α} (l : Filter ι) [NeBot l]
    (h0 : Tendsto (fun k => m ((⋃ n, s n) \ s k)) l (𝓝 0)) : m (⋃ n, s n) = ⨆ n, m (s n) :=
  measure_iUnion_of_tendsto_zero m l h0


/-- If `s : ℕ → Set α` is a monotone sequence of sets such that `∑' k, m (s (k + 1) \ s k) ≠ ∞`,
then `m (⋃ n, s n) = ⨆ n, m (s n)`. -/
theorem iUnion_nat_of_monotone_of_tsum_ne_top (m : OuterMeasure α) {s : ℕ → Set α}
    (h_mono : ∀ n, s n ⊆ s (n + 1)) (h0 : (∑' k, m (s (k + 1) \ s k)) ≠ ∞) :
    m (⋃ n, s n) = ⨆ n, m (s n) := by
  classical
  refine measure_iUnion_of_tendsto_zero m atTop ?_
  refine tendsto_nhds_bot_mono' (ENNReal.tendsto_sum_nat_add _ h0) fun n => ?_
  refine (m.mono ?_).trans (measure_iUnion_le _)
  -- Current goal: `(⋃ k, s k) \ s n ⊆ ⋃ k, s (k + n + 1) \ s (k + n)`
  have h' : Monotone s := @monotone_nat_of_le_succ (Set α) _ _ h_mono
  simp only [diff_subset_iff, iUnion_subset_iff]
  intro i x hx
  have : ∃i, x ∈ s i := by exists i
  rcases Nat.findX this with ⟨j, hj, hlt⟩
  clear hx i
  rcases le_or_lt j n with hjn | hnj
  · exact Or.inl (h' hjn hj)
  have : j - (n + 1) + n + 1 = j := by omega
  refine Or.inr (mem_iUnion.2 ⟨j - (n + 1), ?_, hlt _ ?_⟩)
  · rwa [this]
  · rw [← Nat.succ_le_iff, Nat.succ_eq_add_one, this]


@[deprecated measure_le_inter_add_diff (since := "2024-05-14")]
theorem le_inter_add_diff {m : OuterMeasure α} {t : Set α} (s : Set α) :
    m t ≤ m (t ∩ s) + m (t \ s) :=
  measure_le_inter_add_diff m t s


@[deprecated measure_diff_null (since := "2024-05-14")]
theorem diff_null (m : OuterMeasure α) (s : Set α) {t : Set α} (ht : m t = 0) : m (s \ t) = m s :=
  measure_diff_null ht


@[deprecated measure_union_null (since := "2024-05-14")]
theorem union_null (m : OuterMeasure α) {s₁ s₂ : Set α} (h₁ : m s₁ = 0) (h₂ : m s₂ = 0) :
    m (s₁ ∪ s₂) = 0 :=
  measure_union_null h₁ h₂


theorem coe_fn_injective : Injective fun (μ : OuterMeasure α) (s : Set α) => μ s :=
  DFunLike.coe_injective


@[ext]
theorem ext {μ₁ μ₂ : OuterMeasure α} (h : ∀ s, μ₁ s = μ₂ s) : μ₁ = μ₂ :=
  DFunLike.ext _ _ h


/-- A version of `MeasureTheory.OuterMeasure.ext` that assumes `μ₁ s = μ₂ s` on all *nonempty*
sets `s`, and gets `μ₁ ∅ = μ₂ ∅` from `MeasureTheory.OuterMeasure.empty'`. -/
theorem ext_nonempty {μ₁ μ₂ : OuterMeasure α} (h : ∀ s : Set α, s.Nonempty → μ₁ s = μ₂ s) :
    μ₁ = μ₂ :=
                                                         /-
                                                           α : Type u_1
                                                           μ₁ μ₂ : MeasureTheory.OuterMeasure α
                                                           h : ∀ (s : Set α), s.Nonempty → Eq (μ₁ s) (μ₂ s)
                                                           s : Set α
                                                           he : Eq s EmptyCollection.emptyCollection
                                                           ⊢ Eq (μ₁ s) (μ₂ s)
                                                         -/
  ext fun s => s.eq_empty_or_nonempty.elim (fun he => by simp [he]) (h s)
                                                         /-
                                                           🎉 no goals
                                                         -/


