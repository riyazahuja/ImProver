instance ae_isMeasurablyGenerated : IsMeasurablyGenerated (ae μ) :=
  ⟨fun _s hs =>
    let ⟨t, hst, htm, htμ⟩ := exists_measurable_superset_of_null hs
    ⟨tᶜ, compl_mem_ae_iff.2 htμ, htm.compl, compl_subset_comm.1 hst⟩⟩


/-- See also `MeasureTheory.ae_restrict_uIoc_iff`. -/
theorem ae_uIoc_iff [LinearOrder α] {a b : α} {P : α → Prop} :
    (∀ᵐ x ∂μ, x ∈ Ι a b → P x) ↔ (∀ᵐ x ∂μ, x ∈ Ioc a b → P x) ∧ ∀ᵐ x ∂μ, x ∈ Ioc b a → P x := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder α
    a b : α
    P : α → Prop
    ⊢ Iff (Filter.Eventually (fun x => Membership.mem (Set.uIoc a b) x → P x) (Mea …
  -/
  simp only [uIoc_eq_union, mem_union, or_imp, eventually_and]
  /-
    🎉 no goals
  -/


theorem measure_union (hd : Disjoint s₁ s₂) (h : MeasurableSet s₂) : μ (s₁ ∪ s₂) = μ s₁ + μ s₂ :=
  measure_union₀ h.nullMeasurableSet hd.aedisjoint


theorem measure_union' (hd : Disjoint s₁ s₂) (h : MeasurableSet s₁) : μ (s₁ ∪ s₂) = μ s₁ + μ s₂ :=
  measure_union₀' h.nullMeasurableSet hd.aedisjoint


theorem measure_inter_add_diff (s : Set α) (ht : MeasurableSet t) : μ (s ∩ t) + μ (s \ t) = μ s :=
  measure_inter_add_diff₀ _ ht.nullMeasurableSet


theorem measure_diff_add_inter (s : Set α) (ht : MeasurableSet t) : μ (s \ t) + μ (s ∩ t) = μ s :=
  (add_comm _ _).trans (measure_inter_add_diff s ht)


theorem measure_union_add_inter (s : Set α) (ht : MeasurableSet t) :
    μ (s ∪ t) + μ (s ∩ t) = μ s + μ t := by
  rw [← measure_inter_add_diff (s ∪ t) ht, Set.union_inter_cancel_right, union_diff_right, ←
    measure_inter_add_diff s ht]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t s : Set α
    ht : MeasurableSet t
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (μ t) (μ (SDiff.sdiff s t))) (μ (Inter.inter s t))) …
  -/
  ac_rfl
  /-
    🎉 no goals
  -/


theorem measure_union_add_inter' (hs : MeasurableSet s) (t : Set α) :
    μ (s ∪ t) + μ (s ∩ t) = μ s + μ t := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    t : Set α
    ⊢ Eq (HAdd.hAdd (μ (Union.union s t)) (μ (Inter.inter s t))) (HAdd.hAdd (μ s)  …
  -/
  rw [union_comm, inter_comm, measure_union_add_inter t hs, add_comm]
  /-
    🎉 no goals
  -/


lemma measure_symmDiff_eq (hs : NullMeasurableSet s μ) (ht : NullMeasurableSet t μ) :
    μ (s ∆ t) = μ (s \ t) + μ (t \ s) := by
  simpa only [symmDiff_def, sup_eq_union]
    using measure_union₀ (ht.diff hs) disjoint_sdiff_sdiff.aedisjoint


lemma measure_symmDiff_le (s t u : Set α) :
    μ (s ∆ u) ≤ μ (s ∆ t) + μ (t ∆ u) :=
  le_trans (μ.mono <| symmDiff_triangle s t u) (measure_union_le (s ∆ t) (t ∆ u))


theorem measure_add_measure_compl (h : MeasurableSet s) : μ s + μ sᶜ = μ univ :=
  measure_add_measure_compl₀ h.nullMeasurableSet


theorem measure_biUnion₀ {s : Set β} {f : β → Set α} (hs : s.Countable)
    (hd : s.Pairwise (AEDisjoint μ on f)) (h : ∀ b ∈ s, NullMeasurableSet (f b) μ) :
    μ (⋃ b ∈ s, f b) = ∑' p : s, μ (f p) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set β
    f : β → Set α
    hs : s.Countable
    hd : s.Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) f)
    h : ∀ (b : β), Membership.mem s b → MeasureTheory.NullMeasurableSet (f b) μ
    ⊢ Eq (μ (Set.iUnion fun b => Set.iUnion fun h => f b)) (tsum fun p => μ (f ↑p))
  -/
  haveI := hs.toEncodable
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set β
    f : β → Set α
    hs : s.Countable
    hd : s.Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) f)
    h : ∀ (b : β), Membership.mem s b → MeasureTheory.NullMeasurableSet (f b) μ
    this : Encodable ↑s
    ⊢ Eq (μ (Set.iUnion fun b => Set.iUnion fun h => f b)) (tsum fun p => μ (f ↑p))
  -/
  rw [biUnion_eq_iUnion]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set β
    f : β → Set α
    hs : s.Countable
    hd : s.Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) f)
    h : ∀ (b : β), Membership.mem s b → MeasureTheory.NullMeasurableSet (f b) μ
    this : Encodable ↑s
    ⊢ Eq (μ (Set.iUnion fun x => f ↑x)) (tsum fun p => μ (f ↑p))
  -/
  exact measure_iUnion₀ (hd.on_injective Subtype.coe_injective fun x => x.2) fun x => h x x.2
  /-
    🎉 no goals
  -/


theorem measure_biUnion {s : Set β} {f : β → Set α} (hs : s.Countable) (hd : s.PairwiseDisjoint f)
    (h : ∀ b ∈ s, MeasurableSet (f b)) : μ (⋃ b ∈ s, f b) = ∑' p : s, μ (f p) :=
  measure_biUnion₀ hs hd.aedisjoint fun b hb => (h b hb).nullMeasurableSet


theorem measure_sUnion₀ {S : Set (Set α)} (hs : S.Countable) (hd : S.Pairwise (AEDisjoint μ))
    (h : ∀ s ∈ S, NullMeasurableSet s μ) : μ (⋃₀ S) = ∑' s : S, μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    S : Set (Set α)
    hs : S.Countable
    hd : S.Pairwise (MeasureTheory.AEDisjoint μ)
    h : ∀ (s : Set α), Membership.mem S s → MeasureTheory.NullMeasurableSet s μ
    ⊢ Eq (μ S.sUnion) (tsum fun s => μ ↑s)
  -/
  rw [sUnion_eq_biUnion, measure_biUnion₀ hs hd h]
  /-
    🎉 no goals
  -/


theorem measure_sUnion {S : Set (Set α)} (hs : S.Countable) (hd : S.Pairwise Disjoint)
    (h : ∀ s ∈ S, MeasurableSet s) : μ (⋃₀ S) = ∑' s : S, μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    S : Set (Set α)
    hs : S.Countable
    hd : S.Pairwise Disjoint
    h : ∀ (s : Set α), Membership.mem S s → MeasurableSet s
    ⊢ Eq (μ S.sUnion) (tsum fun s => μ ↑s)
  -/
  rw [sUnion_eq_biUnion, measure_biUnion hs hd h]
  /-
    🎉 no goals
  -/


theorem measure_biUnion_finset₀ {s : Finset ι} {f : ι → Set α}
    (hd : Set.Pairwise (↑s) (AEDisjoint μ on f)) (hm : ∀ b ∈ s, NullMeasurableSet (f b) μ) :
    μ (⋃ b ∈ s, f b) = ∑ p ∈ s, μ (f p) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    f : ι → Set α
    hd : (↑s).Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) f)
    hm : ∀ (b : ι), Membership.mem s b → MeasureTheory.NullMeasurableSet (f b) μ
    ⊢ Eq (μ (Set.iUnion fun b => Set.iUnion fun h => f b)) (s.sum fun p => μ (f p))
  -/
  rw [← Finset.sum_attach, Finset.attach_eq_univ, ← tsum_fintype]
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    f : ι → Set α
    hd : (↑s).Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) f)
    hm : ∀ (b : ι), Membership.mem s b → MeasureTheory.NullMeasurableSet (f b) μ
    ⊢ Eq (μ (Set.iUnion fun b => Set.iUnion fun h => f b)) (tsum fun b => μ (f ↑b))
  -/
  exact measure_biUnion₀ s.countable_toSet hd hm
  /-
    🎉 no goals
  -/


theorem measure_biUnion_finset {s : Finset ι} {f : ι → Set α} (hd : PairwiseDisjoint (↑s) f)
    (hm : ∀ b ∈ s, MeasurableSet (f b)) : μ (⋃ b ∈ s, f b) = ∑ p ∈ s, μ (f p) :=
  measure_biUnion_finset₀ hd.aedisjoint fun b hb => (hm b hb).nullMeasurableSet


/-- The measure of an a.e. disjoint union (even uncountable) of null-measurable sets is at least
the sum of the measures of the sets. -/
theorem tsum_meas_le_meas_iUnion_of_disjoint₀ {ι : Type*} {_ : MeasurableSpace α} (μ : Measure α)
    {As : ι → Set α} (As_mble : ∀ i : ι, NullMeasurableSet (As i) μ)
    (As_disj : Pairwise (AEDisjoint μ on As)) : (∑' i, μ (As i)) ≤ μ (⋃ i, As i) := by
  /-
    α : Type u_1
    ι : Type u_8
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    ⊢ LE.le (tsum fun i => μ (As i)) (μ (Set.iUnion fun i => As i))
  -/
  rw [ENNReal.tsum_eq_iSup_sum, iSup_le_iff]
  /-
    α : Type u_1
    ι : Type u_8
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    ⊢ ∀ (i : Finset ι), LE.le (i.sum fun a => μ (As a)) (μ (Set.iUnion fun i => As …
  -/
  intro s
  /-
    α : Type u_1
    ι : Type u_8
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    s : Finset ι
    ⊢ LE.le (s.sum fun a => μ (As a)) (μ (Set.iUnion fun i => As i))
  -/
  simp only [← measure_biUnion_finset₀ (fun _i _hi _j _hj hij => As_disj hij) fun i _ => As_mble i]
  /-
    α : Type u_1
    ι : Type u_8
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    s : Finset ι
    ⊢ LE.le (μ (Set.iUnion fun b => Set.iUnion fun h => As b)) (μ (Set.iUnion fun  …
  -/
  gcongr
  /-
    case h.h
    α : Type u_1
    ι : Type u_8
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    s : Finset ι
    i✝ : ι
    ⊢ HasSubset.Subset (Set.iUnion fun h => As i✝) (As i✝)
  -/
  exact iUnion_subset fun _ ↦ Subset.rfl
  /-
    🎉 no goals
  -/


/-- The measure of a disjoint union (even uncountable) of measurable sets is at least the sum of
the measures of the sets. -/
theorem tsum_meas_le_meas_iUnion_of_disjoint {ι : Type*} {_ : MeasurableSpace α} (μ : Measure α)
    {As : ι → Set α} (As_mble : ∀ i : ι, MeasurableSet (As i))
    (As_disj : Pairwise (Disjoint on As)) : (∑' i, μ (As i)) ≤ μ (⋃ i, As i) :=
  tsum_meas_le_meas_iUnion_of_disjoint₀ μ (fun i ↦ (As_mble i).nullMeasurableSet)
    (fun _ _ h ↦ Disjoint.aedisjoint (As_disj h))


/-- If `s` is a countable set, then the measure of its preimage can be found as the sum of measures
of the fibers `f ⁻¹' {y}`. -/
theorem tsum_measure_preimage_singleton {s : Set β} (hs : s.Countable) {f : α → β}
    (hf : ∀ y ∈ s, MeasurableSet (f ⁻¹' {y})) : (∑' b : s, μ (f ⁻¹' {↑b})) = μ (f ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set β
    hs : s.Countable
    f : α → β
    hf : ∀ (y : β), Membership.mem s y → MeasurableSet (Set.preimage f (Singleton. …
    ⊢ Eq (tsum fun b => μ (Set.preimage f (Singleton.singleton ↑b))) (μ (Set.preim …
  -/
  rw [← Set.biUnion_preimage_singleton, measure_biUnion hs (pairwiseDisjoint_fiber f s) hf]
  /-
    🎉 no goals
  -/


lemma measure_preimage_eq_zero_iff_of_countable {s : Set β} {f : α → β} (hs : s.Countable) :
    μ (f ⁻¹' s) = 0 ↔ ∀ x ∈ s, μ (f ⁻¹' {x}) = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set β
    f : α → β
    hs : s.Countable
    ⊢ Iff (Eq (μ (Set.preimage f s)) 0) (∀ (x : β), Membership.mem s x → Eq (μ (Se …
  -/
  rw [← biUnion_preimage_singleton, measure_biUnion_null_iff hs]
  /-
    🎉 no goals
  -/


/-- If `s` is a `Finset`, then the measure of its preimage can be found as the sum of measures
of the fibers `f ⁻¹' {y}`. -/
theorem sum_measure_preimage_singleton (s : Finset β) {f : α → β}
    (hf : ∀ y ∈ s, MeasurableSet (f ⁻¹' {y})) : (∑ b ∈ s, μ (f ⁻¹' {b})) = μ (f ⁻¹' ↑s) := by
  simp only [← measure_biUnion_finset (pairwiseDisjoint_fiber f s) hf,
    Finset.set_biUnion_preimage_singleton]


theorem measure_diff_null' (h : μ (s₁ ∩ s₂) = 0) : μ (s₁ \ s₂) = μ s₁ :=
  measure_congr <| diff_ae_eq_self.2 h


theorem measure_add_diff (hs : NullMeasurableSet s μ) (t : Set α) :
    μ s + μ (t \ s) = μ (s ∪ t) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    t : Set α
    ⊢ Eq (HAdd.hAdd (μ s) (μ (SDiff.sdiff t s))) (μ (Union.union s t))
  -/
  rw [← measure_union₀' hs disjoint_sdiff_right.aedisjoint, union_diff_self]
  /-
    🎉 no goals
  -/


theorem measure_diff' (s : Set α) (hm : NullMeasurableSet t μ) (h_fin : μ t ≠ ∞) :
    μ (s \ t) = μ (s ∪ t) - μ t :=
                                       /-
                                         α : Type u_1
                                         m : MeasurableSpace α
                                         μ : MeasureTheory.Measure α
                                         t s : Set α
                                         hm : MeasureTheory.NullMeasurableSet t μ
                                         h_fin : Ne (μ t) Top.top
                                         ⊢ Eq (HAdd.hAdd (μ (SDiff.sdiff s t)) (μ t)) (μ (Union.union s t))
                                       -/
  ENNReal.eq_sub_of_add_eq h_fin <| by rw [add_comm, measure_add_diff hm, union_comm]
                                       /-
                                         🎉 no goals
                                       -/


theorem measure_diff (h : s₂ ⊆ s₁) (h₂ : NullMeasurableSet s₂ μ) (h_fin : μ s₂ ≠ ∞) :
                                    /-
                                      α : Type u_1
                                      m : MeasurableSpace α
                                      μ : MeasureTheory.Measure α
                                      s₁ s₂ : Set α
                                      h : HasSubset.Subset s₂ s₁
                                      h₂ : MeasureTheory.NullMeasurableSet s₂ μ
                                      h_fin : Ne (μ s₂) Top.top
                                      ⊢ Eq (μ (SDiff.sdiff s₁ s₂)) (HSub.hSub (μ s₁) (μ s₂))
                                    -/
    μ (s₁ \ s₂) = μ s₁ - μ s₂ := by rw [measure_diff' _ h₂ h_fin, union_eq_self_of_subset_right h]
                                    /-
                                      🎉 no goals
                                    -/


theorem le_measure_diff : μ s₁ - μ s₂ ≤ μ (s₁ \ s₂) :=
  tsub_le_iff_left.2 <| (measure_le_inter_add_diff μ s₁ s₂).trans <| by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s₁ s₂ : Set α
      ⊢ LE.le (HAdd.hAdd (μ (Inter.inter s₁ s₂)) (μ (SDiff.sdiff s₁ s₂))) (HAdd.hAdd …
    -/
    gcongr; apply inter_subset_right
            /-
              🎉 no goals
            -/


/-- If the measure of the symmetric difference of two sets is finite,
then one has infinite measure if and only if the other one does. -/
theorem measure_eq_top_iff_of_symmDiff (hμst : μ (s ∆ t) ≠ ∞) : μ s = ∞ ↔ μ t = ∞ := by
  suffices h : ∀ u v, μ (u ∆ v) ≠ ∞ → μ u = ∞ → μ v = ∞
    from ⟨h s t hμst, h t s (symmDiff_comm s t ▸ hμst)⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hμst : Ne (μ (symmDiff s t)) Top.top
    ⊢ ∀ (u v : Set α), Ne (μ (symmDiff u v)) Top.top → Eq (μ u) Top.top → Eq (μ v) …
  -/
  intro u v hμuv hμu
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hμst : Ne (μ (symmDiff s t)) Top.top
    u v : Set α
    hμuv : Ne (μ (symmDiff u v)) Top.top
    hμu : Eq (μ u) Top.top
    ⊢ Eq (μ v) Top.top
  -/
  by_contra! hμv
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hμst : Ne (μ (symmDiff s t)) Top.top
    u v : Set α
    hμuv : Ne (μ (symmDiff u v)) Top.top
    hμu : Eq (μ u) Top.top
    hμv : Ne (μ v) Top.top
    ⊢ False
  -/
  apply hμuv
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hμst : Ne (μ (symmDiff s t)) Top.top
    u v : Set α
    hμuv : Ne (μ (symmDiff u v)) Top.top
    hμu : Eq (μ u) Top.top
    hμv : Ne (μ v) Top.top
    ⊢ Eq (μ (symmDiff u v)) Top.top
  -/
  rw [Set.symmDiff_def, eq_top_iff]
  calc
    ∞ = μ u - μ v := by rw [ENNReal.sub_eq_top_iff.2 ⟨hμu, hμv⟩]
    _ ≤ μ (u \ v) := le_measure_diff
    _ ≤ μ (u \ v ∪ v \ u) := measure_mono subset_union_left


/-- If the measure of the symmetric difference of two sets is finite,
then one has finite measure if and only if the other one does. -/
theorem measure_ne_top_iff_of_symmDiff (hμst : μ (s ∆ t) ≠ ∞) : μ s ≠ ∞ ↔ μ t ≠ ∞ :=
    (measure_eq_top_iff_of_symmDiff hμst).ne


theorem measure_diff_lt_of_lt_add (hs : NullMeasurableSet s μ) (hst : s ⊆ t) (hs' : μ s ≠ ∞)
    {ε : ℝ≥0∞} (h : μ t < μ s + ε) : μ (t \ s) < ε := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    hst : HasSubset.Subset s t
    hs' : Ne (μ s) Top.top
    ε : ENNReal
    h : LT.lt (μ t) (HAdd.hAdd (μ s) ε)
    ⊢ LT.lt (μ (SDiff.sdiff t s)) ε
  -/
  rw [measure_diff hst hs hs']; rw [add_comm] at h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    hst : HasSubset.Subset s t
    hs' : Ne (μ s) Top.top
    ε : ENNReal
    h : LT.lt (μ t) (HAdd.hAdd ε (μ s))
    ⊢ LT.lt (HSub.hSub (μ t) (μ s)) ε
  -/
  exact ENNReal.sub_lt_of_lt_add (measure_mono hst) h
  /-
    🎉 no goals
  -/


theorem measure_diff_le_iff_le_add (hs : NullMeasurableSet s μ) (hst : s ⊆ t) (hs' : μ s ≠ ∞)
    {ε : ℝ≥0∞} : μ (t \ s) ≤ ε ↔ μ t ≤ μ s + ε := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    hst : HasSubset.Subset s t
    hs' : Ne (μ s) Top.top
    ε : ENNReal
    ⊢ Iff (LE.le (μ (SDiff.sdiff t s)) ε) (LE.le (μ t) (HAdd.hAdd (μ s) ε))
  -/
  rw [measure_diff hst hs hs', tsub_le_iff_left]
  /-
    🎉 no goals
  -/


theorem measure_eq_measure_of_null_diff {s t : Set α} (hst : s ⊆ t) (h_nulldiff : μ (t \ s) = 0) :
    μ s = μ t := measure_congr <|
      EventuallyLE.antisymm (HasSubset.Subset.eventuallyLE hst) (ae_le_set.mpr h_nulldiff)


theorem measure_eq_measure_of_between_null_diff {s₁ s₂ s₃ : Set α} (h12 : s₁ ⊆ s₂) (h23 : s₂ ⊆ s₃)
    (h_nulldiff : μ (s₃ \ s₁) = 0) : μ s₁ = μ s₂ ∧ μ s₂ = μ s₃ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s₁ s₂ s₃ : Set α
    h12 : HasSubset.Subset s₁ s₂
    h23 : HasSubset.Subset s₂ s₃
    h_nulldiff : Eq (μ (SDiff.sdiff s₃ s₁)) 0
    ⊢ And (Eq (μ s₁) (μ s₂)) (Eq (μ s₂) (μ s₃))
  -/
  have le12 : μ s₁ ≤ μ s₂ := measure_mono h12
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s₁ s₂ s₃ : Set α
    h12 : HasSubset.Subset s₁ s₂
    h23 : HasSubset.Subset s₂ s₃
    h_nulldiff : Eq (μ (SDiff.sdiff s₃ s₁)) 0
    le12 : LE.le (μ s₁) (μ s₂)
    ⊢ And (Eq (μ s₁) (μ s₂)) (Eq (μ s₂) (μ s₃))
  -/
  have le23 : μ s₂ ≤ μ s₃ := measure_mono h23
  have key : μ s₃ ≤ μ s₁ :=
    calc
      μ s₃ = μ (s₃ \ s₁ ∪ s₁) := by rw [diff_union_of_subset (h12.trans h23)]
      _ ≤ μ (s₃ \ s₁) + μ s₁ := measure_union_le _ _
      _ = μ s₁ := by simp only [h_nulldiff, zero_add]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s₁ s₂ s₃ : Set α
    h12 : HasSubset.Subset s₁ s₂
    h23 : HasSubset.Subset s₂ s₃
    h_nulldiff : Eq (μ (SDiff.sdiff s₃ s₁)) 0
    le12 : LE.le (μ s₁) (μ s₂)
    le23 : LE.le (μ s₂) (μ s₃)
    key : LE.le (μ s₃) (μ s₁)
    ⊢ And (Eq (μ s₁) (μ s₂)) (Eq (μ s₂) (μ s₃))
  -/
  exact ⟨le12.antisymm (le23.trans key), le23.antisymm (key.trans le12)⟩
  /-
    🎉 no goals
  -/


theorem measure_eq_measure_smaller_of_between_null_diff {s₁ s₂ s₃ : Set α} (h12 : s₁ ⊆ s₂)
    (h23 : s₂ ⊆ s₃) (h_nulldiff : μ (s₃ \ s₁) = 0) : μ s₁ = μ s₂ :=
  (measure_eq_measure_of_between_null_diff h12 h23 h_nulldiff).1


theorem measure_eq_measure_larger_of_between_null_diff {s₁ s₂ s₃ : Set α} (h12 : s₁ ⊆ s₂)
    (h23 : s₂ ⊆ s₃) (h_nulldiff : μ (s₃ \ s₁) = 0) : μ s₂ = μ s₃ :=
  (measure_eq_measure_of_between_null_diff h12 h23 h_nulldiff).2


lemma measure_compl₀ (h : NullMeasurableSet s μ) (hs : μ s ≠ ∞) :
    μ sᶜ = μ Set.univ - μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    h : MeasureTheory.NullMeasurableSet s μ
    hs : Ne (μ s) Top.top
    ⊢ Eq (μ (HasCompl.compl s)) (HSub.hSub (μ Set.univ) (μ s))
  -/
  rw [← measure_add_measure_compl₀ h, ENNReal.add_sub_cancel_left hs]
  /-
    🎉 no goals
  -/


theorem measure_compl (h₁ : MeasurableSet s) (h_fin : μ s ≠ ∞) : μ sᶜ = μ univ - μ s :=
  measure_compl₀ h₁.nullMeasurableSet h_fin


lemma measure_inter_conull' (ht : μ (s \ t) = 0) : μ (s ∩ t) = μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    ht : Eq (μ (SDiff.sdiff s t)) 0
    ⊢ Eq (μ (Inter.inter s t)) (μ s)
  -/
  rw [← diff_compl, measure_diff_null']; rwa [← diff_eq]
                                         /-
                                           🎉 no goals
                                         -/


lemma measure_inter_conull (ht : μ tᶜ = 0) : μ (s ∩ t) = μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    ht : Eq (μ (HasCompl.compl t)) 0
    ⊢ Eq (μ (Inter.inter s t)) (μ s)
  -/
  rw [← diff_compl, measure_diff_null ht]
  /-
    🎉 no goals
  -/


@[simp]
theorem union_ae_eq_left_iff_ae_subset : (s ∪ t : Set α) =ᵐ[μ] s ↔ t ≤ᵐ[μ] s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (Union.union s t) s) ((MeasureTheory. …
  -/
  rw [ae_le_set]
  refine
    ⟨fun h => by simpa only [union_diff_left] using (ae_eq_set.mp h).1, fun h =>
      eventuallyLE_antisymm_iff.mpr
        ⟨by rwa [ae_le_set, union_diff_left],
          HasSubset.Subset.eventuallyLE subset_union_left⟩⟩


@[simp]
theorem union_ae_eq_right_iff_ae_subset : (s ∪ t : Set α) =ᵐ[μ] t ↔ s ≤ᵐ[μ] t := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (Union.union s t) t) ((MeasureTheory. …
  -/
  rw [union_comm, union_ae_eq_left_iff_ae_subset]
  /-
    🎉 no goals
  -/


theorem ae_eq_of_ae_subset_of_measure_ge (h₁ : s ≤ᵐ[μ] t) (h₂ : μ t ≤ μ s)
    (hsm : NullMeasurableSet s μ) (ht : μ t ≠ ∞) : s =ᵐ[μ] t := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h₁ : (MeasureTheory.ae μ).EventuallyLE s t
    h₂ : LE.le (μ t) (μ s)
    hsm : MeasureTheory.NullMeasurableSet s μ
    ht : Ne (μ t) Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq s t
  -/
  refine eventuallyLE_antisymm_iff.mpr ⟨h₁, ae_le_set.mpr ?_⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h₁ : (MeasureTheory.ae μ).EventuallyLE s t
    h₂ : LE.le (μ t) (μ s)
    hsm : MeasureTheory.NullMeasurableSet s μ
    ht : Ne (μ t) Top.top
    ⊢ Eq (μ (SDiff.sdiff t s)) 0
  -/
  replace h₂ : μ t = μ s := h₂.antisymm (measure_mono_ae h₁)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h₁ : (MeasureTheory.ae μ).EventuallyLE s t
    hsm : MeasureTheory.NullMeasurableSet s μ
    ht : Ne (μ t) Top.top
    h₂ : Eq (μ t) (μ s)
    ⊢ Eq (μ (SDiff.sdiff t s)) 0
  -/
  replace ht : μ s ≠ ∞ := h₂ ▸ ht
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h₁ : (MeasureTheory.ae μ).EventuallyLE s t
    hsm : MeasureTheory.NullMeasurableSet s μ
    h₂ : Eq (μ t) (μ s)
    ht : Ne (μ s) Top.top
    ⊢ Eq (μ (SDiff.sdiff t s)) 0
  -/
  rw [measure_diff' t hsm ht, measure_congr (union_ae_eq_left_iff_ae_subset.mpr h₁), h₂, tsub_self]
  /-
    🎉 no goals
  -/


/-- If `s ⊆ t`, `μ t ≤ μ s`, `μ t ≠ ∞`, and `s` is measurable, then `s =ᵐ[μ] t`. -/
theorem ae_eq_of_subset_of_measure_ge (h₁ : s ⊆ t) (h₂ : μ t ≤ μ s) (hsm : NullMeasurableSet s μ)
    (ht : μ t ≠ ∞) : s =ᵐ[μ] t :=
  ae_eq_of_ae_subset_of_measure_ge (HasSubset.Subset.eventuallyLE h₁) h₂ hsm ht


theorem measure_iUnion_congr_of_subset {ι : Sort*} [Countable ι] {s : ι → Set α} {t : ι → Set α}
    (hsub : ∀ i, s i ⊆ t i) (h_le : ∀ i, μ (t i) ≤ μ (s i)) : μ (⋃ i, s i) = μ (⋃ i, t i) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_8
    inst✝ : Countable ι
    s t : ι → Set α
    hsub : ∀ (i : ι), HasSubset.Subset (s i) (t i)
    h_le : ∀ (i : ι), LE.le (μ (t i)) (μ (s i))
    ⊢ Eq (μ (Set.iUnion fun i => s i)) (μ (Set.iUnion fun i => t i))
  -/
  refine le_antisymm (by gcongr; apply hsub) ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_8
    inst✝ : Countable ι
    s t : ι → Set α
    hsub : ∀ (i : ι), HasSubset.Subset (s i) (t i)
    h_le : ∀ (i : ι), LE.le (μ (t i)) (μ (s i))
    ⊢ LE.le (μ (Set.iUnion fun i => t i)) (μ (Set.iUnion fun i => s i))
  -/
  rcases Classical.em (∃ i, μ (t i) = ∞) with (⟨i, hi⟩ | htop)
  · calc
      μ (⋃ i, t i) ≤ ∞ := le_top
      _ ≤ μ (s i) := hi ▸ h_le i
      _ ≤ μ (⋃ i, s i) := measure_mono <| subset_iUnion _ _
  /-
    case inr
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_8
    inst✝ : Countable ι
    s t : ι → Set α
    hsub : ∀ (i : ι), HasSubset.Subset (s i) (t i)
    h_le : ∀ (i : ι), LE.le (μ (t i)) (μ (s i))
    htop : Not (Exists fun i => Eq (μ (t i)) Top.top)
    ⊢ LE.le (μ (Set.iUnion fun i => t i)) (μ (Set.iUnion fun i => s i))
  -/
  push_neg at htop
  /-
    case inr
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_8
    inst✝ : Countable ι
    s t : ι → Set α
    hsub : ∀ (i : ι), HasSubset.Subset (s i) (t i)
    h_le : ∀ (i : ι), LE.le (μ (t i)) (μ (s i))
    htop : ∀ (i : ι), Ne (μ (t i)) Top.top
    ⊢ LE.le (μ (Set.iUnion fun i => t i)) (μ (Set.iUnion fun i => s i))
  -/
  set M := toMeasurable μ
  have H : ∀ b, (M (t b) ∩ M (⋃ b, s b) : Set α) =ᵐ[μ] M (t b) := by
    refine fun b => ae_eq_of_subset_of_measure_ge inter_subset_left ?_ ?_ ?_
    · calc
        μ (M (t b)) = μ (t b) := measure_toMeasurable _
        _ ≤ μ (s b) := h_le b
        _ ≤ μ (M (t b) ∩ M (⋃ b, s b)) :=
          measure_mono <|
            subset_inter ((hsub b).trans <| subset_toMeasurable _ _)
              ((subset_iUnion _ _).trans <| subset_toMeasurable _ _)
    · measurability
    · rw [measure_toMeasurable]
      exact htop b
  calc
    μ (⋃ b, t b) ≤ μ (⋃ b, M (t b)) := measure_mono (iUnion_mono fun b => subset_toMeasurable _ _)
    _ = μ (⋃ b, M (t b) ∩ M (⋃ b, s b)) := measure_congr (EventuallyEq.countable_iUnion H).symm
    _ ≤ μ (M (⋃ b, s b)) := measure_mono (iUnion_subset fun b => inter_subset_right)
    _ = μ (⋃ b, s b) := measure_toMeasurable _


theorem measure_union_congr_of_subset {t₁ t₂ : Set α} (hs : s₁ ⊆ s₂) (hsμ : μ s₂ ≤ μ s₁)
    (ht : t₁ ⊆ t₂) (htμ : μ t₂ ≤ μ t₁) : μ (s₁ ∪ t₁) = μ (s₂ ∪ t₂) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s₁ s₂ t₁ t₂ : Set α
    hs : HasSubset.Subset s₁ s₂
    hsμ : LE.le (μ s₂) (μ s₁)
    ht : HasSubset.Subset t₁ t₂
    htμ : LE.le (μ t₂) (μ t₁)
    ⊢ Eq (μ (Union.union s₁ t₁)) (μ (Union.union s₂ t₂))
  -/
  rw [union_eq_iUnion, union_eq_iUnion]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s₁ s₂ t₁ t₂ : Set α
    hs : HasSubset.Subset s₁ s₂
    hsμ : LE.le (μ s₂) (μ s₁)
    ht : HasSubset.Subset t₁ t₂
    htμ : LE.le (μ t₂) (μ t₁)
    ⊢ Eq (μ (Set.iUnion fun b => cond b s₁ t₁)) (μ (Set.iUnion fun b => cond b s₂  …
  -/
  exact measure_iUnion_congr_of_subset (Bool.forall_bool.2 ⟨ht, hs⟩) (Bool.forall_bool.2 ⟨htμ, hsμ⟩)
  /-
    🎉 no goals
  -/


@[simp]
theorem measure_iUnion_toMeasurable {ι : Sort*} [Countable ι] (s : ι → Set α) :
    μ (⋃ i, toMeasurable μ (s i)) = μ (⋃ i, s i) :=
  Eq.symm <| measure_iUnion_congr_of_subset (fun _i => subset_toMeasurable _ _) fun _i ↦
    (measure_toMeasurable _).le


theorem measure_biUnion_toMeasurable {I : Set β} (hc : I.Countable) (s : β → Set α) :
    μ (⋃ b ∈ I, toMeasurable μ (s b)) = μ (⋃ b ∈ I, s b) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    I : Set β
    hc : I.Countable
    s : β → Set α
    ⊢ Eq (μ (Set.iUnion fun b => Set.iUnion fun h => MeasureTheory.toMeasurable μ  …
  -/
  haveI := hc.toEncodable
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    I : Set β
    hc : I.Countable
    s : β → Set α
    this : Encodable ↑I
    ⊢ Eq (μ (Set.iUnion fun b => Set.iUnion fun h => MeasureTheory.toMeasurable μ  …
  -/
  simp only [biUnion_eq_iUnion, measure_iUnion_toMeasurable]
  /-
    🎉 no goals
  -/


@[simp]
theorem measure_toMeasurable_union : μ (toMeasurable μ s ∪ t) = μ (s ∪ t) :=
  Eq.symm <|
    measure_union_congr_of_subset (subset_toMeasurable _ _) (measure_toMeasurable _).le Subset.rfl
      le_rfl


@[simp]
theorem measure_union_toMeasurable : μ (s ∪ toMeasurable μ t) = μ (s ∪ t) :=
  Eq.symm <|
    measure_union_congr_of_subset Subset.rfl le_rfl (subset_toMeasurable _ _)
      (measure_toMeasurable _).le


theorem sum_measure_le_measure_univ {s : Finset ι} {t : ι → Set α}
    (h : ∀ i ∈ s, NullMeasurableSet (t i) μ) (H : Set.Pairwise s (AEDisjoint μ on t)) :
    (∑ i ∈ s, μ (t i)) ≤ μ (univ : Set α) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    t : ι → Set α
    h : ∀ (i : ι), Membership.mem s i → MeasureTheory.NullMeasurableSet (t i) μ
    H : (↑s).Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) t)
    ⊢ LE.le (s.sum fun i => μ (t i)) (μ Set.univ)
  -/
  rw [← measure_biUnion_finset₀ H h]
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    t : ι → Set α
    h : ∀ (i : ι), Membership.mem s i → MeasureTheory.NullMeasurableSet (t i) μ
    H : (↑s).Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) t)
    ⊢ LE.le (μ (Set.iUnion fun b => Set.iUnion fun h => t b)) (μ Set.univ)
  -/
  exact measure_mono (subset_univ _)
  /-
    🎉 no goals
  -/


theorem tsum_measure_le_measure_univ {s : ι → Set α} (hs : ∀ i, NullMeasurableSet (s i) μ)
    (H : Pairwise (AEDisjoint μ on s)) : ∑' i, μ (s i) ≤ μ (univ : Set α) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    H : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
    ⊢ LE.le (tsum fun i => μ (s i)) (μ Set.univ)
  -/
  rw [ENNReal.tsum_eq_iSup_sum]
  exact iSup_le fun s =>
    sum_measure_le_measure_univ (fun i _hi => hs i) fun i _hi j _hj hij => H hij


/-- Pigeonhole principle for measure spaces: if `∑' i, μ (s i) > μ univ`, then
one of the intersections `s i ∩ s j` is not empty. -/
theorem exists_nonempty_inter_of_measure_univ_lt_tsum_measure {m : MeasurableSpace α}
    (μ : Measure α) {s : ι → Set α} (hs : ∀ i, NullMeasurableSet (s i) μ)
    (H : μ (univ : Set α) < ∑' i, μ (s i)) : ∃ i j, i ≠ j ∧ (s i ∩ s j).Nonempty := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    H : LT.lt (μ Set.univ) (tsum fun i => μ (s i))
    ⊢ Exists fun i => Exists fun j => And (Ne i j) (Inter.inter (s i) (s j)).Nonem …
  -/
  contrapose! H
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    H : ∀ (i j : ι), Ne i j → Eq (Inter.inter (s i) (s j)) EmptyCollection.emptyCo …
    ⊢ LE.le (tsum fun i => μ (s i)) (μ Set.univ)
  -/
  apply tsum_measure_le_measure_univ hs
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    H : ∀ (i j : ι), Ne i j → Eq (Inter.inter (s i) (s j)) EmptyCollection.emptyCo …
    ⊢ Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
  -/
  intro i j hij
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    H : ∀ (i j : ι), Ne i j → Eq (Inter.inter (s i) (s j)) EmptyCollection.emptyCo …
    i j : ι
    hij : Ne i j
    ⊢ Function.onFun (MeasureTheory.AEDisjoint μ) s i j
  -/
  exact (disjoint_iff_inter_eq_empty.mpr (H i j hij)).aedisjoint
  /-
    🎉 no goals
  -/


/-- Pigeonhole principle for measure spaces: if `s` is a `Finset` and
`∑ i ∈ s, μ (t i) > μ univ`, then one of the intersections `t i ∩ t j` is not empty. -/
theorem exists_nonempty_inter_of_measure_univ_lt_sum_measure {m : MeasurableSpace α} (μ : Measure α)
    {s : Finset ι} {t : ι → Set α} (h : ∀ i ∈ s, NullMeasurableSet (t i) μ)
    (H : μ (univ : Set α) < ∑ i ∈ s, μ (t i)) :
    ∃ i ∈ s, ∃ j ∈ s, ∃ _h : i ≠ j, (t i ∩ t j).Nonempty := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    t : ι → Set α
    h : ∀ (i : ι), Membership.mem s i → MeasureTheory.NullMeasurableSet (t i) μ
    H : LT.lt (μ Set.univ) (s.sum fun i => μ (t i))
    ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
  -/
  contrapose! H
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    t : ι → Set α
    h : ∀ (i : ι), Membership.mem s i → MeasureTheory.NullMeasurableSet (t i) μ
    H : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i j → E …
    ⊢ LE.le (s.sum fun i => μ (t i)) (μ Set.univ)
  -/
  apply sum_measure_le_measure_univ h
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    t : ι → Set α
    h : ∀ (i : ι), Membership.mem s i → MeasureTheory.NullMeasurableSet (t i) μ
    H : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i j → E …
    ⊢ (↑s).Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) t)
  -/
  intro i hi j hj hij
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset ι
    t : ι → Set α
    h : ∀ (i : ι), Membership.mem s i → MeasureTheory.NullMeasurableSet (t i) μ
    H : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i j → E …
    i : ι
    hi : Membership.mem (↑s) i
    j : ι
    hj : Membership.mem (↑s) j
    hij : Ne i j
    ⊢ Function.onFun (MeasureTheory.AEDisjoint μ) t i j
  -/
  exact (disjoint_iff_inter_eq_empty.mpr (H i hi j hj hij)).aedisjoint
  /-
    🎉 no goals
  -/


/-- If two sets `s` and `t` are included in a set `u`, and `μ s + μ t > μ u`,
then `s` intersects `t`. Version assuming that `t` is measurable. -/
theorem nonempty_inter_of_measure_lt_add {m : MeasurableSpace α} (μ : Measure α) {s t u : Set α}
    (ht : MeasurableSet t) (h's : s ⊆ u) (h't : t ⊆ u) (h : μ u < μ s + μ t) :
    (s ∩ t).Nonempty := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    ht : MeasurableSet t
    h's : HasSubset.Subset s u
    h't : HasSubset.Subset t u
    h : LT.lt (μ u) (HAdd.hAdd (μ s) (μ t))
    ⊢ (Inter.inter s t).Nonempty
  -/
  rw [← Set.not_disjoint_iff_nonempty_inter]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    ht : MeasurableSet t
    h's : HasSubset.Subset s u
    h't : HasSubset.Subset t u
    h : LT.lt (μ u) (HAdd.hAdd (μ s) (μ t))
    ⊢ Not (Disjoint s t)
  -/
  contrapose! h
  calc
    μ s + μ t = μ (s ∪ t) := (measure_union h ht).symm
    _ ≤ μ u := measure_mono (union_subset h's h't)


/-- If two sets `s` and `t` are included in a set `u`, and `μ s + μ t > μ u`,
then `s` intersects `t`. Version assuming that `s` is measurable. -/
theorem nonempty_inter_of_measure_lt_add' {m : MeasurableSpace α} (μ : Measure α) {s t u : Set α}
    (hs : MeasurableSet s) (h's : s ⊆ u) (h't : t ⊆ u) (h : μ u < μ s + μ t) :
    (s ∩ t).Nonempty := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    hs : MeasurableSet s
    h's : HasSubset.Subset s u
    h't : HasSubset.Subset t u
    h : LT.lt (μ u) (HAdd.hAdd (μ s) (μ t))
    ⊢ (Inter.inter s t).Nonempty
  -/
  rw [add_comm] at h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    hs : MeasurableSet s
    h's : HasSubset.Subset s u
    h't : HasSubset.Subset t u
    h : LT.lt (μ u) (HAdd.hAdd (μ t) (μ s))
    ⊢ (Inter.inter s t).Nonempty
  -/
  rw [inter_comm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    hs : MeasurableSet s
    h's : HasSubset.Subset s u
    h't : HasSubset.Subset t u
    h : LT.lt (μ u) (HAdd.hAdd (μ t) (μ s))
    ⊢ (Inter.inter t s).Nonempty
  -/
  exact nonempty_inter_of_measure_lt_add μ hs h't h's h
  /-
    🎉 no goals
  -/


/-- Continuity from below:
the measure of the union of a directed sequence of (not necessarily measurable) sets
is the supremum of the measures. -/
theorem _root_.Directed.measure_iUnion [Countable ι] {s : ι → Set α} (hd : Directed (· ⊆ ·) s) :
    μ (⋃ i, s i) = ⨆ i, μ (s i) := by
  -- WLOG, `ι = ℕ`
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    ⊢ Eq (μ (Set.iUnion fun i => s i)) (iSup fun i => μ (s i))
  -/
  rcases Countable.exists_injective_nat ι with ⟨e, he⟩
  /-
    case intro
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    e : ι → Nat
    he : Function.Injective e
    ⊢ Eq (μ (Set.iUnion fun i => s i)) (iSup fun i => μ (s i))
  -/
  generalize ht : Function.extend e s ⊥ = t
  /-
    case intro
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    e : ι → Nat
    he : Function.Injective e
    t : Nat → Set α
    ht : Eq (Function.extend e s Bot.bot) t
    ⊢ Eq (μ (Set.iUnion fun i => s i)) (iSup fun i => μ (s i))
  -/
  replace hd : Directed (· ⊆ ·) t := ht ▸ hd.extend_bot he
  suffices μ (⋃ n, t n) = ⨆ n, μ (t n) by
    simp only [← ht, Function.apply_extend μ, ← iSup_eq_iUnion, iSup_extend_bot he,
      Function.comp_def, Pi.bot_apply, bot_eq_empty, measure_empty] at this
    exact this.trans (iSup_extend_bot he _)
  /-
    case intro
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    e : ι → Nat
    he : Function.Injective e
    t : Nat → Set α
    ht : Eq (Function.extend e s Bot.bot) t
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) t
    ⊢ Eq (μ (Set.iUnion fun n => t n)) (iSup fun n => μ (t n))
  -/
  clear! ι
  -- The `≥` inequality is trivial
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Nat → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) t
    ⊢ Eq (μ (Set.iUnion fun n => t n)) (iSup fun n => μ (t n))
  -/
  refine le_antisymm ?_ (iSup_le fun i ↦ measure_mono <| subset_iUnion _ _)
  -- Choose `T n ⊇ t n` of the same measure, put `Td n = disjointed T`
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Nat → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) t
    ⊢ LE.le (μ (Set.iUnion fun n => t n)) (iSup fun n => μ (t n))
  -/
  set T : ℕ → Set α := fun n => toMeasurable μ (t n)
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Nat → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) t
    T : Nat → Set α := fun n => MeasureTheory.toMeasurable μ (t n)
    ⊢ LE.le (μ (Set.iUnion fun n => t n)) (iSup fun n => μ (t n))
  -/
  set Td : ℕ → Set α := disjointed T
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Nat → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) t
    T : Nat → Set α := fun n => MeasureTheory.toMeasurable μ (t n)
    Td : Nat → Set α := disjointed T
    ⊢ LE.le (μ (Set.iUnion fun n => t n)) (iSup fun n => μ (t n))
  -/
  have hm : ∀ n, MeasurableSet (Td n) := .disjointed fun n ↦ measurableSet_toMeasurable _ _
  calc
    μ (⋃ n, t n) = μ (⋃ n, Td n) := by rw [iUnion_disjointed, measure_iUnion_toMeasurable]
    _ ≤ ∑' n, μ (Td n) := measure_iUnion_le _
    _ = ⨆ I : Finset ℕ, ∑ n ∈ I, μ (Td n) := ENNReal.tsum_eq_iSup_sum
    _ ≤ ⨆ n, μ (t n) := iSup_le fun I => by
      rcases hd.finset_le I with ⟨N, hN⟩
      calc
        (∑ n ∈ I, μ (Td n)) = μ (⋃ n ∈ I, Td n) :=
          (measure_biUnion_finset ((disjoint_disjointed T).set_pairwise I) fun n _ => hm n).symm
        _ ≤ μ (⋃ n ∈ I, T n) := measure_mono (iUnion₂_mono fun n _hn => disjointed_subset _ _)
        _ = μ (⋃ n ∈ I, t n) := measure_biUnion_toMeasurable I.countable_toSet _
        _ ≤ μ (t N) := measure_mono (iUnion₂_subset hN)
        _ ≤ ⨆ n, μ (t n) := le_iSup (μ ∘ t) N


@[deprecated (since := "2024-09-01")] alias measure_iUnion_eq_iSup := Directed.measure_iUnion


/-- Continuity from below:
the measure of the union of a monotone family of sets is equal to the supremum of their measures.
The theorem assumes that the `atTop` filter on the index set is countably generated,
so it works for a family indexed by a countable type, as well as `ℝ`.  -/
theorem _root_.Monotone.measure_iUnion [Preorder ι] [IsDirected ι (· ≤ ·)]
    [(atTop : Filter ι).IsCountablyGenerated] {s : ι → Set α} (hs : Monotone s) :
    μ (⋃ i, s i) = ⨆ i, μ (s i) := by
  cases isEmpty_or_nonempty ι with
  | inl _ => simp
  | inr _ =>
    rcases exists_seq_monotone_tendsto_atTop_atTop ι with ⟨x, hxm, hx⟩
    rw [← hs.iUnion_comp_tendsto_atTop hx, ← Monotone.iSup_comp_tendsto_atTop _ hx]
    exacts [(hs.comp hxm).directed_le.measure_iUnion, fun _ _ h ↦ measure_mono (hs h)]


theorem _root_.Antitone.measure_iUnion [Preorder ι] [IsDirected ι (· ≥ ·)]
    [(atBot : Filter ι).IsCountablyGenerated] {s : ι → Set α} (hs : Antitone s) :
    μ (⋃ i, s i) = ⨆ i, μ (s i) :=
  hs.dual_left.measure_iUnion


/-- Continuity from below: the measure of the union of a sequence of
(not necessarily measurable) sets is the supremum of the measures of the partial unions. -/
theorem measure_iUnion_eq_iSup_accumulate [Preorder ι] [IsDirected ι (· ≤ ·)]
    [(atTop : Filter ι).IsCountablyGenerated] {f : ι → Set α} :
    μ (⋃ i, f i) = ⨆ i, μ (Accumulate f i) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Preorder ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    f : ι → Set α
    ⊢ Eq (μ (Set.iUnion fun i => f i)) (iSup fun i => μ (Set.Accumulate f i))
  -/
  rw [← iUnion_accumulate]
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Preorder ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    f : ι → Set α
    ⊢ Eq (μ (Set.iUnion fun x => Set.Accumulate f x)) (iSup fun i => μ (Set.Accumu …
  -/
  exact monotone_accumulate.measure_iUnion
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-01")]
alias measure_iUnion_eq_iSup' := measure_iUnion_eq_iSup_accumulate


theorem measure_biUnion_eq_iSup {s : ι → Set α} {t : Set ι} (ht : t.Countable)
    (hd : DirectedOn ((· ⊆ ·) on s) t) : μ (⋃ i ∈ t, s i) = ⨆ i ∈ t, μ (s i) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    t : Set ι
    ht : t.Countable
    hd : DirectedOn (Function.onFun (fun x1 x2 => HasSubset.Subset x1 x2) s) t
    ⊢ Eq (μ (Set.iUnion fun i => Set.iUnion fun h => s i)) (iSup fun i => iSup fun …
  -/
  haveI := ht.to_subtype
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    t : Set ι
    ht : t.Countable
    hd : DirectedOn (Function.onFun (fun x1 x2 => HasSubset.Subset x1 x2) s) t
    this : Countable ↑t
    ⊢ Eq (μ (Set.iUnion fun i => Set.iUnion fun h => s i)) (iSup fun i => iSup fun …
  -/
  rw [biUnion_eq_iUnion, hd.directed_val.measure_iUnion, ← iSup_subtype'']
  /-
    🎉 no goals
  -/


/-- **Continuity from above**:
the measure of the intersection of a directed downwards countable family of measurable sets
is the infimum of the measures. -/
theorem _root_.Directed.measure_iInter [Countable ι] {s : ι → Set α}
    (h : ∀ i, NullMeasurableSet (s i) μ) (hd : Directed (· ⊇ ·) s) (hfin : ∃ i, μ (s i) ≠ ∞) :
    μ (⋂ i, s i) = ⨅ i, μ (s i) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hd : Directed (fun x1 x2 => Superset x1 x2) s
    hfin : Exists fun i => Ne (μ (s i)) Top.top
    ⊢ Eq (μ (Set.iInter fun i => s i)) (iInf fun i => μ (s i))
  -/
  rcases hfin with ⟨k, hk⟩
  /-
    case intro
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hd : Directed (fun x1 x2 => Superset x1 x2) s
    k : ι
    hk : Ne (μ (s k)) Top.top
    ⊢ Eq (μ (Set.iInter fun i => s i)) (iInf fun i => μ (s i))
  -/
  have : ∀ t ⊆ s k, μ t ≠ ∞ := fun t ht => ne_top_of_le_ne_top hk (measure_mono ht)
  rw [← ENNReal.sub_sub_cancel hk (iInf_le (fun i => μ (s i)) k), ENNReal.sub_iInf, ←
    ENNReal.sub_sub_cancel hk (measure_mono (iInter_subset _ k)), ←
    measure_diff (iInter_subset _ k) (.iInter h) (this _ (iInter_subset _ k)),
    diff_iInter, Directed.measure_iUnion]
    /-
      case intro
      α : Type u_1
      ι : Type u_5
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
      hd : Directed (fun x1 x2 => Superset x1 x2) s
      k : ι
      hk : Ne (μ (s k)) Top.top
      this : ∀ (t : Set α), HasSubset.Subset t (s k) → Ne (μ t) Top.top
      ⊢ Eq (HSub.hSub (μ (s k)) (iSup fun i => μ (SDiff.sdiff (s k) (s i)))) (HSub.h …
    -/
  · congr 1
    /-
      case intro.e_a
      α : Type u_1
      ι : Type u_5
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
      hd : Directed (fun x1 x2 => Superset x1 x2) s
      k : ι
      hk : Ne (μ (s k)) Top.top
      this : ∀ (t : Set α), HasSubset.Subset t (s k) → Ne (μ t) Top.top
      ⊢ Eq (iSup fun i => μ (SDiff.sdiff (s k) (s i))) (iSup fun i => HSub.hSub (μ ( …
    -/
    refine le_antisymm (iSup_mono' fun i => ?_) (iSup_mono fun i => le_measure_diff)
    /-
      case intro.e_a
      α : Type u_1
      ι : Type u_5
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
      hd : Directed (fun x1 x2 => Superset x1 x2) s
      k : ι
      hk : Ne (μ (s k)) Top.top
      this : ∀ (t : Set α), HasSubset.Subset t (s k) → Ne (μ t) Top.top
      i : ι
      ⊢ Exists fun i' => LE.le (μ (SDiff.sdiff (s k) (s i))) (HSub.hSub (μ (s k)) (μ …
    -/
    rcases hd i k with ⟨j, hji, hjk⟩
    /-
      case intro.e_a.intro.intro
      α : Type u_1
      ι : Type u_5
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
      hd : Directed (fun x1 x2 => Superset x1 x2) s
      k : ι
      hk : Ne (μ (s k)) Top.top
      this : ∀ (t : Set α), HasSubset.Subset t (s k) → Ne (μ t) Top.top
      i j : ι
      hji : Superset (s i) (s j)
      hjk : Superset (s k) (s j)
      ⊢ Exists fun i' => LE.le (μ (SDiff.sdiff (s k) (s i))) (HSub.hSub (μ (s k)) (μ …
    -/
    use j
    /-
      case h
      α : Type u_1
      ι : Type u_5
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
      hd : Directed (fun x1 x2 => Superset x1 x2) s
      k : ι
      hk : Ne (μ (s k)) Top.top
      this : ∀ (t : Set α), HasSubset.Subset t (s k) → Ne (μ t) Top.top
      i j : ι
      hji : Superset (s i) (s j)
      hjk : Superset (s k) (s j)
      ⊢ LE.le (μ (SDiff.sdiff (s k) (s i))) (HSub.hSub (μ (s k)) (μ (s j)))
    -/
    rw [← measure_diff hjk (h _) (this _ hjk)]
    /-
      case h
      α : Type u_1
      ι : Type u_5
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
      hd : Directed (fun x1 x2 => Superset x1 x2) s
      k : ι
      hk : Ne (μ (s k)) Top.top
      this : ∀ (t : Set α), HasSubset.Subset t (s k) → Ne (μ t) Top.top
      i j : ι
      hji : Superset (s i) (s j)
      hjk : Superset (s k) (s j)
      ⊢ LE.le (μ (SDiff.sdiff (s k) (s i))) (μ (SDiff.sdiff (s k) (s j)))
    -/
    gcongr
    /-
      🎉 no goals
    -/
    /-
      case intro
      α : Type u_1
      ι : Type u_5
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable ι
      s : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
      hd : Directed (fun x1 x2 => Superset x1 x2) s
      k : ι
      hk : Ne (μ (s k)) Top.top
      this : ∀ (t : Set α), HasSubset.Subset t (s k) → Ne (μ t) Top.top
      ⊢ Directed (fun x1 x2 => HasSubset.Subset x1 x2) fun i => SDiff.sdiff (s k) (s …
    -/
  · exact hd.mono_comp _ fun _ _ => diff_subset_diff_right
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-30")] alias measure_iInter_eq_iInf := Directed.measure_iInter


/-- **Continuity from above**:
the measure of the intersection of a monotone family of measurable sets
indexed by a type with countably generated `atBot` filter
is equal to the infimum of the measures. -/
theorem _root_.Monotone.measure_iInter [Preorder ι] [IsDirected ι (· ≥ ·)]
    [(atBot : Filter ι).IsCountablyGenerated] {s : ι → Set α} (hs : Monotone s)
    (hsm : ∀ i, NullMeasurableSet (s i) μ) (hfin : ∃ i, μ (s i) ≠ ∞) :
    μ (⋂ i, s i) = ⨅ i, μ (s i) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Preorder ι
    inst✝¹ : IsDirected ι fun x1 x2 => GE.ge x1 x2
    inst✝ : Filter.atBot.IsCountablyGenerated
    s : ι → Set α
    hs : Monotone s
    hsm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hfin : Exists fun i => Ne (μ (s i)) Top.top
    ⊢ Eq (μ (Set.iInter fun i => s i)) (iInf fun i => μ (s i))
  -/
  refine le_antisymm (le_iInf fun i ↦ measure_mono <| iInter_subset _ _) ?_
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Preorder ι
    inst✝¹ : IsDirected ι fun x1 x2 => GE.ge x1 x2
    inst✝ : Filter.atBot.IsCountablyGenerated
    s : ι → Set α
    hs : Monotone s
    hsm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hfin : Exists fun i => Ne (μ (s i)) Top.top
    ⊢ LE.le (iInf fun i => μ (s i)) (μ (Set.iInter fun i => s i))
  -/
  have := hfin.nonempty
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Preorder ι
    inst✝¹ : IsDirected ι fun x1 x2 => GE.ge x1 x2
    inst✝ : Filter.atBot.IsCountablyGenerated
    s : ι → Set α
    hs : Monotone s
    hsm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hfin : Exists fun i => Ne (μ (s i)) Top.top
    this : Nonempty ι
    ⊢ LE.le (iInf fun i => μ (s i)) (μ (Set.iInter fun i => s i))
  -/
  rcases exists_seq_antitone_tendsto_atTop_atBot ι with ⟨x, hxm, hx⟩
  calc
    ⨅ i, μ (s i) ≤ ⨅ n, μ (s (x n)) := le_iInf_comp (μ ∘ s) x
    _ = μ (⋂ n, s (x n)) := by
      refine .symm <| (hs.comp_antitone hxm).directed_ge.measure_iInter (fun n ↦ hsm _) ?_
      rcases hfin with ⟨k, hk⟩
      rcases (hx.eventually_le_atBot k).exists with ⟨n, hn⟩
      exact ⟨n, ne_top_of_le_ne_top hk <| measure_mono <| hs hn⟩
    _ ≤ μ (⋂ i, s i) := by
      refine measure_mono <| iInter_mono' fun i ↦ ?_
      rcases (hx.eventually_le_atBot i).exists with ⟨n, hn⟩
      exact ⟨n, hs hn⟩


/-- **Continuity from above**:
the measure of the intersection of an antitone family of measurable sets
indexed by a type with countably generated `atTop` filter
is equal to the infimum of the measures. -/
theorem _root_.Antitone.measure_iInter [Preorder ι] [IsDirected ι (· ≤ ·)]
    [(atTop : Filter ι).IsCountablyGenerated] {s : ι → Set α} (hs : Antitone s)
    (hsm : ∀ i, NullMeasurableSet (s i) μ) (hfin : ∃ i, μ (s i) ≠ ∞) :
    μ (⋂ i, s i) = ⨅ i, μ (s i) :=
  hs.dual_left.measure_iInter hsm hfin


/-- Continuity from above: the measure of the intersection of a sequence of
measurable sets is the infimum of the measures of the partial intersections. -/
theorem measure_iInter_eq_iInf_measure_iInter_le {α ι : Type*} {_ : MeasurableSpace α}
    {μ : Measure α} [Countable ι] [Preorder ι] [IsDirected ι (· ≤ ·)]
    {f : ι → Set α} (h : ∀ i, NullMeasurableSet (f i) μ) (hfin : ∃ i, μ (f i) ≠ ∞) :
    μ (⋂ i, f i) = ⨅ i, μ (⋂ j ≤ i, f j) := by
  /-
    α : Type u_8
    ι : Type u_9
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Countable ι
    inst✝¹ : Preorder ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    f : ι → Set α
    h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
    hfin : Exists fun i => Ne (μ (f i)) Top.top
    ⊢ Eq (μ (Set.iInter fun i => f i)) (iInf fun i => μ (Set.iInter fun j => Set.i …
  -/
  rw [← Antitone.measure_iInter]
    /-
      α : Type u_8
      ι : Type u_9
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : Countable ι
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
      hfin : Exists fun i => Ne (μ (f i)) Top.top
      ⊢ Eq (μ (Set.iInter fun i => f i)) (μ (Set.iInter fun i => Set.iInter fun j => …
    -/
  · rw [iInter_comm]
    /-
      α : Type u_8
      ι : Type u_9
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : Countable ι
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
      hfin : Exists fun i => Ne (μ (f i)) Top.top
      ⊢ Eq (μ (Set.iInter fun i => f i)) (μ (Set.iInter fun i' => Set.iInter fun i = …
    -/
    exact congrArg μ <| iInter_congr fun i ↦ (biInf_const nonempty_Ici).symm
    /-
      🎉 no goals
    -/
    /-
      case hs
      α : Type u_8
      ι : Type u_9
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : Countable ι
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
      hfin : Exists fun i => Ne (μ (f i)) Top.top
      ⊢ Antitone fun i => Set.iInter fun j => Set.iInter fun h => f j
    -/
  · exact fun i j h ↦ biInter_mono (Iic_subset_Iic.2 h) fun _ _ ↦ Set.Subset.rfl
    /-
      🎉 no goals
    -/
    /-
      case hsm
      α : Type u_8
      ι : Type u_9
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : Countable ι
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
      hfin : Exists fun i => Ne (μ (f i)) Top.top
      ⊢ ∀ (i : ι), MeasureTheory.NullMeasurableSet (Set.iInter fun j => Set.iInter f …
    -/
  · exact fun i ↦ .biInter (to_countable _) fun _ _ ↦ h _
    /-
      🎉 no goals
    -/
    /-
      case hfin
      α : Type u_8
      ι : Type u_9
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : Countable ι
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
      hfin : Exists fun i => Ne (μ (f i)) Top.top
      ⊢ Exists fun i => Ne (μ (Set.iInter fun j => Set.iInter fun h => f j)) Top.top
    -/
  · refine hfin.imp fun k hk ↦ ne_top_of_le_ne_top hk <| measure_mono <| iInter₂_subset k ?_
    /-
      case hfin
      α : Type u_8
      ι : Type u_9
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : Countable ι
      inst✝¹ : Preorder ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      f : ι → Set α
      h : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
      hfin : Exists fun i => Ne (μ (f i)) Top.top
      k : ι
      hk : Ne (μ (f k)) Top.top
      ⊢ LE.le k k
    -/
    rfl
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-30")]
alias measure_iInter_eq_iInf' := measure_iInter_eq_iInf_measure_iInter_le


/-- Continuity from below: the measure of the union of an increasing sequence of (not necessarily
measurable) sets is the limit of the measures. -/
theorem tendsto_measure_iUnion_atTop [Preorder ι] [IsCountablyGenerated (atTop : Filter ι)]
    {s : ι → Set α} (hm : Monotone s) : Tendsto (μ ∘ s) atTop (𝓝 (μ (⋃ n, s n))) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hm : Monotone s
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) Filter.atTop (nhds (μ (Set.iUnion fun  …
  -/
  refine .of_neBot_imp fun h ↦ ?_
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hm : Monotone s
    h : Filter.atTop.NeBot
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) Filter.atTop (nhds (μ (Set.iUnion fun  …
  -/
  have := (atTop_neBot_iff.1 h).2
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hm : Monotone s
    h : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) Filter.atTop (nhds (μ (Set.iUnion fun  …
  -/
  rw [hm.measure_iUnion]
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hm : Monotone s
    h : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) Filter.atTop (nhds (iSup fun i => μ (s …
  -/
  exact tendsto_atTop_iSup fun n m hnm => measure_mono <| hm hnm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-01")] alias tendsto_measure_iUnion := tendsto_measure_iUnion_atTop


theorem tendsto_measure_iUnion_atBot [Preorder ι] [IsCountablyGenerated (atBot : Filter ι)]
    {s : ι → Set α} (hm : Antitone s) : Tendsto (μ ∘ s) atBot (𝓝 (μ (⋃ n, s n))) :=
  tendsto_measure_iUnion_atTop (ι := ιᵒᵈ) hm.dual_left


/-- Continuity from below: the measure of the union of a sequence of (not necessarily measurable)
sets is the limit of the measures of the partial unions. -/
theorem tendsto_measure_iUnion_accumulate {α ι : Type*}
    [Preorder ι] [IsCountablyGenerated (atTop : Filter ι)]
    {_ : MeasurableSpace α} {μ : Measure α} {f : ι → Set α} :
    Tendsto (fun i ↦ μ (Accumulate f i)) atTop (𝓝 (μ (⋃ i, f i))) := by
  /-
    α : Type u_8
    ι : Type u_9
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : ι → Set α
    ⊢ Filter.Tendsto (fun i => μ (Set.Accumulate f i)) Filter.atTop (nhds (μ (Set. …
  -/
  refine .of_neBot_imp fun h ↦ ?_
  /-
    α : Type u_8
    ι : Type u_9
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : ι → Set α
    h : Filter.atTop.NeBot
    ⊢ Filter.Tendsto (fun i => μ (Set.Accumulate f i)) Filter.atTop (nhds (μ (Set. …
  -/
  have := (atTop_neBot_iff.1 h).2
  /-
    α : Type u_8
    ι : Type u_9
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : ι → Set α
    h : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (fun i => μ (Set.Accumulate f i)) Filter.atTop (nhds (μ (Set. …
  -/
  rw [measure_iUnion_eq_iSup_accumulate]
  /-
    α : Type u_8
    ι : Type u_9
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : ι → Set α
    h : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (fun i => μ (Set.Accumulate f i)) Filter.atTop (nhds (iSup fu …
  -/
  exact tendsto_atTop_iSup fun i j hij ↦ by gcongr
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-01")]
alias tendsto_measure_iUnion' := tendsto_measure_iUnion_accumulate


/-- Continuity from above: the measure of the intersection of a decreasing sequence of measurable
sets is the limit of the measures. -/
theorem tendsto_measure_iInter_atTop [Preorder ι]
    [IsCountablyGenerated (atTop : Filter ι)] {s : ι → Set α}
    (hs : ∀ i, NullMeasurableSet (s i) μ) (hm : Antitone s) (hf : ∃ i, μ (s i) ≠ ∞) :
    Tendsto (μ ∘ s) atTop (𝓝 (μ (⋂ n, s n))) := by
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hm : Antitone s
    hf : Exists fun i => Ne (μ (s i)) Top.top
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) Filter.atTop (nhds (μ (Set.iInter fun  …
  -/
  refine .of_neBot_imp fun h ↦ ?_
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hm : Antitone s
    hf : Exists fun i => Ne (μ (s i)) Top.top
    h : Filter.atTop.NeBot
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) Filter.atTop (nhds (μ (Set.iInter fun  …
  -/
  have := (atTop_neBot_iff.1 h).2
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hm : Antitone s
    hf : Exists fun i => Ne (μ (s i)) Top.top
    h : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) Filter.atTop (nhds (μ (Set.iInter fun  …
  -/
  rw [hm.measure_iInter hs hf]
  /-
    α : Type u_1
    ι : Type u_5
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hs : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hm : Antitone s
    hf : Exists fun i => Ne (μ (s i)) Top.top
    h : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) Filter.atTop (nhds (iInf fun i => μ (s …
  -/
  exact tendsto_atTop_iInf fun n m hnm => measure_mono <| hm hnm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-30")]
alias tendsto_measure_iInter := tendsto_measure_iInter_atTop


/-- Continuity from above: the measure of the intersection of an increasing sequence of measurable
sets is the limit of the measures. -/
theorem tendsto_measure_iInter_atBot [Preorder ι] [IsCountablyGenerated (atBot : Filter ι)]
    {s : ι → Set α} (hs : ∀ i, NullMeasurableSet (s i) μ) (hm : Monotone s)
    (hf : ∃ i, μ (s i) ≠ ∞) : Tendsto (μ ∘ s) atBot (𝓝 (μ (⋂ n, s n))) :=
  tendsto_measure_iInter_atTop (ι := ιᵒᵈ) hs hm.dual_left hf


/-- Continuity from above: the measure of the intersection of a sequence of measurable
sets such that one has finite measure is the limit of the measures of the partial intersections. -/
theorem tendsto_measure_iInter_le {α ι : Type*} {_ : MeasurableSpace α} {μ : Measure α}
    [Countable ι] [Preorder ι] {f : ι → Set α} (hm : ∀ i, NullMeasurableSet (f i) μ)
    (hf : ∃ i, μ (f i) ≠ ∞) :
    Tendsto (fun i ↦ μ (⋂ j ≤ i, f j)) atTop (𝓝 (μ (⋂ i, f i))) := by
  /-
    α : Type u_8
    ι : Type u_9
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable ι
    inst✝ : Preorder ι
    f : ι → Set α
    hm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
    hf : Exists fun i => Ne (μ (f i)) Top.top
    ⊢ Filter.Tendsto (fun i => μ (Set.iInter fun j => Set.iInter fun h => f j)) Fi …
  -/
  refine .of_neBot_imp fun hne ↦ ?_
  /-
    α : Type u_8
    ι : Type u_9
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable ι
    inst✝ : Preorder ι
    f : ι → Set α
    hm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
    hf : Exists fun i => Ne (μ (f i)) Top.top
    hne : Filter.atTop.NeBot
    ⊢ Filter.Tendsto (fun i => μ (Set.iInter fun j => Set.iInter fun h => f j)) Fi …
  -/
  cases' atTop_neBot_iff.mp hne
  /-
    case intro
    α : Type u_8
    ι : Type u_9
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable ι
    inst✝ : Preorder ι
    f : ι → Set α
    hm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (f i) μ
    hf : Exists fun i => Ne (μ (f i)) Top.top
    hne : Filter.atTop.NeBot
    left✝ : Nonempty ι
    right✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (fun i => μ (Set.iInter fun j => Set.iInter fun h => f j)) Fi …
  -/
  rw [measure_iInter_eq_iInf_measure_iInter_le hm hf]
  exact tendsto_atTop_iInf
    fun i j hij ↦ measure_mono <| biInter_subset_biInter_left fun k hki ↦ le_trans hki hij


/-- The measure of the intersection of a decreasing sequence of measurable
sets indexed by a linear order with first countable topology is the limit of the measures. -/
theorem tendsto_measure_biInter_gt {ι : Type*} [LinearOrder ι] [TopologicalSpace ι]
    [OrderTopology ι] [DenselyOrdered ι] [FirstCountableTopology ι] {s : ι → Set α}
    {a : ι} (hs : ∀ r > a, NullMeasurableSet (s r) μ) (hm : ∀ i j, a < i → i ≤ j → s i ⊆ s j)
    (hf : ∃ r > a, μ (s r) ≠ ∞) : Tendsto (μ ∘ s) (𝓝[Ioi a] a) (𝓝 (μ (⋂ r > a, s r))) := by
  have : (atBot : Filter (Ioi a)).IsCountablyGenerated := by
    rw [← comap_coe_Ioi_nhdsGT]
    infer_instance
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_8
    inst✝⁴ : LinearOrder ι
    inst✝³ : TopologicalSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : DenselyOrdered ι
    inst✝ : FirstCountableTopology ι
    s : ι → Set α
    a : ι
    hs : ∀ (r : ι), GT.gt r a → MeasureTheory.NullMeasurableSet (s r) μ
    hm : ∀ (i j : ι), LT.lt a i → LE.le i j → HasSubset.Subset (s i) (s j)
    hf : Exists fun r => And (GT.gt r a) (Ne (μ (s r)) Top.top)
    this : Filter.atBot.IsCountablyGenerated
    ⊢ Filter.Tendsto (Function.comp (⇑μ) s) (nhdsWithin a (Set.Ioi a)) (nhds (μ (S …
  -/
  simp_rw [← map_coe_Ioi_atBot, tendsto_map'_iff, ← mem_Ioi, biInter_eq_iInter]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_8
    inst✝⁴ : LinearOrder ι
    inst✝³ : TopologicalSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : DenselyOrdered ι
    inst✝ : FirstCountableTopology ι
    s : ι → Set α
    a : ι
    hs : ∀ (r : ι), GT.gt r a → MeasureTheory.NullMeasurableSet (s r) μ
    hm : ∀ (i j : ι), LT.lt a i → LE.le i j → HasSubset.Subset (s i) (s j)
    hf : Exists fun r => And (GT.gt r a) (Ne (μ (s r)) Top.top)
    this : Filter.atBot.IsCountablyGenerated
    ⊢ Filter.Tendsto (Function.comp (Function.comp (⇑μ) s) Subtype.val) Filter.atB …
  -/
  apply tendsto_measure_iInter_atBot
    /-
      case hs
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_8
      inst✝⁴ : LinearOrder ι
      inst✝³ : TopologicalSpace ι
      inst✝² : OrderTopology ι
      inst✝¹ : DenselyOrdered ι
      inst✝ : FirstCountableTopology ι
      s : ι → Set α
      a : ι
      hs : ∀ (r : ι), GT.gt r a → MeasureTheory.NullMeasurableSet (s r) μ
      hm : ∀ (i j : ι), LT.lt a i → LE.le i j → HasSubset.Subset (s i) (s j)
      hf : Exists fun r => And (GT.gt r a) (Ne (μ (s r)) Top.top)
      this : Filter.atBot.IsCountablyGenerated
      ⊢ ∀ (i : Subtype fun x => Membership.mem (Set.Ioi a) x), MeasureTheory.NullMea …
    -/
  · rwa [Subtype.forall]
    /-
      🎉 no goals
    -/
    /-
      case hm
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_8
      inst✝⁴ : LinearOrder ι
      inst✝³ : TopologicalSpace ι
      inst✝² : OrderTopology ι
      inst✝¹ : DenselyOrdered ι
      inst✝ : FirstCountableTopology ι
      s : ι → Set α
      a : ι
      hs : ∀ (r : ι), GT.gt r a → MeasureTheory.NullMeasurableSet (s r) μ
      hm : ∀ (i j : ι), LT.lt a i → LE.le i j → HasSubset.Subset (s i) (s j)
      hf : Exists fun r => And (GT.gt r a) (Ne (μ (s r)) Top.top)
      this : Filter.atBot.IsCountablyGenerated
      ⊢ Monotone fun x => s ↑x
    -/
  · exact fun i j h ↦ hm i j i.2 h
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_8
      inst✝⁴ : LinearOrder ι
      inst✝³ : TopologicalSpace ι
      inst✝² : OrderTopology ι
      inst✝¹ : DenselyOrdered ι
      inst✝ : FirstCountableTopology ι
      s : ι → Set α
      a : ι
      hs : ∀ (r : ι), GT.gt r a → MeasureTheory.NullMeasurableSet (s r) μ
      hm : ∀ (i j : ι), LT.lt a i → LE.le i j → HasSubset.Subset (s i) (s j)
      hf : Exists fun r => And (GT.gt r a) (Ne (μ (s r)) Top.top)
      this : Filter.atBot.IsCountablyGenerated
      ⊢ Exists fun i => Ne (μ (s ↑i)) Top.top
    -/
  · simpa only [Subtype.exists, exists_prop]
    /-
      🎉 no goals
    -/


theorem measure_if {x : β} {t : Set β} {s : Set α} [Decidable (x ∈ t)] :
                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      m : MeasurableSpace α
                                                                      μ : MeasureTheory.Measure α
                                                                      x : β
                                                                      t : Set β
                                                                      s : Set α
                                                                      inst✝ : Decidable (Membership.mem t x)
                                                                      ⊢ Eq (μ (ite (Membership.mem t x) s EmptyCollection.emptyCollection)) (t.indic …
                                                                    -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
    μ (if x ∈ t then s else ∅) = indicator t (fun _ => μ s) x := by split_ifs with h <;> simp [h]
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


/-- Obtain a measure by giving an outer measure where all sets in the σ-algebra are
  Carathéodory measurable. -/
def OuterMeasure.toMeasure (m : OuterMeasure α) (h : ms ≤ m.caratheodory) : Measure α :=
  Measure.ofMeasurable (fun s _ => m s) m.empty fun _f hf hd =>
    m.iUnion_eq_of_caratheodory (fun i => h _ (hf i)) hd


theorem le_toOuterMeasure_caratheodory (μ : Measure α) : ms ≤ μ.toOuterMeasure.caratheodory :=
  fun _s hs _t => (measure_inter_add_diff _ hs).symm


@[simp]
theorem toMeasure_toOuterMeasure (m : OuterMeasure α) (h : ms ≤ m.caratheodory) :
    (m.toMeasure h).toOuterMeasure = m.trim :=
  rfl


@[simp]
theorem toMeasure_apply (m : OuterMeasure α) (h : ms ≤ m.caratheodory) {s : Set α}
    (hs : MeasurableSet s) : m.toMeasure h s = m s :=
  m.trim_eq hs


theorem le_toMeasure_apply (m : OuterMeasure α) (h : ms ≤ m.caratheodory) (s : Set α) :
    m s ≤ m.toMeasure h s :=
  m.le_trim s


theorem toMeasure_apply₀ (m : OuterMeasure α) (h : ms ≤ m.caratheodory) {s : Set α}
    (hs : NullMeasurableSet s (m.toMeasure h)) : m.toMeasure h s = m s := by
  /-
    α : Type u_1
    ms : MeasurableSpace α
    m : MeasureTheory.OuterMeasure α
    h : LE.le ms m.caratheodory
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s (m.toMeasure h)
    ⊢ Eq ((m.toMeasure h) s) (m s)
  -/
  refine le_antisymm ?_ (le_toMeasure_apply _ _ _)
  /-
    α : Type u_1
    ms : MeasurableSpace α
    m : MeasureTheory.OuterMeasure α
    h : LE.le ms m.caratheodory
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s (m.toMeasure h)
    ⊢ LE.le ((m.toMeasure h) s) (m s)
  -/
  rcases hs.exists_measurable_subset_ae_eq with ⟨t, hts, htm, heq⟩
  calc
    m.toMeasure h s = m.toMeasure h t := measure_congr heq.symm
    _ = m t := toMeasure_apply m h htm
    _ ≤ m s := m.mono hts


@[simp]
theorem toOuterMeasure_toMeasure {μ : Measure α} :
    μ.toOuterMeasure.toMeasure (le_toOuterMeasure_caratheodory _) = μ :=
  Measure.ext fun _s => μ.toOuterMeasure.trim_eq


@[simp]
theorem boundedBy_measure (μ : Measure α) : OuterMeasure.boundedBy μ = μ.toOuterMeasure :=
  μ.toOuterMeasure.boundedBy_eq_self


/-- If `u` is a superset of `t` with the same (finite) measure (both sets possibly non-measurable),
then for any measurable set `s` one also has `μ (t ∩ s) = μ (u ∩ s)`. -/
theorem measure_inter_eq_of_measure_eq {s t u : Set α} (hs : MeasurableSet s) (h : μ t = μ u)
    (htu : t ⊆ u) (ht_ne_top : μ t ≠ ∞) : μ (t ∩ s) = μ (u ∩ s) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    hs : MeasurableSet s
    h : Eq (μ t) (μ u)
    htu : HasSubset.Subset t u
    ht_ne_top : Ne (μ t) Top.top
    ⊢ Eq (μ (Inter.inter t s)) (μ (Inter.inter u s))
  -/
  rw [h] at ht_ne_top
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    hs : MeasurableSet s
    h : Eq (μ t) (μ u)
    htu : HasSubset.Subset t u
    ht_ne_top : Ne (μ u) Top.top
    ⊢ Eq (μ (Inter.inter t s)) (μ (Inter.inter u s))
  -/
  refine le_antisymm (by gcongr) ?_
  have A : μ (u ∩ s) + μ (u \ s) ≤ μ (t ∩ s) + μ (u \ s) :=
    calc
      μ (u ∩ s) + μ (u \ s) = μ u := measure_inter_add_diff _ hs
      _ = μ t := h.symm
      _ = μ (t ∩ s) + μ (t \ s) := (measure_inter_add_diff _ hs).symm
      _ ≤ μ (t ∩ s) + μ (u \ s) := by gcongr
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    hs : MeasurableSet s
    h : Eq (μ t) (μ u)
    htu : HasSubset.Subset t u
    ht_ne_top : Ne (μ u) Top.top
    A : LE.le (HAdd.hAdd (μ (Inter.inter u s)) (μ (SDiff.sdiff u s))) (HAdd.hAdd ( …
    ⊢ LE.le (μ (Inter.inter u s)) (μ (Inter.inter t s))
  -/
  have B : μ (u \ s) ≠ ∞ := (lt_of_le_of_lt (measure_mono diff_subset) ht_ne_top.lt_top).ne
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t u : Set α
    hs : MeasurableSet s
    h : Eq (μ t) (μ u)
    htu : HasSubset.Subset t u
    ht_ne_top : Ne (μ u) Top.top
    A : LE.le (HAdd.hAdd (μ (Inter.inter u s)) (μ (SDiff.sdiff u s))) (HAdd.hAdd ( …
    B : Ne (μ (SDiff.sdiff u s)) Top.top
    ⊢ LE.le (μ (Inter.inter u s)) (μ (Inter.inter t s))
  -/
  exact ENNReal.le_of_add_le_add_right B A
  /-
    🎉 no goals
  -/


/-- The measurable superset `toMeasurable μ t` of `t` (which has the same measure as `t`)
satisfies, for any measurable set `s`, the equality `μ (toMeasurable μ t ∩ s) = μ (u ∩ s)`.
Here, we require that the measure of `t` is finite. The conclusion holds without this assumption
when the measure is s-finite (for example when it is σ-finite),
see `measure_toMeasurable_inter_of_sFinite`. -/
theorem measure_toMeasurable_inter {s t : Set α} (hs : MeasurableSet s) (ht : μ t ≠ ∞) :
    μ (toMeasurable μ t ∩ s) = μ (t ∩ s) :=
  (measure_inter_eq_of_measure_eq hs (measure_toMeasurable t).symm (subset_toMeasurable μ t)
      ht).symm


instance instZero {_ : MeasurableSpace α} : Zero (Measure α) :=
  ⟨{  toOuterMeasure := 0
      m_iUnion := fun _f _hf _hd => tsum_zero.symm
      trim_le := OuterMeasure.trim_zero.le }⟩


@[simp]
theorem zero_toOuterMeasure {_m : MeasurableSpace α} : (0 : Measure α).toOuterMeasure = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_zero {_m : MeasurableSpace α} : ⇑(0 : Measure α) = 0 :=
  rfl


@[simp] lemma _root_.MeasureTheory.OuterMeasure.toMeasure_zero
    [ms : MeasurableSpace α] (h : ms ≤ (0 : OuterMeasure α).caratheodory) :
    (0 : OuterMeasure α).toMeasure h = 0 := by
  /-
    α : Type u_1
    ms : MeasurableSpace α
    h : LE.le ms (MeasureTheory.OuterMeasure.caratheodory 0)
    ⊢ Eq (MeasureTheory.OuterMeasure.toMeasure 0 h) 0
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    ms : MeasurableSpace α
    h : LE.le ms (MeasureTheory.OuterMeasure.caratheodory 0)
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.OuterMeasure.toMeasure 0 h) s) (0 s)
  -/
  simp [hs]
  /-
    🎉 no goals
  -/


@[simp] lemma _root_.MeasureTheory.OuterMeasure.toMeasure_eq_zero {ms : MeasurableSpace α}
    {μ : OuterMeasure α} (h : ms ≤ μ.caratheodory) : μ.toMeasure h = 0 ↔ μ = 0 where
              /-
                α : Type u_1
                ms : MeasurableSpace α
                μ : MeasureTheory.OuterMeasure α
                h : LE.le ms μ.caratheodory
                hμ : Eq (μ.toMeasure h) 0
                ⊢ Eq μ 0
              -/
  mp hμ := by ext s; exact le_bot_iff.1 <| (le_toMeasure_apply _ _ _).trans_eq congr($hμ s)
                     /-
                       🎉 no goals
                     -/
            /-
              α : Type u_1
              ms : MeasurableSpace α
              μ : MeasureTheory.OuterMeasure α
              h : LE.le ms μ.caratheodory
              ⊢ Eq μ 0 → Eq (μ.toMeasure h) 0
            -/
  mpr := by rintro rfl; simp
                        /-
                          🎉 no goals
                        -/


@[nontriviality]
lemma apply_eq_zero_of_isEmpty [IsEmpty α] {_ : MeasurableSpace α} (μ : Measure α) (s : Set α) :
    μ s = 0 := by
  /-
    α : Type u_1
    inst✝ : IsEmpty α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Eq (μ s) 0
  -/
  rw [eq_empty_of_isEmpty s, measure_empty]
  /-
    🎉 no goals
  -/


instance instSubsingleton [IsEmpty α] {m : MeasurableSpace α} : Subsingleton (Measure α) :=
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   δ : Type u_4
                   ι : Type u_5
                   R : Type u_6
                   R' : Type u_7
                   m0 : MeasurableSpace α
                   mβ : MeasurableSpace β
                   inst✝¹ : MeasurableSpace γ
                   μ✝ μ₁ μ₂ μ₃ ν✝ ν' ν₁ ν₂ : MeasureTheory.Measure α
                   s s' t : Set α
                   inst✝ : IsEmpty α
                   m : MeasurableSpace α
                   μ ν : MeasureTheory.Measure α
                   ⊢ Eq μ ν
                 -/
  ⟨fun μ ν => by ext1 s _; rw [apply_eq_zero_of_isEmpty, apply_eq_zero_of_isEmpty]⟩
                           /-
                             🎉 no goals
                           -/


theorem eq_zero_of_isEmpty [IsEmpty α] {_m : MeasurableSpace α} (μ : Measure α) : μ = 0 :=
  Subsingleton.elim μ 0


instance instInhabited {_ : MeasurableSpace α} : Inhabited (Measure α) :=
  ⟨0⟩


instance instAdd {_ : MeasurableSpace α} : Add (Measure α) :=
  ⟨fun μ₁ μ₂ =>
    { toOuterMeasure := μ₁.toOuterMeasure + μ₂.toOuterMeasure
      m_iUnion := fun s hs hd =>
        show μ₁ (⋃ i, s i) + μ₂ (⋃ i, s i) = ∑' i, (μ₁ (s i) + μ₂ (s i)) by
          /-
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            δ : Type u_4
            ι : Type u_5
            R : Type u_6
            R' : Type u_7
            m0 : MeasurableSpace α
            mβ : MeasurableSpace β
            inst✝ : MeasurableSpace γ
            μ μ₁✝ μ₂✝ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
            s✝ s' t : Set α
            x✝ : MeasurableSpace α
            μ₁ μ₂ : MeasureTheory.Measure α
            s : Nat → Set α
            hs : ∀ (i : Nat), MeasurableSet (s i)
            hd : Pairwise (Function.onFun Disjoint s)
            ⊢ Eq (HAdd.hAdd (μ₁ (Set.iUnion fun i => s i)) (μ₂ (Set.iUnion fun i => s i))) …
          -/
          rw [ENNReal.tsum_add, measure_iUnion hd hs, measure_iUnion hd hs]
          /-
            🎉 no goals
          -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      δ : Type u_4
                      ι : Type u_5
                      R : Type u_6
                      R' : Type u_7
                      m0 : MeasurableSpace α
                      mβ : MeasurableSpace β
                      inst✝ : MeasurableSpace γ
                      μ μ₁✝ μ₂✝ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
                      s s' t : Set α
                      x✝ : MeasurableSpace α
                      μ₁ μ₂ : MeasureTheory.Measure α
                      ⊢ LE.le (HAdd.hAdd μ₁.toOuterMeasure μ₂.toOuterMeasure).trim (HAdd.hAdd μ₁.toO …
                    -/
      trim_le := by rw [OuterMeasure.trim_add, μ₁.trimmed, μ₂.trimmed] }⟩
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem add_toOuterMeasure {_m : MeasurableSpace α} (μ₁ μ₂ : Measure α) :
    (μ₁ + μ₂).toOuterMeasure = μ₁.toOuterMeasure + μ₂.toOuterMeasure :=
  rfl


@[simp, norm_cast]
theorem coe_add {_m : MeasurableSpace α} (μ₁ μ₂ : Measure α) : ⇑(μ₁ + μ₂) = μ₁ + μ₂ :=
  rfl


theorem add_apply {_m : MeasurableSpace α} (μ₁ μ₂ : Measure α) (s : Set α) :
    (μ₁ + μ₂) s = μ₁ s + μ₂ s :=
  rfl


instance instSMul {_ : MeasurableSpace α} : SMul R (Measure α) :=
  ⟨fun c μ =>
    { toOuterMeasure := c • μ.toOuterMeasure
      m_iUnion := fun s hs hd => by
        simp only [OuterMeasure.smul_apply, coe_toOuterMeasure, ENNReal.tsum_const_smul,
          measure_iUnion hd hs]
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      δ : Type u_4
                      ι : Type u_5
                      R : Type u_6
                      R' : Type u_7
                      m0 : MeasurableSpace α
                      mβ : MeasurableSpace β
                      inst✝⁴ : MeasurableSpace γ
                      μ✝ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
                      s s' t : Set α
                      inst✝³ : SMul R ENNReal
                      inst✝² : IsScalarTower R ENNReal ENNReal
                      inst✝¹ : SMul R' ENNReal
                      inst✝ : IsScalarTower R' ENNReal ENNReal
                      x✝ : MeasurableSpace α
                      c : R
                      μ : MeasureTheory.Measure α
                      ⊢ LE.le (HSMul.hSMul c μ.toOuterMeasure).trim (HSMul.hSMul c μ.toOuterMeasure)
                    -/
      trim_le := by rw [OuterMeasure.trim_smul, μ.trimmed] }⟩
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem smul_toOuterMeasure {_m : MeasurableSpace α} (c : R) (μ : Measure α) :
    (c • μ).toOuterMeasure = c • μ.toOuterMeasure :=
  rfl


@[simp, norm_cast]
theorem coe_smul {_m : MeasurableSpace α} (c : R) (μ : Measure α) : ⇑(c • μ) = c • ⇑μ :=
  rfl


@[simp]
theorem smul_apply {_m : MeasurableSpace α} (c : R) (μ : Measure α) (s : Set α) :
    (c • μ) s = c • μ s :=
  rfl


instance instSMulCommClass [SMulCommClass R R' ℝ≥0∞] {_ : MeasurableSpace α} :
    SMulCommClass R R' (Measure α) :=
  ⟨fun _ _ _ => ext fun _ _ => smul_comm _ _ _⟩


instance instIsScalarTower [SMul R R'] [IsScalarTower R R' ℝ≥0∞] {_ : MeasurableSpace α} :
    IsScalarTower R R' (Measure α) :=
  ⟨fun _ _ _ => ext fun _ _ => smul_assoc _ _ _⟩


instance instIsCentralScalar [SMul Rᵐᵒᵖ ℝ≥0∞] [IsCentralScalar R ℝ≥0∞] {_ : MeasurableSpace α} :
    IsCentralScalar R (Measure α) :=
  ⟨fun _ _ => ext fun _ _ => op_smul_eq_smul _ _⟩


instance instNoZeroSMulDivisors [Zero R] [SMulWithZero R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    [NoZeroSMulDivisors R ℝ≥0∞] : NoZeroSMulDivisors R (Measure α) where
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               γ : Type u_3
                                               δ : Type u_4
                                               ι : Type u_5
                                               R : Type u_6
                                               R' : Type u_7
                                               m0 : MeasurableSpace α
                                               mβ : MeasurableSpace β
                                               inst✝⁴ : MeasurableSpace γ
                                               μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
                                               s s' t : Set α
                                               inst✝³ : Zero R
                                               inst✝² : SMulWithZero R ENNReal
                                               inst✝¹ : IsScalarTower R ENNReal ENNReal
                                               inst✝ : NoZeroSMulDivisors R ENNReal
                                               c✝ : R
                                               x✝ : MeasureTheory.Measure α
                                               h : Eq (HSMul.hSMul c✝ x✝) 0
                                               ⊢ Or (Eq c✝ 0) (Eq x✝ 0)
                                             -/
  eq_zero_or_eq_zero_of_smul_eq_zero h := by simpa [Ne, ext_iff', forall_or_left] using h
                                             /-
                                               🎉 no goals
                                             -/


instance instMulAction [Monoid R] [MulAction R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    {_ : MeasurableSpace α} : MulAction R (Measure α) :=
  Injective.mulAction _ toOuterMeasure_injective smul_toOuterMeasure


instance instAddCommMonoid {_ : MeasurableSpace α} : AddCommMonoid (Measure α) :=
  toOuterMeasure_injective.addCommMonoid toOuterMeasure zero_toOuterMeasure add_toOuterMeasure
    fun _ _ => smul_toOuterMeasure _ _


/-- Coercion to function as an additive monoid homomorphism. -/
def coeAddHom {_ : MeasurableSpace α} : Measure α →+ Set α → ℝ≥0∞ where
  toFun := (⇑)
  map_zero' := coe_zero
  map_add' := coe_add


@[simp]
theorem coe_finset_sum {_m : MeasurableSpace α} (I : Finset ι) (μ : ι → Measure α) :
    ⇑(∑ i ∈ I, μ i) = ∑ i ∈ I, ⇑(μ i) := map_sum coeAddHom μ I


theorem finset_sum_apply {m : MeasurableSpace α} (I : Finset ι) (μ : ι → Measure α) (s : Set α) :
                                            /-
                                              α : Type u_1
                                              ι : Type u_5
                                              m : MeasurableSpace α
                                              I : Finset ι
                                              μ : ι → MeasureTheory.Measure α
                                              s : Set α
                                              ⊢ Eq ((I.sum fun i => μ i) s) (I.sum fun i => (μ i) s)
                                            -/
    (∑ i ∈ I, μ i) s = ∑ i ∈ I, μ i s := by rw [coe_finset_sum, Finset.sum_apply]
                                            /-
                                              🎉 no goals
                                            -/


instance instDistribMulAction [Monoid R] [DistribMulAction R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    {_ : MeasurableSpace α} : DistribMulAction R (Measure α) :=
  Injective.distribMulAction ⟨⟨toOuterMeasure, zero_toOuterMeasure⟩, add_toOuterMeasure⟩
    toOuterMeasure_injective smul_toOuterMeasure


instance instModule [Semiring R] [Module R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    {_ : MeasurableSpace α} : Module R (Measure α) :=
  Injective.module R ⟨⟨toOuterMeasure, zero_toOuterMeasure⟩, add_toOuterMeasure⟩
    toOuterMeasure_injective smul_toOuterMeasure


@[simp]
theorem coe_nnreal_smul_apply {_m : MeasurableSpace α} (c : ℝ≥0) (μ : Measure α) (s : Set α) :
    (c • μ) s = c * μ s :=
  rfl


@[simp]
theorem nnreal_smul_coe_apply {_m : MeasurableSpace α} (c : ℝ≥0) (μ : Measure α) (s : Set α) :
    c • μ s = c * μ s := by
  /-
    α : Type u_1
    _m : MeasurableSpace α
    c : NNReal
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Eq (HSMul.hSMul c (μ s)) (HMul.hMul (↑c) (μ s))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ae_smul_measure {p : α → Prop} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    (h : ∀ᵐ x ∂μ, p x) (c : R) : ∀ᵐ x ∂c • μ, p x :=
                 /-
                   α : Type u_1
                   R : Type u_6
                   m0 : MeasurableSpace α
                   μ : MeasureTheory.Measure α
                   p : α → Prop
                   inst✝¹ : SMul R ENNReal
                   inst✝ : IsScalarTower R ENNReal ENNReal
                   h : Filter.Eventually (fun x => p x) (MeasureTheory.ae μ)
                   c : R
                   ⊢ Eq ((HSMul.hSMul c μ) (setOf fun a => Not (p a))) 0
                 -/
  ae_iff.2 <| by rw [smul_apply, ae_iff.1 h, ← smul_one_smul ℝ≥0∞, smul_zero]
                 /-
                   🎉 no goals
                 -/


theorem ae_smul_measure_le [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] (c : R) :
    ae (c • μ) ≤ ae μ := fun _ h ↦ ae_smul_measure h c


lemma ae_smul_measure_iff (hc : c ≠ 0) {μ : Measure α} : (∀ᵐ x ∂c • μ, p x) ↔ ∀ᵐ x ∂μ, p x := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    R : Type u_8
    inst✝³ : Zero R
    inst✝² : SMulWithZero R ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : NoZeroSMulDivisors R ENNReal
    c : R
    p : α → Prop
    hc : Ne c 0
    μ : MeasureTheory.Measure α
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (HSMul.hSMul c μ)))  …
  -/
  simp [ae_iff, hc]
  /-
    🎉 no goals
  -/


@[simp] lemma ae_smul_measure_eq (hc : c ≠ 0) (μ : Measure α) : ae (c • μ) = ae μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    R : Type u_8
    inst✝³ : Zero R
    inst✝² : SMulWithZero R ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : NoZeroSMulDivisors R ENNReal
    c : R
    hc : Ne c 0
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.ae (HSMul.hSMul c μ)) (MeasureTheory.ae μ)
  -/
  ext; exact ae_smul_measure_iff hc
       /-
         🎉 no goals
       -/


theorem measure_eq_left_of_subset_of_measure_add_eq {s t : Set α} (h : (μ + ν) t ≠ ∞) (h' : s ⊆ t)
    (h'' : (μ + ν) s = (μ + ν) t) : μ s = μ t := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    h : Ne ((HAdd.hAdd μ ν) t) Top.top
    h' : HasSubset.Subset s t
    h'' : Eq ((HAdd.hAdd μ ν) s) ((HAdd.hAdd μ ν) t)
    ⊢ Eq (μ s) (μ t)
  -/
  refine le_antisymm (measure_mono h') ?_
  have : μ t + ν t ≤ μ s + ν t :=
    calc
      μ t + ν t = μ s + ν s := h''.symm
      _ ≤ μ s + ν t := by gcongr
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    h : Ne ((HAdd.hAdd μ ν) t) Top.top
    h' : HasSubset.Subset s t
    h'' : Eq ((HAdd.hAdd μ ν) s) ((HAdd.hAdd μ ν) t)
    this : LE.le (HAdd.hAdd (μ t) (ν t)) (HAdd.hAdd (μ s) (ν t))
    ⊢ LE.le (μ t) (μ s)
  -/
  apply ENNReal.le_of_add_le_add_right _ this
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    h : Ne ((HAdd.hAdd μ ν) t) Top.top
    h' : HasSubset.Subset s t
    h'' : Eq ((HAdd.hAdd μ ν) s) ((HAdd.hAdd μ ν) t)
    this : LE.le (HAdd.hAdd (μ t) (ν t)) (HAdd.hAdd (μ s) (ν t))
    ⊢ Ne (ν t) Top.top
  -/
  exact ne_top_of_le_ne_top h (le_add_left le_rfl)
  /-
    🎉 no goals
  -/


theorem measure_eq_right_of_subset_of_measure_add_eq {s t : Set α} (h : (μ + ν) t ≠ ∞) (h' : s ⊆ t)
    (h'' : (μ + ν) s = (μ + ν) t) : ν s = ν t := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    h : Ne ((HAdd.hAdd μ ν) t) Top.top
    h' : HasSubset.Subset s t
    h'' : Eq ((HAdd.hAdd μ ν) s) ((HAdd.hAdd μ ν) t)
    ⊢ Eq (ν s) (ν t)
  -/
  rw [add_comm] at h'' h
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    h : Ne ((HAdd.hAdd ν μ) t) Top.top
    h' : HasSubset.Subset s t
    h'' : Eq ((HAdd.hAdd ν μ) s) ((HAdd.hAdd ν μ) t)
    ⊢ Eq (ν s) (ν t)
  -/
  exact measure_eq_left_of_subset_of_measure_add_eq h h' h''
  /-
    🎉 no goals
  -/


theorem measure_toMeasurable_add_inter_left {s t : Set α} (hs : MeasurableSet s)
    (ht : (μ + ν) t ≠ ∞) : μ (toMeasurable (μ + ν) t ∩ s) = μ (t ∩ s) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    ht : Ne ((HAdd.hAdd μ ν) t) Top.top
    ⊢ Eq (μ (Inter.inter (MeasureTheory.toMeasurable (HAdd.hAdd μ ν) t) s)) (μ (In …
  -/
  refine (measure_inter_eq_of_measure_eq hs ?_ (subset_toMeasurable _ _) ?_).symm
  · refine
      measure_eq_left_of_subset_of_measure_add_eq ?_ (subset_toMeasurable _ _)
        (measure_toMeasurable t).symm
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s t : Set α
      hs : MeasurableSet s
      ht : Ne ((HAdd.hAdd μ ν) t) Top.top
      ⊢ Ne ((HAdd.hAdd μ ν) (MeasureTheory.toMeasurable (HAdd.hAdd μ ν) t)) Top.top
    -/
    rwa [measure_toMeasurable t]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s t : Set α
      hs : MeasurableSet s
      ht : Ne ((HAdd.hAdd μ ν) t) Top.top
      ⊢ Ne (μ t) Top.top
    -/
  · simp only [not_or, ENNReal.add_eq_top, Pi.add_apply, Ne, coe_add] at ht
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s t : Set α
      hs : MeasurableSet s
      ht : And (Not (Eq (μ t) Top.top)) (Not (Eq (ν t) Top.top))
      ⊢ Ne (μ t) Top.top
    -/
    exact ht.1
    /-
      🎉 no goals
    -/


theorem measure_toMeasurable_add_inter_right {s t : Set α} (hs : MeasurableSet s)
    (ht : (μ + ν) t ≠ ∞) : ν (toMeasurable (μ + ν) t ∩ s) = ν (t ∩ s) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    ht : Ne ((HAdd.hAdd μ ν) t) Top.top
    ⊢ Eq (ν (Inter.inter (MeasureTheory.toMeasurable (HAdd.hAdd μ ν) t) s)) (ν (In …
  -/
  rw [add_comm] at ht ⊢
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    ht : Ne ((HAdd.hAdd ν μ) t) Top.top
    ⊢ Eq (ν (Inter.inter (MeasureTheory.toMeasurable (HAdd.hAdd ν μ) t) s)) (ν (In …
  -/
  exact measure_toMeasurable_add_inter_left hs ht
  /-
    🎉 no goals
  -/


/-- Measures are partially ordered. -/
instance instPartialOrder {_ : MeasurableSpace α} : PartialOrder (Measure α) where
  le m₁ m₂ := ∀ s, m₁ s ≤ m₂ s
  le_refl _ _ := le_rfl
  le_trans _ _ _ h₁ h₂ s := le_trans (h₁ s) (h₂ s)
  le_antisymm _ _ h₁ h₂ := ext fun s _ => le_antisymm (h₁ s) (h₂ s)


theorem toOuterMeasure_le : μ₁.toOuterMeasure ≤ μ₂.toOuterMeasure ↔ μ₁ ≤ μ₂ := .rfl


theorem le_iff : μ₁ ≤ μ₂ ↔ ∀ s, MeasurableSet s → μ₁ s ≤ μ₂ s := outerMeasure_le_iff


theorem le_intro (h : ∀ s, MeasurableSet s → s.Nonempty → μ₁ s ≤ μ₂ s) : μ₁ ≤ μ₂ :=
                                                      /-
                                                        α : Type u_1
                                                        m0 : MeasurableSpace α
                                                        μ₁ μ₂ : MeasureTheory.Measure α
                                                        h : ∀ (s : Set α), MeasurableSet s → s.Nonempty → LE.le (μ₁ s) (μ₂ s)
                                                        s : Set α
                                                        hs : MeasurableSet s
                                                        ⊢ Eq s EmptyCollection.emptyCollection → LE.le (μ₁ s) (μ₂ s)
                                                      -/
  le_iff.2 fun s hs ↦ s.eq_empty_or_nonempty.elim (by rintro rfl; simp) (h s hs)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem le_iff' : μ₁ ≤ μ₂ ↔ ∀ s, μ₁ s ≤ μ₂ s := .rfl


theorem lt_iff : μ < ν ↔ μ ≤ ν ∧ ∃ s, MeasurableSet s ∧ μ s < ν s :=
  lt_iff_le_not_le.trans <|
                            /-
                              α : Type u_1
                              m0 : MeasurableSpace α
                              μ ν : MeasureTheory.Measure α
                              ⊢ Iff (Not (LE.le ν μ)) (Exists fun s => And (MeasurableSet s) (LT.lt (μ s) (ν …
                            -/
    and_congr Iff.rfl <| by simp only [le_iff, not_forall, not_le, exists_prop]
                            /-
                              🎉 no goals
                            -/


theorem lt_iff' : μ < ν ↔ μ ≤ ν ∧ ∃ s, μ s < ν s :=
                                                    /-
                                                      α : Type u_1
                                                      m0 : MeasurableSpace α
                                                      μ ν : MeasureTheory.Measure α
                                                      ⊢ Iff (Not (LE.le ν μ)) (Exists fun s => LT.lt (μ s) (ν s))
                                                    -/
  lt_iff_le_not_le.trans <| and_congr Iff.rfl <| by simp only [le_iff', not_forall, not_le]
                                                    /-
                                                      🎉 no goals
                                                    -/


instance instAddLeftMono {_ : MeasurableSpace α} : AddLeftMono (Measure α) :=
  ⟨fun _ν _μ₁ _μ₂ hμ s => add_le_add_left (hμ s) _⟩


protected theorem le_add_left (h : μ ≤ ν) : μ ≤ ν' + ν := fun s => le_add_left (h s)


protected theorem le_add_right (h : μ ≤ ν) : μ ≤ ν + ν' := fun s => le_add_right (h s)


theorem sInf_caratheodory (s : Set α) (hs : MeasurableSet s) :
    MeasurableSet[(sInf (toOuterMeasure '' m)).caratheodory] s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    m : Set (MeasureTheory.Measure α)
    s : Set α
    hs : MeasurableSet s
    ⊢ MeasurableSet s
  -/
  rw [OuterMeasure.sInf_eq_boundedBy_sInfGen]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    m : Set (MeasureTheory.Measure α)
    s : Set α
    hs : MeasurableSet s
    ⊢ MeasurableSet s
  -/
  refine OuterMeasure.boundedBy_caratheodory fun t => ?_
  simp only [OuterMeasure.sInfGen, le_iInf_iff, forall_mem_image, measure_eq_iInf t,
    coe_toOuterMeasure]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    m : Set (MeasureTheory.Measure α)
    s : Set α
    hs : MeasurableSet s
    t : Set α
    ⊢ ∀ ⦃x : MeasureTheory.Measure α⦄, Membership.mem m x → ∀ (i : Set α), HasSubs …
  -/
  intro μ hμ u htu _hu
  have hm : ∀ {s t}, s ⊆ t → OuterMeasure.sInfGen (toOuterMeasure '' m) s ≤ μ t := by
    intro s t hst
    rw [OuterMeasure.sInfGen_def, iInf_image]
    exact iInf₂_le_of_le μ hμ <| measure_mono hst
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    m : Set (MeasureTheory.Measure α)
    s : Set α
    hs : MeasurableSet s
    t : Set α
    μ : MeasureTheory.Measure α
    hμ : Membership.mem m μ
    u : Set α
    htu : HasSubset.Subset t u
    _hu : MeasurableSet u
    hm : ∀ {s t : Set α}, HasSubset.Subset s t → LE.le (MeasureTheory.OuterMeasure …
    ⊢ LE.le (HAdd.hAdd (iInf fun μ => iInf fun x => μ (Inter.inter t s)) (iInf fun …
  -/
  rw [← measure_inter_add_diff u hs]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    m : Set (MeasureTheory.Measure α)
    s : Set α
    hs : MeasurableSet s
    t : Set α
    μ : MeasureTheory.Measure α
    hμ : Membership.mem m μ
    u : Set α
    htu : HasSubset.Subset t u
    _hu : MeasurableSet u
    hm : ∀ {s t : Set α}, HasSubset.Subset s t → LE.le (MeasureTheory.OuterMeasure …
    ⊢ LE.le (HAdd.hAdd (iInf fun μ => iInf fun x => μ (Inter.inter t s)) (iInf fun …
  -/
  exact add_le_add (hm <| inter_subset_inter_left _ htu) (hm <| diff_subset_diff_left htu)
  /-
    🎉 no goals
  -/


instance {_ : MeasurableSpace α} : InfSet (Measure α) :=
  ⟨fun m => (sInf (toOuterMeasure '' m)).toMeasure <| sInf_caratheodory⟩


theorem sInf_apply (hs : MeasurableSet s) : sInf m s = sInf (toOuterMeasure '' m) s :=
  toMeasure_apply _ _ hs


private theorem measure_sInf_le (h : μ ∈ m) : sInf m ≤ μ :=
  have : sInf (toOuterMeasure '' m) ≤ μ.toOuterMeasure := sInf_le (mem_image_of_mem _ h)
                          /-
                            α : Type u_1
                            m0 : MeasurableSpace α
                            μ : MeasureTheory.Measure α
                            m : Set (MeasureTheory.Measure α)
                            h : Membership.mem m μ
                            this : LE.le (InfSet.sInf (Set.image MeasureTheory.Measure.toOuterMeasure m))  …
                            s : Set α
                            hs : MeasurableSet s
                            ⊢ LE.le ((InfSet.sInf m) s) (μ s)
                          -/
  le_iff.2 fun s hs => by rw [sInf_apply hs]; exact this s
                                              /-
                                                🎉 no goals
                                              -/


private theorem measure_le_sInf (h : ∀ μ' ∈ m, μ ≤ μ') : μ ≤ sInf m :=
  have : μ.toOuterMeasure ≤ sInf (toOuterMeasure '' m) :=
    le_sInf <| forall_mem_image.2 fun _ hμ ↦ toOuterMeasure_le.2 <| h _ hμ
                          /-
                            α : Type u_1
                            m0 : MeasurableSpace α
                            μ : MeasureTheory.Measure α
                            m : Set (MeasureTheory.Measure α)
                            h : ∀ (μ' : MeasureTheory.Measure α), Membership.mem m μ' → LE.le μ μ'
                            this : LE.le μ.toOuterMeasure (InfSet.sInf (Set.image MeasureTheory.Measure.to …
                            s : Set α
                            hs : MeasurableSet s
                            ⊢ LE.le (μ s) ((InfSet.sInf m) s)
                          -/
  le_iff.2 fun s hs => by rw [sInf_apply hs]; exact this s
                                              /-
                                                🎉 no goals
                                              -/


instance instCompleteSemilatticeInf {_ : MeasurableSpace α} : CompleteSemilatticeInf (Measure α) :=
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          δ : Type u_4
          ι : Type u_5
          R : Type u_6
          R' : Type u_7
          m0 : MeasurableSpace α
          mβ : MeasurableSpace β
          inst✝ : MeasurableSpace γ
          μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
          s s' t : Set α
          m : Set (MeasureTheory.Measure α)
          x✝ : MeasurableSpace α
          ⊢ PartialOrder (MeasureTheory.Measure α)
        -/
  { (by infer_instance : PartialOrder (Measure α)),
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          δ : Type u_4
          ι : Type u_5
          R : Type u_6
          R' : Type u_7
          m0 : MeasurableSpace α
          mβ : MeasurableSpace β
          inst✝ : MeasurableSpace γ
          μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
          s s' t : Set α
          m : Set (MeasureTheory.Measure α)
          x✝ : MeasurableSpace α
          ⊢ InfSet (MeasureTheory.Measure α)
        -/
    (by infer_instance : InfSet (Measure α)) with
        /-
          🎉 no goals
        -/
    sInf_le := fun _s _a => measure_sInf_le
    le_sInf := fun _s _a => measure_le_sInf }


instance instCompleteLattice {_ : MeasurableSpace α} : CompleteLattice (Measure α) :=
  { completeLatticeOfCompleteSemilatticeInf (Measure α) with
    top :=
      { toOuterMeasure := ⊤,
        m_iUnion := by
          /-
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            δ : Type u_4
            ι : Type u_5
            R : Type u_6
            R' : Type u_7
            m0 : MeasurableSpace α
            mβ : MeasurableSpace β
            inst✝ : MeasurableSpace γ
            μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
            s s' t : Set α
            m : Set (MeasureTheory.Measure α)
            x✝ : MeasurableSpace α
            ⊢ ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), MeasurableSet (f i)) → Pairwise (Function …
          -/
          intro f _ _
          /-
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            δ : Type u_4
            ι : Type u_5
            R : Type u_6
            R' : Type u_7
            m0 : MeasurableSpace α
            mβ : MeasurableSpace β
            inst✝ : MeasurableSpace γ
            μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
            s s' t : Set α
            m : Set (MeasureTheory.Measure α)
            x✝ : MeasurableSpace α
            f : Nat → Set α
            a✝¹ : ∀ (i : Nat), MeasurableSet (f i)
            a✝ : Pairwise (Function.onFun Disjoint f)
            ⊢ Eq (Top.top (Set.iUnion fun i => f i)) (tsum fun i => Top.top (f i))
          -/
          refine (measure_iUnion_le _).antisymm ?_
          if hne : (⋃ i, f i).Nonempty then
            rw [OuterMeasure.top_apply hne]
            exact le_top
          else
            simp_all [Set.not_nonempty_iff_eq_empty]
        trim_le := le_top },
    le_top := fun _ => toOuterMeasure_le.mp le_top
    bot := 0
    bot_le := fun _a _s => bot_le }


lemma inf_apply {s : Set α} (hs : MeasurableSet s) :
    (μ ⊓ ν) s = sInf {m | ∃ t, m = μ (t ∩ s) + ν (tᶜ ∩ s)} := by
  -- `(μ ⊓ ν) s` is defined as `⊓ (t : ℕ → Set α) (ht : s ⊆ ⋃ n, t n), ∑' n, μ (t n) ⊓ ν (t n)`
  rw [← sInf_pair, Measure.sInf_apply hs, OuterMeasure.sInf_apply
    (image_nonempty.2 <| insert_nonempty μ {ν})]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (iInf fun t => iInf fun x => tsum fun n => iInf fun μ_1 => iInf fun x =>  …
  -/
  refine le_antisymm (le_sInf fun m ⟨t, ht₁⟩ ↦ ?_) (le_iInf₂ fun t' ht' ↦ ?_)
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      m : ENNReal
      x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
      t : Set α
      ht₁ : Eq m (HAdd.hAdd (μ (Inter.inter t s)) (ν (Inter.inter (HasCompl.compl t) …
      ⊢ LE.le (iInf fun t => iInf fun x => tsum fun n => iInf fun μ_1 => iInf fun x  …
    -/
  · subst ht₁
    -- We first show `(μ ⊓ ν) s ≤ μ (t ∩ s) + ν (tᶜ ∩ s)` for any `t : Set α`
    -- For this, define the sequence `t' : ℕ → Set α` where `t' 0 = t ∩ s`, `t' 1 = tᶜ ∩ s` and
    -- `∅` otherwise. Then, we have by construction
    -- `(μ ⊓ ν) s ≤ ∑' n, μ (t' n) ⊓ ν (t' n) ≤ μ (t' 0) + ν (t' 1) = μ (t ∩ s) + ν (tᶜ ∩ s)`.
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
      ⊢ LE.le (iInf fun t => iInf fun x => tsum fun n => iInf fun μ_1 => iInf fun x  …
    -/
    set t' : ℕ → Set α := fun n ↦ if n = 0 then t ∩ s else if n = 1 then tᶜ ∩ s else ∅ with ht'
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
      t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
      ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
      ⊢ LE.le (iInf fun t => iInf fun x => tsum fun n => iInf fun μ_1 => iInf fun x  …
    -/
    refine (iInf₂_le t' fun x hx ↦ ?_).trans ?_
      /-
        case refine_1.refine_1
        α : Type u_1
        m0 : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        hs : MeasurableSet s
        t : Set α
        x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
        t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
        ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
        x : α
        hx : Membership.mem s x
        ⊢ Membership.mem (Set.iUnion t') x
      -/
    · by_cases hxt : x ∈ t
        /-
          case pos
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t : Set α
          x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
          t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
          ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
          x : α
          hx : Membership.mem s x
          hxt : Membership.mem t x
          ⊢ Membership.mem (Set.iUnion t') x
        -/
      · refine mem_iUnion.2 ⟨0, ?_⟩
        /-
          case pos
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t : Set α
          x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
          t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
          ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
          x : α
          hx : Membership.mem s x
          hxt : Membership.mem t x
          ⊢ Membership.mem (ite (Eq 0 0) (Inter.inter t s) (ite (Eq 0 1) (Inter.inter (H …
        -/
        simp [hx, hxt]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t : Set α
          x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
          t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
          ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
          x : α
          hx : Membership.mem s x
          hxt : Not (Membership.mem t x)
          ⊢ Membership.mem (Set.iUnion t') x
        -/
      · refine mem_iUnion.2 ⟨1, ?_⟩
        /-
          case neg
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t : Set α
          x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
          t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
          ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
          x : α
          hx : Membership.mem s x
          hxt : Not (Membership.mem t x)
          ⊢ Membership.mem (ite (Eq 1 0) (Inter.inter t s) (ite (Eq 1 1) (Inter.inter (H …
        -/
        simp [hx, hxt]
        /-
          🎉 no goals
        -/
      /-
        case refine_1.refine_2
        α : Type u_1
        m0 : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        hs : MeasurableSet s
        t : Set α
        x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
        t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
        ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
        ⊢ LE.le (tsum fun n => iInf fun μ_1 => iInf fun x => μ_1 (t' n)) (HAdd.hAdd (μ …
      -/
    · simp only [iInf_image, coe_toOuterMeasure, iInf_pair]
      rw [tsum_eq_add_tsum_ite 0, tsum_eq_add_tsum_ite 1, if_neg zero_ne_one.symm,
        (tsum_eq_zero_iff ENNReal.summable).2 _, add_zero]
        /-
          case refine_1.refine_2
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t : Set α
          x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
          t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
          ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
          ⊢ LE.le (HAdd.hAdd (Min.min (μ (t' 0)) (ν (t' 0))) (Min.min (μ (t' 1)) (ν (t'  …
        -/
      · exact add_le_add (inf_le_left.trans <| by simp [ht']) (inf_le_right.trans <| by simp [ht'])
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t : Set α
          x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
          t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
          ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
          ⊢ ∀ (x : Nat), Eq (ite (Eq x 1) 0 (ite (Eq x 0) 0 (Min.min (μ (t' x)) (ν (t' x …
        -/
      · simp only [ite_eq_left_iff]
        /-
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t : Set α
          x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
          t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
          ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
          ⊢ ∀ (x : Nat), Not (Eq x 1) → Not (Eq x 0) → Eq (Min.min (μ (t' x)) (ν (t' x)) …
        -/
        intro n hn₁ hn₀
        /-
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t : Set α
          x✝ : Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter. …
          t' : Nat → Set α := fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Int …
          ht' : Eq t' fun n => ite (Eq n 0) (Inter.inter t s) (ite (Eq n 1) (Inter.inter …
          n : Nat
          hn₁ : Not (Eq n 1)
          hn₀ : Not (Eq n 0)
          ⊢ Eq (Min.min (μ (t' n)) (ν (t' n))) 0
        -/
        simp only [ht', if_neg hn₀, if_neg hn₁, measure_empty, iInf_pair, le_refl, inf_of_le_left]
        /-
          🎉 no goals
        -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t' : Nat → Set α
      ht' : HasSubset.Subset s (Set.iUnion t')
      ⊢ LE.le (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter …
    -/
  · simp only [iInf_image, coe_toOuterMeasure, iInf_pair]
    -- Conversely, fixing `t' : ℕ → Set α` such that `s ⊆ ⋃ n, t' n`, we construct `t : Set α`
    -- for which `μ (t ∩ s) + ν (tᶜ ∩ s) ≤ ∑' n, μ (t' n) ⊓ ν (t' n)`.
    -- Denoting `I := {n | μ (t' n) ≤ ν (t' n)}`, we set `t = ⋃ n ∈ I, t' n`.
    -- Clearly `μ (t ∩ s) ≤ ∑' n ∈ I, μ (t' n)` and `ν (tᶜ ∩ s) ≤ ∑' n ∉ I, ν (t' n)`, so
    -- `μ (t ∩ s) + ν (tᶜ ∩ s) ≤ ∑' n ∈ I, μ (t' n) + ∑' n ∉ I, ν (t' n)`
    -- where the RHS equals `∑' n, μ (t' n) ⊓ ν (t' n)` by the choice of `I`.
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t' : Nat → Set α
      ht' : HasSubset.Subset s (Set.iUnion t')
      ⊢ LE.le (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter …
    -/
    set t := ⋃ n ∈ {k : ℕ | μ (t' k) ≤ ν (t' k)}, t' n with ht
    suffices hadd : μ (t ∩ s) + ν (tᶜ ∩ s) ≤ ∑' n, μ (t' n) ⊓ ν (t' n) by
      exact le_trans (sInf_le ⟨t, rfl⟩) hadd
    have hle₁ : μ (t ∩ s) ≤ ∑' (n : {k | μ (t' k) ≤ ν (t' k)}), μ (t' n) :=
      (measure_mono inter_subset_left).trans <| measure_biUnion_le _ (to_countable _) _
    have hcap : tᶜ ∩ s ⊆ ⋃ n ∈ {k | ν (t' k) < μ (t' k)}, t' n := by
      simp_rw [ht, compl_iUnion]
      refine fun x ⟨hx₁, hx₂⟩ ↦ mem_iUnion₂.2 ?_
      obtain ⟨i, hi⟩ := mem_iUnion.1 <| ht' hx₂
      refine ⟨i, ?_, hi⟩
      by_contra h
      simp only [mem_setOf_eq, not_lt] at h
      exact mem_iInter₂.1 hx₁ i h hi
    have hle₂ : ν (tᶜ ∩ s) ≤ ∑' (n : {k | ν (t' k) < μ (t' k)}), ν (t' n) :=
      (measure_mono hcap).trans (measure_biUnion_le ν (to_countable {k | ν (t' k) < μ (t' k)}) _)
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t' : Nat → Set α
      ht' : HasSubset.Subset s (Set.iUnion t')
      t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
      ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
      hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
      hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
      hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
      ⊢ LE.le (HAdd.hAdd (μ (Inter.inter t s)) (ν (Inter.inter (HasCompl.compl t) s) …
    -/
    refine (add_le_add hle₁ hle₂).trans ?_
    have heq : {k | μ (t' k) ≤ ν (t' k)} ∪ {k | ν (t' k) < μ (t' k)} = univ := by
      ext k; simp [le_or_lt]
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t' : Nat → Set α
      ht' : HasSubset.Subset s (Set.iUnion t')
      t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
      ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
      hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
      hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
      hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
      heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
      ⊢ LE.le (HAdd.hAdd (tsum fun n => μ (t' ↑n)) (tsum fun n => ν (t' ↑n))) (tsum  …
    -/
    conv in ∑' (n : ℕ), μ (t' n) ⊓ ν (t' n) => rw [← tsum_univ, ← heq]
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t' : Nat → Set α
      ht' : HasSubset.Subset s (Set.iUnion t')
      t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
      ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
      hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
      hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
      hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
      heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
      ⊢ LE.le (HAdd.hAdd (tsum fun n => μ (t' ↑n)) (tsum fun n => ν (t' ↑n))) (tsum  …
    -/
    rw [tsum_union_disjoint (f := fun n ↦ μ (t' n) ⊓ ν (t' n)) ?_ ENNReal.summable ENNReal.summable]
      /-
        case refine_2
        α : Type u_1
        m0 : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        hs : MeasurableSet s
        t' : Nat → Set α
        ht' : HasSubset.Subset s (Set.iUnion t')
        t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
        ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
        hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
        hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
        hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
        heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
        ⊢ LE.le (HAdd.hAdd (tsum fun n => μ (t' ↑n)) (tsum fun n => ν (t' ↑n))) (HAdd. …
      -/
    · refine add_le_add (tsum_congr ?_).le (tsum_congr ?_).le
        /-
          case refine_2.refine_1
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t' : Nat → Set α
          ht' : HasSubset.Subset s (Set.iUnion t')
          t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
          ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
          hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
          hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
          hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
          heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
          ⊢ ∀ (b : ↑(setOf fun k => LE.le (μ (t' k)) (ν (t' k)))), Eq (μ (t' ↑b)) (Min.m …
        -/
      · rw [Subtype.forall]
        /-
          case refine_2.refine_1
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t' : Nat → Set α
          ht' : HasSubset.Subset s (Set.iUnion t')
          t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
          ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
          hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
          hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
          hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
          heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
          ⊢ ∀ (a : Nat) (b : Membership.mem (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) …
        -/
        intro n hn; simpa
                    /-
                      🎉 no goals
                    -/
        /-
          case refine_2.refine_2
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t' : Nat → Set α
          ht' : HasSubset.Subset s (Set.iUnion t')
          t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
          ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
          hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
          hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
          hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
          heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
          ⊢ ∀ (b : ↑(setOf fun k => LT.lt (ν (t' k)) (μ (t' k)))), Eq (ν (t' ↑b)) (Min.m …
        -/
      · rw [Subtype.forall]
        /-
          case refine_2.refine_2
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t' : Nat → Set α
          ht' : HasSubset.Subset s (Set.iUnion t')
          t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
          ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
          hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
          hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
          hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
          heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
          ⊢ ∀ (a : Nat) (b : Membership.mem (setOf fun k => LT.lt (ν (t' k)) (μ (t' k))) …
        -/
        intro n hn
        /-
          case refine_2.refine_2
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t' : Nat → Set α
          ht' : HasSubset.Subset s (Set.iUnion t')
          t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
          ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
          hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
          hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
          hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
          heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
          n : Nat
          hn : Membership.mem (setOf fun k => LT.lt (ν (t' k)) (μ (t' k))) n
          ⊢ Eq (ν (t' ↑⟨n, hn⟩)) (Min.min (μ (t' ↑⟨n, hn⟩)) (ν (t' ↑⟨n, hn⟩)))
        -/
        rw [mem_setOf_eq] at hn
        /-
          case refine_2.refine_2
          α : Type u_1
          m0 : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          hs : MeasurableSet s
          t' : Nat → Set α
          ht' : HasSubset.Subset s (Set.iUnion t')
          t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
          ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
          hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
          hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
          hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
          heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
          n : Nat
          hn✝ : Membership.mem (setOf fun k => LT.lt (ν (t' k)) (μ (t' k))) n
          hn : LT.lt (ν (t' n)) (μ (t' n))
          ⊢ Eq (ν (t' ↑⟨n, hn✝⟩)) (Min.min (μ (t' ↑⟨n, hn✝⟩)) (ν (t' ↑⟨n, hn✝⟩)))
        -/
        simp [le_of_lt hn]
        /-
          🎉 no goals
        -/
      /-
        α : Type u_1
        m0 : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        hs : MeasurableSet s
        t' : Nat → Set α
        ht' : HasSubset.Subset s (Set.iUnion t')
        t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
        ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
        hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
        hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
        hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
        heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
        ⊢ Disjoint (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun k => LT.lt  …
      -/
    · rw [Set.disjoint_iff]
      /-
        α : Type u_1
        m0 : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        hs : MeasurableSet s
        t' : Nat → Set α
        ht' : HasSubset.Subset s (Set.iUnion t')
        t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
        ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
        hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
        hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
        hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
        heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
        ⊢ HasSubset.Subset (Inter.inter (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) ( …
      -/
      rintro k ⟨hk₁, hk₂⟩
      /-
        case intro
        α : Type u_1
        m0 : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        hs : MeasurableSet s
        t' : Nat → Set α
        ht' : HasSubset.Subset s (Set.iUnion t')
        t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
        ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
        hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
        hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
        hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
        heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
        k : Nat
        hk₁ : Membership.mem (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) k
        hk₂ : Membership.mem (setOf fun k => LT.lt (ν (t' k)) (μ (t' k))) k
        ⊢ Membership.mem EmptyCollection.emptyCollection k
      -/
      rw [mem_setOf_eq] at hk₁ hk₂
      /-
        case intro
        α : Type u_1
        m0 : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        hs : MeasurableSet s
        t' : Nat → Set α
        ht' : HasSubset.Subset s (Set.iUnion t')
        t : Set α := Set.iUnion fun n => Set.iUnion fun h => t' n
        ht : Eq t (Set.iUnion fun n => Set.iUnion fun h => t' n)
        hle₁ : LE.le (μ (Inter.inter t s)) (tsum fun n => μ (t' ↑n))
        hcap : HasSubset.Subset (Inter.inter (HasCompl.compl t) s) (Set.iUnion fun n = …
        hle₂ : LE.le (ν (Inter.inter (HasCompl.compl t) s)) (tsum fun n => ν (t' ↑n))
        heq : Eq (Union.union (setOf fun k => LE.le (μ (t' k)) (ν (t' k))) (setOf fun  …
        k : Nat
        hk₁ : LE.le (μ (t' k)) (ν (t' k))
        hk₂ : LT.lt (ν (t' k)) (μ (t' k))
        ⊢ Membership.mem EmptyCollection.emptyCollection k
      -/
      exact False.elim <| hk₂.not_le hk₁
      /-
        🎉 no goals
      -/


@[simp]
theorem _root_.MeasureTheory.OuterMeasure.toMeasure_top :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         γ : Type u_3
                                         δ : Type u_4
                                         ι : Type u_5
                                         R : Type u_6
                                         R' : Type u_7
                                         m0 : MeasurableSpace α
                                         mβ : MeasurableSpace β
                                         inst✝ : MeasurableSpace γ
                                         μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
                                         s s' t : Set α
                                         ⊢ LE.le m0 Top.top.caratheodory
                                       -/
    (⊤ : OuterMeasure α).toMeasure (by rw [OuterMeasure.top_caratheodory]; exact le_top) =
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
      (⊤ : Measure α) :=
  toOuterMeasure_toMeasure (μ := ⊤)


@[simp]
theorem toOuterMeasure_top {_ : MeasurableSpace α} :
    (⊤ : Measure α).toOuterMeasure = (⊤ : OuterMeasure α) :=
  rfl


@[simp]
theorem top_add : ⊤ + μ = ⊤ :=
  top_unique <| Measure.le_add_right le_rfl


@[simp]
theorem add_top : μ + ⊤ = ⊤ :=
  top_unique <| Measure.le_add_left le_rfl


protected theorem zero_le {_m0 : MeasurableSpace α} (μ : Measure α) : 0 ≤ μ :=
  bot_le


theorem nonpos_iff_eq_zero' : μ ≤ 0 ↔ μ = 0 :=
  μ.zero_le.le_iff_eq


@[simp]
theorem measure_univ_eq_zero : μ univ = 0 ↔ μ = 0 :=
  ⟨fun h => bot_unique fun s => (h ▸ measure_mono (subset_univ s) : μ s ≤ 0), fun h =>
    h.symm ▸ rfl⟩


theorem measure_univ_ne_zero : μ univ ≠ 0 ↔ μ ≠ 0 :=
  measure_univ_eq_zero.not


instance [NeZero μ] : NeZero (μ univ) := ⟨measure_univ_ne_zero.2 <| NeZero.ne μ⟩


@[simp]
theorem measure_univ_pos : 0 < μ univ ↔ μ ≠ 0 :=
  pos_iff_ne_zero.trans measure_univ_ne_zero


lemma nonempty_of_neZero (μ : Measure α) [NeZero μ] : Nonempty α :=
  (isEmpty_or_nonempty α).resolve_left fun h ↦ by
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NeZero μ
      h : IsEmpty α
      ⊢ False
    -/
    simpa [eq_empty_of_isEmpty] using NeZero.ne (μ univ)
    /-
      🎉 no goals
    -/


/-- Lift a linear map between `OuterMeasure` spaces such that for each measure `μ` every measurable
set is caratheodory-measurable w.r.t. `f μ` to a linear map between `Measure` spaces. -/
def liftLinear [MeasurableSpace β] (f : OuterMeasure α →ₗ[ℝ≥0∞] OuterMeasure β)
    (hf : ∀ μ : Measure α, ‹_› ≤ (f μ.toOuterMeasure).caratheodory) :
    Measure α →ₗ[ℝ≥0∞] Measure β where
  toFun μ := (f μ.toOuterMeasure).toMeasure (hf μ)
  map_add' μ₁ μ₂ := ext fun s hs => by
    simp only [map_add, coe_add, Pi.add_apply, toMeasure_apply, add_toOuterMeasure,
      OuterMeasure.coe_add, hs]
  map_smul' c μ := ext fun s hs => by
    simp only [LinearMap.map_smulₛₗ, coe_smul, Pi.smul_apply,
      toMeasure_apply, smul_toOuterMeasure (R := ℝ≥0∞), OuterMeasure.coe_smul (R := ℝ≥0∞),
      smul_apply, hs]


lemma liftLinear_apply₀ {f : OuterMeasure α →ₗ[ℝ≥0∞] OuterMeasure β} (hf) {s : Set β}
    (hs : NullMeasurableSet s (liftLinear f hf μ)) : liftLinear f hf μ s = f μ.toOuterMeasure s :=
  toMeasure_apply₀ _ (hf μ) hs


@[simp]
theorem liftLinear_apply {f : OuterMeasure α →ₗ[ℝ≥0∞] OuterMeasure β} (hf) {s : Set β}
    (hs : MeasurableSet s) : liftLinear f hf μ s = f μ.toOuterMeasure s :=
  toMeasure_apply _ (hf μ) hs


theorem le_liftLinear_apply {f : OuterMeasure α →ₗ[ℝ≥0∞] OuterMeasure β} (hf) (s : Set β) :
    f μ.toOuterMeasure s ≤ liftLinear f hf μ s :=
  le_toMeasure_apply _ (hf μ) s


open Classical in
/-- The pushforward of a measure as a linear map. It is defined to be `0` if `f` is not
a measurable function. -/
def mapₗ [MeasurableSpace α] [MeasurableSpace β] (f : α → β) : Measure α →ₗ[ℝ≥0∞] Measure β :=
  if hf : Measurable f then
    liftLinear (OuterMeasure.map f) fun μ _s hs t =>
      le_toOuterMeasure_caratheodory μ _ (hf hs) (f ⁻¹' t)
  else 0


theorem mapₗ_congr {f g : α → β} (hf : Measurable f) (hg : Measurable g) (h : f =ᵐ[μ] g) :
    mapₗ f μ = mapₗ g μ := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f g : α → β
    hf : Measurable f
    hg : Measurable g
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Eq ((MeasureTheory.Measure.mapₗ f) μ) ((MeasureTheory.Measure.mapₗ g) μ)
  -/
  ext1 s hs
  simpa only [mapₗ, hf, hg, hs, dif_pos, liftLinear_apply, OuterMeasure.map_apply]
    using measure_congr (h.preimage s)


open Classical in
/-- The pushforward of a measure. It is defined to be `0` if `f` is not an almost everywhere
measurable function. -/
irreducible_def map [MeasurableSpace α] [MeasurableSpace β] (f : α → β) (μ : Measure α) :
    Measure β :=
  if hf : AEMeasurable f μ then mapₗ (hf.mk f) μ else 0


theorem mapₗ_mk_apply_of_aemeasurable {f : α → β} (hf : AEMeasurable f μ) :
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       m0 : MeasurableSpace α
                                       mβ : MeasurableSpace β
                                       μ : MeasureTheory.Measure α
                                       f : α → β
                                       hf : AEMeasurable f μ
                                       ⊢ Eq ((MeasureTheory.Measure.mapₗ (AEMeasurable.mk f hf)) μ) (MeasureTheory.Me …
                                     -/
    mapₗ (hf.mk f) μ = map f μ := by simp [map, hf]
                                     /-
                                       🎉 no goals
                                     -/


theorem mapₗ_apply_of_measurable {f : α → β} (hf : Measurable f) (μ : Measure α) :
    mapₗ f μ = map f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    μ : MeasureTheory.Measure α
    ⊢ Eq ((MeasureTheory.Measure.mapₗ f) μ) (MeasureTheory.Measure.map f μ)
  -/
  simp only [← mapₗ_mk_apply_of_aemeasurable hf.aemeasurable]
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    μ : MeasureTheory.Measure α
    ⊢ Eq ((MeasureTheory.Measure.mapₗ f) μ) ((MeasureTheory.Measure.mapₗ (AEMeasur …
  -/
  exact mapₗ_congr hf hf.aemeasurable.measurable_mk hf.aemeasurable.ae_eq_mk
  /-
    🎉 no goals
  -/


@[simp]
theorem map_add (μ ν : Measure α) {f : α → β} (hf : Measurable f) :
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              m0 : MeasurableSpace α
                                              mβ : MeasurableSpace β
                                              μ ν : MeasureTheory.Measure α
                                              f : α → β
                                              hf : Measurable f
                                              ⊢ Eq (MeasureTheory.Measure.map f (HAdd.hAdd μ ν)) (HAdd.hAdd (MeasureTheory.M …
                                            -/
    (μ + ν).map f = μ.map f + ν.map f := by simp [← mapₗ_apply_of_measurable hf]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem map_zero (f : α → β) : (0 : Measure α).map f = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    ⊢ Eq (MeasureTheory.Measure.map f 0) 0
  -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  by_cases hf : AEMeasurable f (0 : Measure α) <;> simp [map, hf]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem map_of_not_aemeasurable {f : α → β} {μ : Measure α} (hf : ¬AEMeasurable f μ) :
                      /-
                        α : Type u_1
                        β : Type u_2
                        m0 : MeasurableSpace α
                        mβ : MeasurableSpace β
                        f : α → β
                        μ : MeasureTheory.Measure α
                        hf : Not (AEMeasurable f μ)
                        ⊢ Eq (MeasureTheory.Measure.map f μ) 0
                      -/
    μ.map f = 0 := by simp [map, hf]
                      /-
                        🎉 no goals
                      -/


theorem _root_.AEMeasurable.of_map_ne_zero {f : α → β} {μ : Measure α} (hf : μ.map f ≠ 0) :
    AEMeasurable f μ := not_imp_comm.1 map_of_not_aemeasurable hf


theorem map_congr {f g : α → β} (h : f =ᵐ[μ] g) : Measure.map f μ = Measure.map g μ := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f g : α → β
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Eq (MeasureTheory.Measure.map f μ) (MeasureTheory.Measure.map g μ)
  -/
  by_cases hf : AEMeasurable f μ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f g : α → β
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : AEMeasurable f μ
      ⊢ Eq (MeasureTheory.Measure.map f μ) (MeasureTheory.Measure.map g μ)
    -/
  · have hg : AEMeasurable g μ := hf.congr h
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f g : α → β
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : AEMeasurable f μ
      hg : AEMeasurable g μ
      ⊢ Eq (MeasureTheory.Measure.map f μ) (MeasureTheory.Measure.map g μ)
    -/
    simp only [← mapₗ_mk_apply_of_aemeasurable hf, ← mapₗ_mk_apply_of_aemeasurable hg]
    exact
      mapₗ_congr hf.measurable_mk hg.measurable_mk (hf.ae_eq_mk.symm.trans (h.trans hg.ae_eq_mk))
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f g : α → β
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : Not (AEMeasurable f μ)
      ⊢ Eq (MeasureTheory.Measure.map f μ) (MeasureTheory.Measure.map g μ)
    -/
  · have hg : ¬AEMeasurable g μ := by simpa [← aemeasurable_congr h] using hf
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f g : α → β
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : Not (AEMeasurable f μ)
      hg : Not (AEMeasurable g μ)
      ⊢ Eq (MeasureTheory.Measure.map f μ) (MeasureTheory.Measure.map g μ)
    -/
    simp [map_of_not_aemeasurable, hf, hg]
    /-
      🎉 no goals
    -/


@[simp]
protected theorem map_smul {R : Type*} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    (c : R) (μ : Measure α) (f : α → β) : (c • μ).map f = c • μ.map f := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    R : Type u_8
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    c : R
    μ : MeasureTheory.Measure α
    f : α → β
    ⊢ Eq (MeasureTheory.Measure.map f (HSMul.hSMul c μ)) (HSMul.hSMul c (MeasureTh …
  -/
  suffices ∀ c : ℝ≥0∞, (c • μ).map f = c • μ.map f by simpa using this (c • 1)
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    R : Type u_8
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    c : R
    μ : MeasureTheory.Measure α
    f : α → β
    ⊢ ∀ (c : ENNReal), Eq (MeasureTheory.Measure.map f (HSMul.hSMul c μ)) (HSMul.h …
  -/
  clear c; intro c
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    R : Type u_8
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    μ : MeasureTheory.Measure α
    f : α → β
    c : ENNReal
    ⊢ Eq (MeasureTheory.Measure.map f (HSMul.hSMul c μ)) (HSMul.hSMul c (MeasureTh …
  -/
  rcases eq_or_ne c 0 with (rfl | hc); · simp
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    R : Type u_8
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    μ : MeasureTheory.Measure α
    f : α → β
    c : ENNReal
    hc : Ne c 0
    ⊢ Eq (MeasureTheory.Measure.map f (HSMul.hSMul c μ)) (HSMul.hSMul c (MeasureTh …
  -/
  by_cases hf : AEMeasurable f μ
  · have hfc : AEMeasurable f (c • μ) :=
      ⟨hf.mk f, hf.measurable_mk, (ae_smul_measure_iff hc).2 hf.ae_eq_mk⟩
    simp only [← mapₗ_mk_apply_of_aemeasurable hf, ← mapₗ_mk_apply_of_aemeasurable hfc,
      LinearMap.map_smulₛₗ, RingHom.id_apply]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      R : Type u_8
      inst✝¹ : SMul R ENNReal
      inst✝ : IsScalarTower R ENNReal ENNReal
      μ : MeasureTheory.Measure α
      f : α → β
      c : ENNReal
      hc : Ne c 0
      hf : AEMeasurable f μ
      hfc : AEMeasurable f (HSMul.hSMul c μ)
      ⊢ Eq (HSMul.hSMul c ((MeasureTheory.Measure.mapₗ (AEMeasurable.mk f hfc)) μ))  …
    -/
    congr 1
    /-
      case pos.e_a
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      R : Type u_8
      inst✝¹ : SMul R ENNReal
      inst✝ : IsScalarTower R ENNReal ENNReal
      μ : MeasureTheory.Measure α
      f : α → β
      c : ENNReal
      hc : Ne c 0
      hf : AEMeasurable f μ
      hfc : AEMeasurable f (HSMul.hSMul c μ)
      ⊢ Eq ((MeasureTheory.Measure.mapₗ (AEMeasurable.mk f hfc)) μ) ((MeasureTheory. …
    -/
    apply mapₗ_congr hfc.measurable_mk hf.measurable_mk
    /-
      case pos.e_a
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      R : Type u_8
      inst✝¹ : SMul R ENNReal
      inst✝ : IsScalarTower R ENNReal ENNReal
      μ : MeasureTheory.Measure α
      f : α → β
      c : ENNReal
      hc : Ne c 0
      hf : AEMeasurable f μ
      hfc : AEMeasurable f (HSMul.hSMul c μ)
      ⊢ (MeasureTheory.ae μ).EventuallyEq (AEMeasurable.mk f hfc) (AEMeasurable.mk f …
    -/
    exact EventuallyEq.trans ((ae_smul_measure_iff hc).1 hfc.ae_eq_mk.symm) hf.ae_eq_mk
    /-
      🎉 no goals
    -/
  · have hfc : ¬AEMeasurable f (c • μ) := by
      intro hfc
      exact hf ⟨hfc.mk f, hfc.measurable_mk, (ae_smul_measure_iff hc).1 hfc.ae_eq_mk⟩
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      R : Type u_8
      inst✝¹ : SMul R ENNReal
      inst✝ : IsScalarTower R ENNReal ENNReal
      μ : MeasureTheory.Measure α
      f : α → β
      c : ENNReal
      hc : Ne c 0
      hf : Not (AEMeasurable f μ)
      hfc : Not (AEMeasurable f (HSMul.hSMul c μ))
      ⊢ Eq (MeasureTheory.Measure.map f (HSMul.hSMul c μ)) (HSMul.hSMul c (MeasureTh …
    -/
    simp [map_of_not_aemeasurable hf, map_of_not_aemeasurable hfc]
    /-
      🎉 no goals
    -/



@[deprecated Measure.map_smul (since := "2024-11-13")]
protected theorem map_smul_nnreal (c : ℝ≥0) (μ : Measure α) (f : α → β) :
    (c • μ).map f = c • μ.map f :=
  μ.map_smul c f


lemma map_apply₀ {f : α → β} (hf : AEMeasurable f μ) {s : Set β}
    (hs : NullMeasurableSet s (map f μ)) : μ.map f s = μ (f ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f μ
    s : Set β
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.map f μ)
    ⊢ Eq ((MeasureTheory.Measure.map f μ) s) (μ (Set.preimage f s))
  -/
  rw [map, dif_pos hf, mapₗ, dif_pos hf.measurable_mk] at hs ⊢
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f μ
    s : Set β
    hs : MeasureTheory.NullMeasurableSet s ((MeasureTheory.Measure.liftLinear (Mea …
    ⊢ Eq (((MeasureTheory.Measure.liftLinear (MeasureTheory.OuterMeasure.map (AEMe …
  -/
  rw [liftLinear_apply₀ _ hs, measure_congr (hf.ae_eq_mk.preimage s)]
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f μ
    s : Set β
    hs : MeasureTheory.NullMeasurableSet s ((MeasureTheory.Measure.liftLinear (Mea …
    ⊢ Eq (((MeasureTheory.OuterMeasure.map (AEMeasurable.mk f hf)) μ.toOuterMeasur …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- We can evaluate the pushforward on measurable sets. For non-measurable sets, see
  `MeasureTheory.Measure.le_map_apply` and `MeasurableEquiv.map_apply`. -/
@[simp]
theorem map_apply_of_aemeasurable (hf : AEMeasurable f μ) {s : Set β} (hs : MeasurableSet s) :
    μ.map f s = μ (f ⁻¹' s) := map_apply₀ hf hs.nullMeasurableSet


@[simp]
theorem map_apply (hf : Measurable f) {s : Set β} (hs : MeasurableSet s) :
    μ.map f s = μ (f ⁻¹' s) :=
  map_apply_of_aemeasurable hf.aemeasurable hs


theorem map_toOuterMeasure (hf : AEMeasurable f μ) :
    (μ.map f).toOuterMeasure = (OuterMeasure.map f μ.toOuterMeasure).trim := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f μ
    ⊢ Eq (MeasureTheory.Measure.map f μ).toOuterMeasure ((MeasureTheory.OuterMeasu …
  -/
  rw [← trimmed, OuterMeasure.trim_eq_trim_iff]
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f μ
    ⊢ ∀ (s : Set β), MeasurableSet s → Eq ((MeasureTheory.Measure.map f μ).toOuter …
  -/
  intro s hs
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f μ
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.map f μ).toOuterMeasure s) (((MeasureTheory.Outer …
  -/
  simp [hf, hs]
  /-
    🎉 no goals
  -/


@[simp] lemma map_eq_zero_iff (hf : AEMeasurable f μ) : μ.map f = 0 ↔ μ = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f μ
    ⊢ Iff (Eq (MeasureTheory.Measure.map f μ) 0) (Eq μ 0)
  -/
  simp_rw [← measure_univ_eq_zero, map_apply_of_aemeasurable hf .univ, preimage_univ]
  /-
    🎉 no goals
  -/


@[simp] lemma mapₗ_eq_zero_iff (hf : Measurable f) : Measure.mapₗ f μ = 0 ↔ μ = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : Measurable f
    ⊢ Iff (Eq ((MeasureTheory.Measure.mapₗ f) μ) 0) (Eq μ 0)
  -/
  rw [mapₗ_apply_of_measurable hf, map_eq_zero_iff hf.aemeasurable]
  /-
    🎉 no goals
  -/


/-- If `map f μ = μ`, then the measure of the preimage of any null measurable set `s`
is equal to the measure of `s`.
Note that this lemma does not assume (a.e.) measurability of `f`. -/
lemma measure_preimage_of_map_eq_self {f : α → α} (hf : map f μ = μ)
    {s : Set α} (hs : NullMeasurableSet s μ) : μ (f ⁻¹' s) = μ s := by
  if hfm : AEMeasurable f μ then
    rw [← map_apply₀ hfm, hf]
    rwa [hf]
  else
    rw [map_of_not_aemeasurable hfm] at hf
    simp [← hf]


lemma map_ne_zero_iff (hf : AEMeasurable f μ) : μ.map f ≠ 0 ↔ μ ≠ 0 := (map_eq_zero_iff hf).not

lemma mapₗ_ne_zero_iff (hf : Measurable f) : Measure.mapₗ f μ ≠ 0 ↔ μ ≠ 0 :=
  (mapₗ_eq_zero_iff hf).not


@[simp]
theorem map_id : map id μ = μ :=
  ext fun _ => map_apply measurable_id


@[simp]
theorem map_id' : map (fun x => x) μ = μ :=
  map_id


/-- Mapping a measure twice is the same as mapping the measure with the composition. This version is
for measurable functions. See `map_map_of_aemeasurable` when they are just ae measurable. -/
theorem map_map {g : β → γ} {f : α → β} (hg : Measurable g) (hf : Measurable f) :
    (μ.map f).map g = μ.map (g ∘ f) :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       m0 : MeasurableSpace α
                       mβ : MeasurableSpace β
                       inst✝ : MeasurableSpace γ
                       μ : MeasureTheory.Measure α
                       g : β → γ
                       f : α → β
                       hg : Measurable g
                       hf : Measurable f
                       s : Set γ
                       hs : MeasurableSet s
                       ⊢ Eq ((MeasureTheory.Measure.map g (MeasureTheory.Measure.map f μ)) s) ((Measu …
                     -/
  ext fun s hs => by simp [hf, hg, hs, hg hs, hg.comp hf, ← preimage_comp]
                     /-
                       🎉 no goals
                     -/


@[mono]
theorem map_mono {f : α → β} (h : μ ≤ ν) (hf : Measurable f) : μ.map f ≤ ν.map f :=
                         /-
                           α : Type u_1
                           β : Type u_2
                           m0 : MeasurableSpace α
                           mβ : MeasurableSpace β
                           μ ν : MeasureTheory.Measure α
                           f : α → β
                           h : LE.le μ ν
                           hf : Measurable f
                           s : Set β
                           hs : MeasurableSet s
                           ⊢ LE.le ((MeasureTheory.Measure.map f μ) s) ((MeasureTheory.Measure.map f ν) s)
                         -/
  le_iff.2 fun s hs ↦ by simp [hf.aemeasurable, hs, h _]
                         /-
                           🎉 no goals
                         -/


/-- Even if `s` is not measurable, we can bound `map f μ s` from below.
  See also `MeasurableEquiv.map_apply`. -/
theorem le_map_apply {f : α → β} (hf : AEMeasurable f μ) (s : Set β) : μ (f ⁻¹' s) ≤ μ.map f s :=
  calc
                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             m0 : MeasurableSpace α
                                                             mβ : MeasurableSpace β
                                                             μ : MeasureTheory.Measure α
                                                             f : α → β
                                                             hf : AEMeasurable f μ
                                                             s : Set β
                                                             ⊢ LE.le (μ (Set.preimage f s)) (μ (Set.preimage f (MeasureTheory.toMeasurable  …
                                                           -/
    μ (f ⁻¹' s) ≤ μ (f ⁻¹' toMeasurable (μ.map f) s) := by gcongr; apply subset_toMeasurable
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    _ = μ.map f (toMeasurable (μ.map f) s) :=
      (map_apply_of_aemeasurable hf <| measurableSet_toMeasurable _ _).symm
    _ = μ.map f s := measure_toMeasurable _


theorem le_map_apply_image {f : α → β} (hf : AEMeasurable f μ) (s : Set α) :
    μ s ≤ μ.map f (f '' s) :=
  (measure_mono (subset_preimage_image f s)).trans (le_map_apply hf _)


/-- Even if `s` is not measurable, `map f μ s = 0` implies that `μ (f ⁻¹' s) = 0`. -/
theorem preimage_null_of_map_null {f : α → β} (hf : AEMeasurable f μ) {s : Set β}
    (hs : μ.map f s = 0) : μ (f ⁻¹' s) = 0 :=
  nonpos_iff_eq_zero.mp <| (le_map_apply hf s).trans_eq hs


theorem tendsto_ae_map {f : α → β} (hf : AEMeasurable f μ) : Tendsto f (ae μ) (ae (μ.map f)) :=
  fun _ hs => preimage_null_of_map_null hf hs


/-- Sum of an indexed family of measures. -/
noncomputable def sum (f : ι → Measure α) : Measure α :=
  (OuterMeasure.sum fun i => (f i).toOuterMeasure).toMeasure <|
    le_trans (le_iInf fun _ => le_toOuterMeasure_caratheodory _)
      (OuterMeasure.le_sum_caratheodory _)


theorem le_sum_apply (f : ι → Measure α) (s : Set α) : ∑' i, f i s ≤ sum f s :=
  le_toMeasure_apply _ _ _


@[simp]
theorem sum_apply (f : ι → Measure α) {s : Set α} (hs : MeasurableSet s) :
    sum f s = ∑' i, f i s :=
  toMeasure_apply _ _ hs


theorem sum_apply₀ (f : ι → Measure α) {s : Set α} (hs : NullMeasurableSet s (sum f)) :
    sum f s = ∑' i, f i s := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    f : ι → MeasureTheory.Measure α
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.sum f)
    ⊢ Eq ((MeasureTheory.Measure.sum f) s) (tsum fun i => (f i) s)
  -/
  apply le_antisymm ?_ (le_sum_apply _ _)
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    f : ι → MeasureTheory.Measure α
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.sum f)
    ⊢ LE.le ((MeasureTheory.Measure.sum f) s) (tsum fun i => (f i) s)
  -/
  rcases hs.exists_measurable_subset_ae_eq with ⟨t, ts, t_meas, ht⟩
  calc
  sum f s = sum f t := measure_congr ht.symm
  _ = ∑' i, f i t := sum_apply _ t_meas
  _ ≤ ∑' i, f i s := ENNReal.tsum_le_tsum fun i ↦ measure_mono ts


theorem sum_apply_of_countable [Countable ι] (f : ι → Measure α) (s : Set α) :
    sum f s = ∑' i, f i s := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    inst✝ : Countable ι
    f : ι → MeasureTheory.Measure α
    s : Set α
    ⊢ Eq ((MeasureTheory.Measure.sum f) s) (tsum fun i => (f i) s)
  -/
  apply le_antisymm ?_ (le_sum_apply _ _)
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    inst✝ : Countable ι
    f : ι → MeasureTheory.Measure α
    s : Set α
    ⊢ LE.le ((MeasureTheory.Measure.sum f) s) (tsum fun i => (f i) s)
  -/
  rcases exists_measurable_superset_forall_eq f s with ⟨t, hst, htm, ht⟩
  calc
  sum f s ≤ sum f t := measure_mono hst
  _ = ∑' i, f i t := sum_apply _ htm
  _ = ∑' i, f i s := by simp [ht]


theorem le_sum (μ : ι → Measure α) (i : ι) : μ i ≤ sum μ :=
                         /-
                           α : Type u_1
                           ι : Type u_5
                           m0 : MeasurableSpace α
                           μ : ι → MeasureTheory.Measure α
                           i : ι
                           s : Set α
                           hs : MeasurableSet s
                           ⊢ LE.le ((μ i) s) ((MeasureTheory.Measure.sum μ) s)
                         -/
  le_iff.2 fun s hs ↦ by simpa only [sum_apply μ hs] using ENNReal.le_tsum i
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem sum_apply_eq_zero [Countable ι] {μ : ι → Measure α} {s : Set α} :
    sum μ s = 0 ↔ ∀ i, μ i s = 0 := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure α
    s : Set α
    ⊢ Iff (Eq ((MeasureTheory.Measure.sum μ) s) 0) (∀ (i : ι), Eq ((μ i) s) 0)
  -/
  simp [sum_apply_of_countable]
  /-
    🎉 no goals
  -/


theorem sum_apply_eq_zero' {μ : ι → Measure α} {s : Set α} (hs : MeasurableSet s) :
                                       /-
                                         α : Type u_1
                                         ι : Type u_5
                                         m0 : MeasurableSpace α
                                         μ : ι → MeasureTheory.Measure α
                                         s : Set α
                                         hs : MeasurableSet s
                                         ⊢ Iff (Eq ((MeasureTheory.Measure.sum μ) s) 0) (∀ (i : ι), Eq ((μ i) s) 0)
                                       -/
    sum μ s = 0 ↔ ∀ i, μ i s = 0 := by simp [hs]
                                       /-
                                         🎉 no goals
                                       -/


@[simp] lemma sum_eq_zero : sum f = 0 ↔ ∀ i, f i = 0 := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    f : ι → MeasureTheory.Measure α
    ⊢ Iff (Eq (MeasureTheory.Measure.sum f) 0) (∀ (i : ι), Eq (f i) 0)
  -/
  simp +contextual [Measure.ext_iff, forall_swap (α := ι)]
  /-
    🎉 no goals
  -/


@[simp]
lemma sum_zero : Measure.sum (fun (_ : ι) ↦ (0 : Measure α)) = 0 := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ⊢ Eq (MeasureTheory.Measure.sum fun x => 0) 0
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.sum fun x => 0) s) (0 s)
  -/
  simp [Measure.sum_apply _ hs]
  /-
    🎉 no goals
  -/


theorem sum_sum {ι' : Type*} (μ : ι → ι' → Measure α) :
    (sum fun n => sum (μ n)) = sum (fun (p : ι × ι') ↦ μ p.1 p.2) := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ι' : Type u_8
    μ : ι → ι' → MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.sum fun n => MeasureTheory.Measure.sum (μ n)) (Mea …
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ι' : Type u_8
    μ : ι → ι' → MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.sum fun n => MeasureTheory.Measure.sum (μ n)) s)  …
  -/
  simp [sum_apply _ hs, ENNReal.tsum_prod']
  /-
    🎉 no goals
  -/


theorem sum_comm {ι' : Type*} (μ : ι → ι' → Measure α) :
    (sum fun n => sum (μ n)) = sum fun m => sum fun n => μ n m := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ι' : Type u_8
    μ : ι → ι' → MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.sum fun n => MeasureTheory.Measure.sum (μ n)) (Mea …
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ι' : Type u_8
    μ : ι → ι' → MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.sum fun n => MeasureTheory.Measure.sum (μ n)) s)  …
  -/
  simp_rw [sum_apply _ hs]
  /-
    case h
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ι' : Type u_8
    μ : ι → ι' → MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (tsum fun i => tsum fun i_1 => (μ i i_1) s) (tsum fun i => tsum fun i_1 = …
  -/
  rw [ENNReal.tsum_comm]
  /-
    🎉 no goals
  -/


theorem ae_sum_iff [Countable ι] {μ : ι → Measure α} {p : α → Prop} :
    (∀ᵐ x ∂sum μ, p x) ↔ ∀ i, ∀ᵐ x ∂μ i, p x :=
  sum_apply_eq_zero


theorem ae_sum_iff' {μ : ι → Measure α} {p : α → Prop} (h : MeasurableSet { x | p x }) :
    (∀ᵐ x ∂sum μ, p x) ↔ ∀ i, ∀ᵐ x ∂μ i, p x :=
  sum_apply_eq_zero' h.compl


@[simp]
theorem sum_fintype [Fintype ι] (μ : ι → Measure α) : sum μ = ∑ i, μ i := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    inst✝ : Fintype ι
    μ : ι → MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.sum μ) (Finset.univ.sum fun i => μ i)
  -/
  ext1 s hs
  /-
    case h
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    inst✝ : Fintype ι
    μ : ι → MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.sum μ) s) ((Finset.univ.sum fun i => μ i) s)
  -/
  simp only [sum_apply, finset_sum_apply, hs, tsum_fintype]
  /-
    🎉 no goals
  -/


theorem sum_coe_finset (s : Finset ι) (μ : ι → Measure α) :
                                                /-
                                                  α : Type u_1
                                                  ι : Type u_5
                                                  m0 : MeasurableSpace α
                                                  s : Finset ι
                                                  μ : ι → MeasureTheory.Measure α
                                                  ⊢ Eq (MeasureTheory.Measure.sum fun i => μ ↑i) (s.sum fun i => μ i)
                                                -/
    (sum fun i : s => μ i) = ∑ i ∈ s, μ i := by rw [sum_fintype, Finset.sum_coe_sort s μ]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem ae_sum_eq [Countable ι] (μ : ι → Measure α) : ae (sum μ) = ⨆ i, ae (μ i) :=
  Filter.ext fun _ => ae_sum_iff.trans mem_iSup.symm


theorem sum_bool (f : Bool → Measure α) : sum f = f true + f false := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    f : Bool → MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.sum f) (HAdd.hAdd (f Bool.true) (f Bool.false))
  -/
  rw [sum_fintype, Fintype.sum_bool]
  /-
    🎉 no goals
  -/


theorem sum_cond (μ ν : Measure α) : (sum fun b => cond b μ ν) = μ + ν :=
  sum_bool _


@[simp]
theorem sum_of_isEmpty [IsEmpty ι] (μ : ι → Measure α) : sum μ = 0 := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    inst✝ : IsEmpty ι
    μ : ι → MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.sum μ) 0
  -/
  rw [← measure_univ_eq_zero, sum_apply _ MeasurableSet.univ, tsum_empty]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-11")] alias sum_of_empty := sum_of_isEmpty


theorem sum_add_sum_compl (s : Set ι) (μ : ι → Measure α) :
    ((sum fun i : s => μ i) + sum fun i : ↥sᶜ => μ i) = sum μ := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    s : Set ι
    μ : ι → MeasureTheory.Measure α
    ⊢ Eq (HAdd.hAdd (MeasureTheory.Measure.sum fun i => μ ↑i) (MeasureTheory.Measu …
  -/
  ext1 t ht
  /-
    case h
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    s : Set ι
    μ : ι → MeasureTheory.Measure α
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq ((HAdd.hAdd (MeasureTheory.Measure.sum fun i => μ ↑i) (MeasureTheory.Meas …
  -/
  simp only [add_apply, sum_apply _ ht]
  /-
    case h
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    s : Set ι
    μ : ι → MeasureTheory.Measure α
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq (HAdd.hAdd (tsum fun i => (μ ↑i) t) (tsum fun i => (μ ↑i) t)) (tsum fun i …
  -/
  exact tsum_add_tsum_compl (f := fun i => μ i t) ENNReal.summable ENNReal.summable
  /-
    🎉 no goals
  -/


theorem sum_congr {μ ν : ℕ → Measure α} (h : ∀ n, μ n = ν n) : sum μ = sum ν :=
  congr_arg sum (funext h)


theorem sum_add_sum {ι : Type*} (μ ν : ι → Measure α) : sum μ + sum ν = sum fun n => μ n + ν n := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    ι : Type u_8
    μ ν : ι → MeasureTheory.Measure α
    ⊢ Eq (HAdd.hAdd (MeasureTheory.Measure.sum μ) (MeasureTheory.Measure.sum ν)) ( …
  -/
  ext1 s hs
  simp only [add_apply, sum_apply _ hs, Pi.add_apply, coe_add,
    tsum_add ENNReal.summable ENNReal.summable]


@[simp] lemma sum_comp_equiv {ι ι' : Type*} (e : ι' ≃ ι) (m : ι → Measure α) :
    sum (m ∘ e) = sum m := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    ι : Type u_8
    ι' : Type u_9
    e : Equiv ι' ι
    m : ι → MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.sum (Function.comp m ⇑e)) (MeasureTheory.Measure.s …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    ι : Type u_8
    ι' : Type u_9
    e : Equiv ι' ι
    m : ι → MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.sum (Function.comp m ⇑e)) s) ((MeasureTheory.Meas …
  -/
  simpa [hs, sum_apply] using e.tsum_eq (fun n ↦ m n s)
  /-
    🎉 no goals
  -/


@[simp] lemma sum_extend_zero {ι ι' : Type*} {f : ι → ι'} (hf : Injective f) (m : ι → Measure α) :
    sum (Function.extend f m 0) = sum m := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    ι : Type u_8
    ι' : Type u_9
    f : ι → ι'
    hf : Function.Injective f
    m : ι → MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.sum (Function.extend f m 0)) (MeasureTheory.Measur …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    ι : Type u_8
    ι' : Type u_9
    f : ι → ι'
    hf : Function.Injective f
    m : ι → MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.sum (Function.extend f m 0)) s) ((MeasureTheory.M …
  -/
  simp [*, Function.apply_extend (fun μ : Measure α ↦ μ s)]
  /-
    🎉 no goals
  -/

/-- We say that `μ` is absolutely continuous with respect to `ν`, or that `μ` is dominated by `ν`,
  if `ν(A) = 0` implies that `μ(A) = 0`. -/
def AbsolutelyContinuous {_m0 : MeasurableSpace α} (μ ν : Measure α) : Prop :=
  ∀ ⦃s : Set α⦄, ν s = 0 → μ s = 0


@[inherit_doc MeasureTheory.Measure.AbsolutelyContinuous]
scoped[MeasureTheory] infixl:50 " ≪ " => MeasureTheory.Measure.AbsolutelyContinuous


theorem absolutelyContinuous_of_le (h : μ ≤ ν) : μ ≪ ν := fun s hs =>
  nonpos_iff_eq_zero.1 <| hs ▸ le_iff'.1 h s


alias _root_.LE.le.absolutelyContinuous := absolutelyContinuous_of_le


theorem absolutelyContinuous_of_eq (h : μ = ν) : μ ≪ ν :=
  h.le.absolutelyContinuous


alias _root_.Eq.absolutelyContinuous := absolutelyContinuous_of_eq


theorem mk (h : ∀ ⦃s : Set α⦄, MeasurableSet s → ν s = 0 → μ s = 0) : μ ≪ ν := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : ∀ ⦃s : Set α⦄, MeasurableSet s → Eq (ν s) 0 → Eq (μ s) 0
    ⊢ μ.AbsolutelyContinuous ν
  -/
  intro s hs
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : ∀ ⦃s : Set α⦄, MeasurableSet s → Eq (ν s) 0 → Eq (μ s) 0
    s : Set α
    hs : Eq (ν s) 0
    ⊢ Eq (μ s) 0
  -/
  rcases exists_measurable_superset_of_null hs with ⟨t, h1t, h2t, h3t⟩
  /-
    case intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : ∀ ⦃s : Set α⦄, MeasurableSet s → Eq (ν s) 0 → Eq (μ s) 0
    s : Set α
    hs : Eq (ν s) 0
    t : Set α
    h1t : HasSubset.Subset s t
    h2t : MeasurableSet t
    h3t : Eq (ν t) 0
    ⊢ Eq (μ s) 0
  -/
  exact measure_mono_null h1t (h h2t h3t)
  /-
    🎉 no goals
  -/


@[refl]
protected theorem refl {_m0 : MeasurableSpace α} (μ : Measure α) : μ ≪ μ :=
  rfl.absolutelyContinuous


protected theorem rfl : μ ≪ μ := fun _s hs => hs


instance instIsRefl {_ : MeasurableSpace α} : IsRefl (Measure α) (· ≪ ·) :=
  ⟨fun _ => AbsolutelyContinuous.rfl⟩


@[simp]
                                                             /-
                                                               α : Type u_1
                                                               m0 : MeasurableSpace α
                                                               μ : MeasureTheory.Measure α
                                                               x✝¹ : Set α
                                                               x✝ : Eq (μ x✝¹) 0
                                                               ⊢ Eq (0 x✝¹) 0
                                                             -/
protected lemma zero (μ : Measure α) : 0 ≪ μ := fun _ _ ↦ by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


@[trans]
protected theorem trans (h1 : μ₁ ≪ μ₂) (h2 : μ₂ ≪ μ₃) : μ₁ ≪ μ₃ := fun _s hs => h1 <| h2 hs


@[mono]
protected theorem map (h : μ ≪ ν) {f : α → β} (hf : Measurable f) : μ.map f ≪ ν.map f :=
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           m0 : MeasurableSpace α
                                           mβ : MeasurableSpace β
                                           μ ν : MeasureTheory.Measure α
                                           h : μ.AbsolutelyContinuous ν
                                           f : α → β
                                           hf : Measurable f
                                           s : Set β
                                           hs : MeasurableSet s
                                           ⊢ Eq ((MeasureTheory.Measure.map f ν) s) 0 → Eq ((MeasureTheory.Measure.map f  …
                                         -/
  AbsolutelyContinuous.mk fun s hs => by simpa [hf, hs] using @h _
                                         /-
                                           🎉 no goals
                                         -/


protected theorem smul_left [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] (h : μ ≪ ν) (c : R) :
    c • μ ≪ ν := fun s hνs => by
  /-
    α : Type u_1
    R : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    h : μ.AbsolutelyContinuous ν
    c : R
    s : Set α
    hνs : Eq (ν s) 0
    ⊢ Eq ((HSMul.hSMul c μ) s) 0
  -/
  simp only [h hνs, smul_apply, smul_zero, ← smul_one_smul ℝ≥0∞ c (0 : ℝ≥0∞)]
  /-
    🎉 no goals
  -/


/-- If `μ ≪ ν`, then `c • μ ≪ c • ν`.

Earlier, this name was used for what's now called `AbsolutelyContinuous.smul_left`. -/
protected theorem smul [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] (h : μ ≪ ν) (c : R) :
    c • μ ≪ c • ν := by
  /-
    α : Type u_1
    R : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    h : μ.AbsolutelyContinuous ν
    c : R
    ⊢ (HSMul.hSMul c μ).AbsolutelyContinuous (HSMul.hSMul c ν)
  -/
  intro s hνs
  /-
    α : Type u_1
    R : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    h : μ.AbsolutelyContinuous ν
    c : R
    s : Set α
    hνs : Eq ((HSMul.hSMul c ν) s) 0
    ⊢ Eq ((HSMul.hSMul c μ) s) 0
  -/
  rw [smul_apply, ← smul_one_smul ℝ≥0∞, smul_eq_mul, mul_eq_zero] at hνs ⊢
  /-
    α : Type u_1
    R : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    h : μ.AbsolutelyContinuous ν
    c : R
    s : Set α
    hνs : Or (Eq (HSMul.hSMul c 1) 0) (Eq (ν s) 0)
    ⊢ Or (Eq (HSMul.hSMul c 1) 0) (Eq (μ s) 0)
  -/
  exact hνs.imp_right fun hs ↦ h hs
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-14")] protected alias smul_both := AbsolutelyContinuous.smul


protected lemma add (h1 : μ₁ ≪ ν) (h2 : μ₂ ≪ ν') : μ₁ + μ₂ ≪ ν + ν' := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ₁ μ₂ ν ν' : MeasureTheory.Measure α
    h1 : μ₁.AbsolutelyContinuous ν
    h2 : μ₂.AbsolutelyContinuous ν'
    ⊢ (HAdd.hAdd μ₁ μ₂).AbsolutelyContinuous (HAdd.hAdd ν ν')
  -/
  intro s hs
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ₁ μ₂ ν ν' : MeasureTheory.Measure α
    h1 : μ₁.AbsolutelyContinuous ν
    h2 : μ₂.AbsolutelyContinuous ν'
    s : Set α
    hs : Eq ((HAdd.hAdd ν ν') s) 0
    ⊢ Eq ((HAdd.hAdd μ₁ μ₂) s) 0
  -/
  simp only [coe_add, Pi.add_apply, add_eq_zero] at hs ⊢
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ₁ μ₂ ν ν' : MeasureTheory.Measure α
    h1 : μ₁.AbsolutelyContinuous ν
    h2 : μ₂.AbsolutelyContinuous ν'
    s : Set α
    hs : And (Eq (ν s) 0) (Eq (ν' s) 0)
    ⊢ And (Eq (μ₁ s) 0) (Eq (μ₂ s) 0)
  -/
  exact ⟨h1 hs.1, h2 hs.2⟩
  /-
    🎉 no goals
  -/


lemma add_left_iff {μ₁ μ₂ ν : Measure α} :
    μ₁ + μ₂ ≪ ν ↔ μ₁ ≪ ν ∧ μ₂ ≪ ν := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ₁ μ₂ ν : MeasureTheory.Measure α
    ⊢ Iff ((HAdd.hAdd μ₁ μ₂).AbsolutelyContinuous ν) (And (μ₁.AbsolutelyContinuous …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ (h.1.add h.2).trans ?_⟩
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ₁ μ₂ ν : MeasureTheory.Measure α
      h : (HAdd.hAdd μ₁ μ₂).AbsolutelyContinuous ν
      ⊢ And (μ₁.AbsolutelyContinuous ν) (μ₂.AbsolutelyContinuous ν)
    -/
  · have : ∀ s, ν s = 0 → μ₁ s = 0 ∧ μ₂ s = 0 := by intro s hs0; simpa using h hs0
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ₁ μ₂ ν : MeasureTheory.Measure α
      h : (HAdd.hAdd μ₁ μ₂).AbsolutelyContinuous ν
      this : ∀ (s : Set α), Eq (ν s) 0 → And (Eq (μ₁ s) 0) (Eq (μ₂ s) 0)
      ⊢ And (μ₁.AbsolutelyContinuous ν) (μ₂.AbsolutelyContinuous ν)
    -/
    exact ⟨fun s hs0 ↦ (this s hs0).1, fun s hs0 ↦ (this s hs0).2⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ₁ μ₂ ν : MeasureTheory.Measure α
      h : And (μ₁.AbsolutelyContinuous ν) (μ₂.AbsolutelyContinuous ν)
      ⊢ (HAdd.hAdd ν ν).AbsolutelyContinuous ν
    -/
  · rw [← two_smul ℝ≥0]
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ₁ μ₂ ν : MeasureTheory.Measure α
      h : And (μ₁.AbsolutelyContinuous ν) (μ₂.AbsolutelyContinuous ν)
      ⊢ (HSMul.hSMul 2 ν).AbsolutelyContinuous ν
    -/
    exact AbsolutelyContinuous.rfl.smul_left 2
    /-
      🎉 no goals
    -/


lemma add_left {μ₁ μ₂ ν : Measure α} (h₁ : μ₁ ≪ ν) (h₂ : μ₂ ≪ ν) : μ₁ + μ₂ ≪ ν :=
  Measure.AbsolutelyContinuous.add_left_iff.mpr ⟨h₁, h₂⟩


lemma add_right (h1 : μ ≪ ν) (ν' : Measure α) : μ ≪ ν + ν' := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h1 : μ.AbsolutelyContinuous ν
    ν' : MeasureTheory.Measure α
    ⊢ μ.AbsolutelyContinuous (HAdd.hAdd ν ν')
  -/
  intro s hs
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h1 : μ.AbsolutelyContinuous ν
    ν' : MeasureTheory.Measure α
    s : Set α
    hs : Eq ((HAdd.hAdd ν ν') s) 0
    ⊢ Eq (μ s) 0
  -/
  simp only [coe_add, Pi.add_apply, add_eq_zero] at hs ⊢
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h1 : μ.AbsolutelyContinuous ν
    ν' : MeasureTheory.Measure α
    s : Set α
    hs : And (Eq (ν s) 0) (Eq (ν' s) 0)
    ⊢ Eq (μ s) 0
  -/
  exact h1 hs.1
  /-
    🎉 no goals
  -/


@[simp]
lemma absolutelyContinuous_zero_iff : μ ≪ 0 ↔ μ = 0 :=
  ⟨fun h ↦ measure_univ_eq_zero.mp (h rfl), fun h ↦ h.symm ▸ AbsolutelyContinuous.zero _⟩


alias absolutelyContinuous_refl := AbsolutelyContinuous.refl

alias absolutelyContinuous_rfl := AbsolutelyContinuous.rfl


lemma absolutelyContinuous_sum_left {μs : ι → Measure α} (hμs : ∀ i, μs i ≪ ν) :
    Measure.sum μs ≪ ν :=
                                            /-
                                              α : Type u_1
                                              ι : Type u_5
                                              m0 : MeasurableSpace α
                                              ν : MeasureTheory.Measure α
                                              μs : ι → MeasureTheory.Measure α
                                              hμs : ∀ (i : ι), (μs i).AbsolutelyContinuous ν
                                              s : Set α
                                              hs : MeasurableSet s
                                              hs0 : Eq (ν s) 0
                                              ⊢ Eq ((MeasureTheory.Measure.sum μs) s) 0
                                            -/
  AbsolutelyContinuous.mk fun s hs hs0 ↦ by simp [sum_apply _ hs, fun i ↦ hμs i hs0]
                                            /-
                                              🎉 no goals
                                            -/


lemma absolutelyContinuous_sum_right {μs : ι → Measure α} (i : ι) (hνμ : ν ≪ μs i) :
    ν ≪ Measure.sum μs := by
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ν : MeasureTheory.Measure α
    μs : ι → MeasureTheory.Measure α
    i : ι
    hνμ : ν.AbsolutelyContinuous (μs i)
    ⊢ ν.AbsolutelyContinuous (MeasureTheory.Measure.sum μs)
  -/
  refine AbsolutelyContinuous.mk fun s hs hs0 ↦ ?_
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ν : MeasureTheory.Measure α
    μs : ι → MeasureTheory.Measure α
    i : ι
    hνμ : ν.AbsolutelyContinuous (μs i)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq ((MeasureTheory.Measure.sum μs) s) 0
    ⊢ Eq (ν s) 0
  -/
  simp only [sum_apply _ hs, ENNReal.tsum_eq_zero] at hs0
  /-
    α : Type u_1
    ι : Type u_5
    m0 : MeasurableSpace α
    ν : MeasureTheory.Measure α
    μs : ι → MeasureTheory.Measure α
    i : ι
    hνμ : ν.AbsolutelyContinuous (μs i)
    s : Set α
    hs : MeasurableSet s
    hs0 : ∀ (i : ι), Eq ((μs i) s) 0
    ⊢ Eq (ν s) 0
  -/
  exact hνμ (hs0 i)
  /-
    🎉 no goals
  -/


lemma smul_absolutelyContinuous {c : ℝ≥0∞} : c • μ ≪ μ := .smul_left .rfl _


theorem absolutelyContinuous_of_le_smul {μ' : Measure α} {c : ℝ≥0∞} (hμ'_le : μ' ≤ c • μ) :
    μ' ≪ μ :=
  (Measure.absolutelyContinuous_of_le hμ'_le).trans smul_absolutelyContinuous


lemma absolutelyContinuous_smul {c : ℝ≥0∞} (hc : c ≠ 0) : μ ≪ c • μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c 0
    ⊢ μ.AbsolutelyContinuous (HSMul.hSMul c μ)
  -/
  simp [AbsolutelyContinuous, hc]
  /-
    🎉 no goals
  -/


theorem ae_le_iff_absolutelyContinuous : ae μ ≤ ae ν ↔ μ ≪ ν :=
  ⟨fun h s => by
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : LE.le (MeasureTheory.ae μ) (MeasureTheory.ae ν)
      s : Set α
      ⊢ Eq (ν s) 0 → Eq (μ s) 0
    -/
    rw [measure_zero_iff_ae_nmem, measure_zero_iff_ae_nmem]
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : LE.le (MeasureTheory.ae μ) (MeasureTheory.ae ν)
      s : Set α
      ⊢ Filter.Eventually (fun a => Not (Membership.mem s a)) (MeasureTheory.ae ν) → …
    -/
    exact fun hs => h hs, fun h _ hs => h hs⟩
    /-
      🎉 no goals
    -/


alias ⟨_root_.LE.le.absolutelyContinuous_of_ae, AbsolutelyContinuous.ae_le⟩ :=
  ae_le_iff_absolutelyContinuous


alias ae_mono' := AbsolutelyContinuous.ae_le


theorem AbsolutelyContinuous.ae_eq (h : μ ≪ ν) {f g : α → δ} (h' : f =ᵐ[ν] g) : f =ᵐ[μ] g :=
  h.ae_le h'


protected theorem _root_.MeasureTheory.AEDisjoint.of_absolutelyContinuous
    (h : AEDisjoint μ s t) {ν : Measure α} (h' : ν ≪ μ) :
    AEDisjoint ν s t := h' h


protected theorem _root_.MeasureTheory.AEDisjoint.of_le
    (h : AEDisjoint μ s t) {ν : Measure α} (h' : ν ≤ μ) :
    AEDisjoint ν s t :=
  h.of_absolutelyContinuous (Measure.absolutelyContinuous_of_le h')


/-- A map `f : α → β` is said to be *quasi measure preserving* (a.k.a. non-singular) w.r.t. measures
`μa` and `μb` if it is measurable and `μb s = 0` implies `μa (f ⁻¹' s) = 0`. -/
structure QuasiMeasurePreserving {m0 : MeasurableSpace α} (f : α → β)
  (μa : Measure α := by volume_tac)
  (μb : Measure β := by volume_tac) : Prop where
  protected measurable : Measurable f
  protected absolutelyContinuous : μa.map f ≪ μb


protected theorem id {_m0 : MeasurableSpace α} (μ : Measure α) : QuasiMeasurePreserving id μ μ :=
  ⟨measurable_id, map_id.absolutelyContinuous⟩


protected theorem _root_.Measurable.quasiMeasurePreserving
    {_m0 : MeasurableSpace α} (hf : Measurable f) (μ : Measure α) :
    QuasiMeasurePreserving f μ (μ.map f) :=
  ⟨hf, AbsolutelyContinuous.rfl⟩


theorem mono_left (h : QuasiMeasurePreserving f μa μb) (ha : μa' ≪ μa) :
    QuasiMeasurePreserving f μa' μb :=
  ⟨h.1, (ha.map h.1).trans h.2⟩


theorem mono_right (h : QuasiMeasurePreserving f μa μb) (ha : μb ≪ μb') :
    QuasiMeasurePreserving f μa μb' :=
  ⟨h.1, h.2.trans ha⟩


@[mono]
theorem mono (ha : μa' ≪ μa) (hb : μb ≪ μb') (h : QuasiMeasurePreserving f μa μb) :
    QuasiMeasurePreserving f μa' μb' :=
  (h.mono_left ha).mono_right hb


protected theorem comp {g : β → γ} {f : α → β} (hg : QuasiMeasurePreserving g μb μc)
    (hf : QuasiMeasurePreserving f μa μb) : QuasiMeasurePreserving (g ∘ f) μa μc :=
  ⟨hg.measurable.comp hf.measurable, by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      μa : MeasureTheory.Measure α
      μb : MeasureTheory.Measure β
      μc : MeasureTheory.Measure γ
      g : β → γ
      f : α → β
      hg : MeasureTheory.Measure.QuasiMeasurePreserving g μb μc
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μa μb
      ⊢ (MeasureTheory.Measure.map (Function.comp g f) μa).AbsolutelyContinuous μc
    -/
    rw [← map_map hg.1 hf.1]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      μa : MeasureTheory.Measure α
      μb : MeasureTheory.Measure β
      μc : MeasureTheory.Measure γ
      g : β → γ
      f : α → β
      hg : MeasureTheory.Measure.QuasiMeasurePreserving g μb μc
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μa μb
      ⊢ (MeasureTheory.Measure.map g (MeasureTheory.Measure.map f μa)).AbsolutelyCon …
    -/
    exact (hf.2.map hg.1).trans hg.2⟩
    /-
      🎉 no goals
    -/


protected theorem iterate {f : α → α} (hf : QuasiMeasurePreserving f μa μa) :
    ∀ n, QuasiMeasurePreserving f^[n] μa μa
  | 0 => QuasiMeasurePreserving.id μa
  | n + 1 => (hf.iterate n).comp hf


protected theorem aemeasurable (hf : QuasiMeasurePreserving f μa μb) : AEMeasurable f μa :=
  hf.1.aemeasurable


theorem smul_measure {R : Type*} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    (hf : QuasiMeasurePreserving f μa μb) (c : R) : QuasiMeasurePreserving f (c • μa) (c • μb) :=
            /-
              α : Type u_1
              β : Type u_2
              m0 : MeasurableSpace α
              mβ : MeasurableSpace β
              μa : MeasureTheory.Measure α
              μb : MeasureTheory.Measure β
              f : α → β
              R : Type u_8
              inst✝¹ : SMul R ENNReal
              inst✝ : IsScalarTower R ENNReal ENNReal
              hf : MeasureTheory.Measure.QuasiMeasurePreserving f μa μb
              c : R
              ⊢ (MeasureTheory.Measure.map f (HSMul.hSMul c μa)).AbsolutelyContinuous (HSMul …
            -/
  ⟨hf.1, by rw [Measure.map_smul]; exact hf.2.smul c⟩
                                   /-
                                     🎉 no goals
                                   -/


theorem ae_map_le (h : QuasiMeasurePreserving f μa μb) : ae (μa.map f) ≤ ae μb :=
  h.2.ae_le


theorem tendsto_ae (h : QuasiMeasurePreserving f μa μb) : Tendsto f (ae μa) (ae μb) :=
  (tendsto_ae_map h.aemeasurable).mono_right h.ae_map_le


theorem ae (h : QuasiMeasurePreserving f μa μb) {p : β → Prop} (hg : ∀ᵐ x ∂μb, p x) :
    ∀ᵐ x ∂μa, p (f x) :=
  h.tendsto_ae hg


theorem ae_eq (h : QuasiMeasurePreserving f μa μb) {g₁ g₂ : β → δ} (hg : g₁ =ᵐ[μb] g₂) :
    g₁ ∘ f =ᵐ[μa] g₂ ∘ f :=
  h.ae hg


theorem preimage_null (h : QuasiMeasurePreserving f μa μb) {s : Set β} (hs : μb s = 0) :
    μa (f ⁻¹' s) = 0 :=
  preimage_null_of_map_null h.aemeasurable (h.2 hs)


theorem preimage_mono_ae {s t : Set β} (hf : QuasiMeasurePreserving f μa μb) (h : s ≤ᵐ[μb] t) :
    f ⁻¹' s ≤ᵐ[μa] f ⁻¹' t :=
  eventually_map.mp <|
    Eventually.filter_mono (tendsto_ae_map hf.aemeasurable) (Eventually.filter_mono hf.ae_map_le h)


theorem preimage_ae_eq {s t : Set β} (hf : QuasiMeasurePreserving f μa μb) (h : s =ᵐ[μb] t) :
    f ⁻¹' s =ᵐ[μa] f ⁻¹' t :=
  EventuallyLE.antisymm (hf.preimage_mono_ae h.le) (hf.preimage_mono_ae h.symm.le)


/-- The preimage of a null measurable set under a (quasi) measure preserving map is a null
measurable set. -/
theorem _root_.MeasureTheory.NullMeasurableSet.preimage {s : Set β} (hs : NullMeasurableSet s μb)
    (hf : QuasiMeasurePreserving f μa μb) : NullMeasurableSet (f ⁻¹' s) μa :=
  let ⟨t, htm, hst⟩ := hs
  ⟨f ⁻¹' t, hf.measurable htm, hf.preimage_ae_eq hst⟩


theorem preimage_iterate_ae_eq {s : Set α} {f : α → α} (hf : QuasiMeasurePreserving f μ μ) (k : ℕ)
    (hs : f ⁻¹' s =ᵐ[μ] s) : f^[k] ⁻¹' s =ᵐ[μ] s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → α
    hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
    k : Nat
    hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (Nat.iterate f k) s) s
  -/
  induction' k with k ih; · rfl
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → α
    hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
    hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
    k : Nat
    ih : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (Nat.iterate f k) s) s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (Nat.iterate f (HAdd.hAdd k  …
  -/
  rw [iterate_succ, preimage_comp]
  /-
    case succ
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → α
    hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
    hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
    k : Nat
    ih : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (Nat.iterate f k) s) s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage f (Set.preimage (Nat.iterate …
  -/
  exact EventuallyEq.trans (hf.preimage_ae_eq ih) hs
  /-
    🎉 no goals
  -/


theorem image_zpow_ae_eq {s : Set α} {e : α ≃ α} (he : QuasiMeasurePreserving e μ μ)
    (he' : QuasiMeasurePreserving e.symm μ μ) (k : ℤ) (hs : e '' s =ᵐ[μ] s) :
    (⇑(e ^ k)) '' s =ᵐ[μ] s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    e : Equiv α α
    he : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e) μ μ
    he' : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e.symm) μ μ
    k : Int
    hs : (MeasureTheory.ae μ).EventuallyEq (Set.image (⇑e) s) s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.image (⇑(HPow.hPow e k)) s) s
  -/
  rw [Equiv.image_eq_preimage]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    e : Equiv α α
    he : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e) μ μ
    he' : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e.symm) μ μ
    k : Int
    hs : (MeasureTheory.ae μ).EventuallyEq (Set.image (⇑e) s) s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(HPow.hPow e k).symm) s) s
  -/
  obtain ⟨k, rfl | rfl⟩ := k.eq_nat_or_neg
    /-
      case intro.inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      e : Equiv α α
      he : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e) μ μ
      he' : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e.symm) μ μ
      hs : (MeasureTheory.ae μ).EventuallyEq (Set.image (⇑e) s) s
      k : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(HPow.hPow e ↑k).symm) s) s
    -/
  · replace hs : (⇑e⁻¹) ⁻¹' s =ᵐ[μ] s := by rwa [Equiv.image_eq_preimage] at hs
    /-
      case intro.inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      e : Equiv α α
      he : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e) μ μ
      he' : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e.symm) μ μ
      k : Nat
      hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(Inv.inv e)) s) s
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(HPow.hPow e ↑k).symm) s) s
    -/
    replace he' : (⇑e⁻¹)^[k] ⁻¹' s =ᵐ[μ] s := he'.preimage_iterate_ae_eq k hs
    /-
      case intro.inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      e : Equiv α α
      he : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e) μ μ
      k : Nat
      hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(Inv.inv e)) s) s
      he' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (Nat.iterate (⇑(Inv.inv  …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(HPow.hPow e ↑k).symm) s) s
    -/
    rwa [Equiv.Perm.iterate_eq_pow e⁻¹ k, inv_pow e k] at he'
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      e : Equiv α α
      he : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e) μ μ
      he' : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e.symm) μ μ
      hs : (MeasureTheory.ae μ).EventuallyEq (Set.image (⇑e) s) s
      k : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(HPow.hPow e (Neg.neg ↑k)) …
    -/
  · rw [zpow_neg, zpow_natCast]
    replace hs : e ⁻¹' s =ᵐ[μ] s := by
      convert he.preimage_ae_eq hs.symm
      rw [Equiv.preimage_image]
    /-
      case intro.inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      e : Equiv α α
      he : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e) μ μ
      he' : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e.symm) μ μ
      k : Nat
      hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑e) s) s
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(Inv.inv (HPow.hPow e k)). …
    -/
    replace he : (⇑e)^[k] ⁻¹' s =ᵐ[μ] s := he.preimage_iterate_ae_eq k hs
    /-
      case intro.inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      e : Equiv α α
      he' : MeasureTheory.Measure.QuasiMeasurePreserving (⇑e.symm) μ μ
      k : Nat
      hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑e) s) s
      he : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (Nat.iterate (⇑e) k) s) s
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(Inv.inv (HPow.hPow e k)). …
    -/
    rwa [Equiv.Perm.iterate_eq_pow e k] at he
    /-
      🎉 no goals
    -/

-- Need to specify `α := Set α` below because of diamond; see https://github.com/leanprover-community/mathlib4/issues/10941

theorem limsup_preimage_iterate_ae_eq {f : α → α} (hf : QuasiMeasurePreserving f μ μ)
    (hs : f ⁻¹' s =ᵐ[μ] s) : limsup (α := Set α) (fun n => (preimage f)^[n] s) atTop =ᵐ[μ] s :=
  limsup_ae_eq_of_forall_ae_eq (fun n => (preimage f)^[n] s) fun n ↦ by
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → α
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
      hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq ((fun n => Nat.iterate (Set.preimage f) n  …
    -/
    simpa only [Set.preimage_iterate_eq] using hf.preimage_iterate_ae_eq n hs
    /-
      🎉 no goals
    -/

-- Need to specify `α := Set α` below because of diamond; see https://github.com/leanprover-community/mathlib4/issues/10941

theorem liminf_preimage_iterate_ae_eq {f : α → α} (hf : QuasiMeasurePreserving f μ μ)
    (hs : f ⁻¹' s =ᵐ[μ] s) : liminf (α := Set α) (fun n => (preimage f)^[n] s) atTop =ᵐ[μ] s :=
  liminf_ae_eq_of_forall_ae_eq (fun n => (preimage f)^[n] s) fun n ↦ by
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → α
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
      hs : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq ((fun n => Nat.iterate (Set.preimage f) n  …
    -/
    simpa only [Set.preimage_iterate_eq] using hf.preimage_iterate_ae_eq n hs
    /-
      🎉 no goals
    -/


/-- For a quasi measure preserving self-map `f`, if a null measurable set `s` is a.e. invariant,
then it is a.e. equal to a measurable invariant set.
-/
theorem exists_preimage_eq_of_preimage_ae {f : α → α} (h : QuasiMeasurePreserving f μ μ)
    (hs : NullMeasurableSet s μ) (hs' : f ⁻¹' s =ᵐ[μ] s) :
    ∃ t : Set α, MeasurableSet t ∧ t =ᵐ[μ] s ∧ f ⁻¹' t = t := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → α
    h : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
    hs : MeasureTheory.NullMeasurableSet s μ
    hs' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
    ⊢ Exists fun t => And (MeasurableSet t) (And ((MeasureTheory.ae μ).EventuallyE …
  -/
  obtain ⟨t, htm, ht⟩ := hs
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → α
    h : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
    hs' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
    t : Set α
    htm : MeasurableSet t
    ht : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ Exists fun t => And (MeasurableSet t) (And ((MeasureTheory.ae μ).EventuallyE …
  -/
  refine ⟨limsup (f^[·] ⁻¹' t) atTop, ?_, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → α
      h : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
      hs' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
      t : Set α
      htm : MeasurableSet t
      ht : (MeasureTheory.ae μ).EventuallyEq s t
      ⊢ MeasurableSet (Filter.limsup (fun x => Set.preimage (Nat.iterate f x) t) Fil …
    -/
  · exact .measurableSet_limsup fun n ↦ h.measurable.iterate n htm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → α
      h : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
      hs' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
      t : Set α
      htm : MeasurableSet t
      ht : (MeasureTheory.ae μ).EventuallyEq s t
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.limsup (fun x => Set.preimage (Nat …
    -/
  · have : f ⁻¹' t =ᵐ[μ] t := (h.preimage_ae_eq ht.symm).trans (hs'.trans ht)
    /-
      case intro.intro.refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → α
      h : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
      hs' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
      t : Set α
      htm : MeasurableSet t
      ht : (MeasureTheory.ae μ).EventuallyEq s t
      this : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f t) t
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.limsup (fun x => Set.preimage (Nat …
    -/
    exact limsup_ae_eq_of_forall_ae_eq _ fun n ↦ .trans (h.preimage_iterate_ae_eq _ this) ht.symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_3
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → α
      h : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
      hs' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
      t : Set α
      htm : MeasurableSet t
      ht : (MeasureTheory.ae μ).EventuallyEq s t
      ⊢ Eq (Set.preimage f (Filter.limsup (fun x => Set.preimage (Nat.iterate f x) t …
    -/
  · simp only [Set.preimage_iterate_eq]
    /-
      case intro.intro.refine_3
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → α
      h : MeasureTheory.Measure.QuasiMeasurePreserving f μ μ
      hs' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage f s) s
      t : Set α
      htm : MeasurableSet t
      ht : (MeasureTheory.ae μ).EventuallyEq s t
      ⊢ Eq (Set.preimage f (Filter.limsup (fun x => Nat.iterate (Set.preimage f) x t …
    -/
    exact CompleteLatticeHom.apply_limsup_iterate (CompleteLatticeHom.setPreimage f) t
    /-
      🎉 no goals
    -/


@[to_additive]
theorem smul_ae_eq_of_ae_eq {G α : Type*} [Group G] [MulAction G α] {_ : MeasurableSpace α}
    {s t : Set α} {μ : Measure α} (g : G)
    (h_qmp : QuasiMeasurePreserving (g⁻¹ • · : α → α) μ μ)
    (h_ae_eq : s =ᵐ[μ] t) : (g • s : Set α) =ᵐ[μ] (g • t : Set α) := by
  /-
    G : Type u_8
    α : Type u_9
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x✝ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    g : G
    h_qmp : MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMul.hSMul (In …
    h_ae_eq : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul g s) (HSMul.hSMul g t)
  -/
  simpa only [← preimage_smul_inv] using h_qmp.ae_eq h_ae_eq
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pairwise_aedisjoint_of_aedisjoint_forall_ne_one {G α : Type*} [Group G] [MulAction G α]
    {_ : MeasurableSpace α} {μ : Measure α} {s : Set α}
    (h_ae_disjoint : ∀ g ≠ (1 : G), AEDisjoint μ (g • s) s)
    (h_qmp : ∀ g : G, QuasiMeasurePreserving (g • ·) μ μ) :
    Pairwise (AEDisjoint μ on fun g : G => g • s) := by
  /-
    G : Type u_8
    α : Type u_9
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    h_ae_disjoint : ∀ (g : G), Ne g 1 → MeasureTheory.AEDisjoint μ (HSMul.hSMul g  …
    h_qmp : ∀ (g : G), MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMu …
    ⊢ Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) fun g => HSMul.hSMul g …
  -/
  intro g₁ g₂ hg
  /-
    G : Type u_8
    α : Type u_9
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    h_ae_disjoint : ∀ (g : G), Ne g 1 → MeasureTheory.AEDisjoint μ (HSMul.hSMul g  …
    h_qmp : ∀ (g : G), MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMu …
    g₁ g₂ : G
    hg : Ne g₁ g₂
    ⊢ Function.onFun (MeasureTheory.AEDisjoint μ) (fun g => HSMul.hSMul g s) g₁ g₂
  -/
  let g := g₂⁻¹ * g₁
  replace hg : g ≠ 1 := by
    rw [Ne, inv_mul_eq_one]
    exact hg.symm
  have : (g₂⁻¹ • ·) ⁻¹' (g • s ∩ s) = g₁ • s ∩ g₂ • s := by
    rw [preimage_eq_iff_eq_image (MulAction.bijective g₂⁻¹), image_smul, smul_set_inter, smul_smul,
      smul_smul, inv_mul_cancel, one_smul]
  /-
    G : Type u_8
    α : Type u_9
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    h_ae_disjoint : ∀ (g : G), Ne g 1 → MeasureTheory.AEDisjoint μ (HSMul.hSMul g  …
    h_qmp : ∀ (g : G), MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMu …
    g₁ g₂ : G
    g : G := HMul.hMul (Inv.inv g₂) g₁
    hg : Ne g 1
    this : Eq (Set.preimage (fun x => HSMul.hSMul (Inv.inv g₂) x) (Inter.inter (HS …
    ⊢ Function.onFun (MeasureTheory.AEDisjoint μ) (fun g => HSMul.hSMul g s) g₁ g₂
  -/
  change μ (g₁ • s ∩ g₂ • s) = 0
  /-
    G : Type u_8
    α : Type u_9
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    h_ae_disjoint : ∀ (g : G), Ne g 1 → MeasureTheory.AEDisjoint μ (HSMul.hSMul g  …
    h_qmp : ∀ (g : G), MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMu …
    g₁ g₂ : G
    g : G := HMul.hMul (Inv.inv g₂) g₁
    hg : Ne g 1
    this : Eq (Set.preimage (fun x => HSMul.hSMul (Inv.inv g₂) x) (Inter.inter (HS …
    ⊢ Eq (μ (Inter.inter (HSMul.hSMul g₁ s) (HSMul.hSMul g₂ s))) 0
  -/
  exact this ▸ (h_qmp g₂⁻¹).preimage_null (h_ae_disjoint g hg)
  /-
    🎉 no goals
  -/


/-- The filter of sets `s` such that `sᶜ` has finite measure. -/
def cofinite {m0 : MeasurableSpace α} (μ : Measure α) : Filter α :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       ι : Type u_5
                       R : Type u_6
                       R' : Type u_7
                       m0✝ : MeasurableSpace α
                       mβ : MeasurableSpace β
                       inst✝ : MeasurableSpace γ
                       μ✝ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
                       s s' t : Set α
                       f : α → β
                       m0 : MeasurableSpace α
                       μ : MeasureTheory.Measure α
                       ⊢ (fun x => LT.lt (μ x) Top.top) EmptyCollection.emptyCollection
                     -/
  comk (μ · < ∞) (by simp) (fun _ ht _ hs ↦ (measure_mono hs).trans_lt ht) fun s hs t ht ↦
                     /-
                       🎉 no goals
                     -/
    (measure_union_le s t).trans_lt <| ENNReal.add_lt_top.2 ⟨hs, ht⟩


theorem mem_cofinite : s ∈ μ.cofinite ↔ μ sᶜ < ∞ :=
  Iff.rfl


                                                             /-
                                                               α : Type u_1
                                                               m0 : MeasurableSpace α
                                                               μ : MeasureTheory.Measure α
                                                               s : Set α
                                                               ⊢ Iff (Membership.mem μ.cofinite (HasCompl.compl s)) (LT.lt (μ s) Top.top)
                                                             -/
theorem compl_mem_cofinite : sᶜ ∈ μ.cofinite ↔ μ s < ∞ := by rw [mem_cofinite, compl_compl]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem eventually_cofinite {p : α → Prop} : (∀ᶠ x in μ.cofinite, p x) ↔ μ { x | ¬p x } < ∞ :=
  Iff.rfl


instance cofinite.instIsMeasurablyGenerated : IsMeasurablyGenerated μ.cofinite where
  exists_measurable_subset s hs := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Type u_5
      R : Type u_6
      R' : Type u_7
      m0 : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
      s✝ s' t : Set α
      f : α → β
      s : Set α
      hs : Membership.mem μ.cofinite s
      ⊢ Exists fun t => And (Membership.mem μ.cofinite t) (And (MeasurableSet t) (Ha …
    -/
    refine ⟨(toMeasurable μ sᶜ)ᶜ, ?_, (measurableSet_toMeasurable _ _).compl, ?_⟩
      /-
        case refine_1
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ι : Type u_5
        R : Type u_6
        R' : Type u_7
        m0 : MeasurableSpace α
        mβ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
        s✝ s' t : Set α
        f : α → β
        s : Set α
        hs : Membership.mem μ.cofinite s
        ⊢ Membership.mem μ.cofinite (HasCompl.compl (MeasureTheory.toMeasurable μ (Has …
      -/
    · rwa [compl_mem_cofinite, measure_toMeasurable]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ι : Type u_5
        R : Type u_6
        R' : Type u_7
        m0 : MeasurableSpace α
        mβ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
        s✝ s' t : Set α
        f : α → β
        s : Set α
        hs : Membership.mem μ.cofinite s
        ⊢ HasSubset.Subset (HasCompl.compl (MeasureTheory.toMeasurable μ (HasCompl.com …
      -/
    · rw [compl_subset_comm]
      /-
        case refine_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ι : Type u_5
        R : Type u_6
        R' : Type u_7
        m0 : MeasurableSpace α
        mβ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
        s✝ s' t : Set α
        f : α → β
        s : Set α
        hs : Membership.mem μ.cofinite s
        ⊢ HasSubset.Subset (HasCompl.compl s) (MeasureTheory.toMeasurable μ (HasCompl. …
      -/
      apply subset_toMeasurable
      /-
        🎉 no goals
      -/


protected theorem _root_.AEMeasurable.nullMeasurable {f : α → β} (h : AEMeasurable f μ) :
    NullMeasurable f μ :=
  let ⟨_g, hgm, hg⟩ := h; hgm.nullMeasurable.congr hg.symm


lemma _root_.AEMeasurable.nullMeasurableSet_preimage {f : α → β} {s : Set β}
    (hf : AEMeasurable f μ) (hs : MeasurableSet s) : NullMeasurableSet (f ⁻¹' s) μ :=
  hf.nullMeasurable hs


theorem NullMeasurableSet.mono_ac (h : NullMeasurableSet s μ) (hle : ν ≪ μ) :
    NullMeasurableSet s ν :=
  h.preimage <| (QuasiMeasurePreserving.id μ).mono_left hle


theorem NullMeasurableSet.mono (h : NullMeasurableSet s μ) (hle : ν ≤ μ) : NullMeasurableSet s ν :=
  h.mono_ac hle.absolutelyContinuous


theorem AEDisjoint.preimage {ν : Measure β} {f : α → β} {s t : Set β} (ht : AEDisjoint ν s t)
    (hf : QuasiMeasurePreserving f μ ν) : AEDisjoint μ (f ⁻¹' s) (f ⁻¹' t) :=
  hf.preimage_null ht


@[simp]
theorem ae_eq_bot : ae μ = ⊥ ↔ μ = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Iff (Eq (MeasureTheory.ae μ) Bot.bot) (Eq μ 0)
  -/
  rw [← empty_mem_iff_bot, mem_ae_iff, compl_empty, measure_univ_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem ae_neBot : (ae μ).NeBot ↔ μ ≠ 0 :=
  neBot_iff.trans (not_congr ae_eq_bot)


instance Measure.ae.neBot [NeZero μ] : (ae μ).NeBot := ae_neBot.2 <| NeZero.ne μ


@[simp]
theorem ae_zero {_m0 : MeasurableSpace α} : ae (0 : Measure α) = ⊥ :=
  ae_eq_bot.2 rfl


@[mono]
theorem ae_mono (h : μ ≤ ν) : ae μ ≤ ae ν :=
  h.absolutelyContinuous.ae_le


theorem mem_ae_map_iff {f : α → β} (hf : AEMeasurable f μ) {s : Set β} (hs : MeasurableSet s) :
    s ∈ ae (μ.map f) ↔ f ⁻¹' s ∈ ae μ := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : AEMeasurable f μ
    s : Set β
    hs : MeasurableSet s
    ⊢ Iff (Membership.mem (MeasureTheory.ae (MeasureTheory.Measure.map f μ)) s) (M …
  -/
  simp only [mem_ae_iff, map_apply_of_aemeasurable hf hs.compl, preimage_compl]
  /-
    🎉 no goals
  -/


theorem mem_ae_of_mem_ae_map {f : α → β} (hf : AEMeasurable f μ) {s : Set β}
    (hs : s ∈ ae (μ.map f)) : f ⁻¹' s ∈ ae μ :=
  (tendsto_ae_map hf).eventually hs


theorem ae_map_iff {f : α → β} (hf : AEMeasurable f μ) {p : β → Prop}
    (hp : MeasurableSet { x | p x }) : (∀ᵐ y ∂μ.map f, p y) ↔ ∀ᵐ x ∂μ, p (f x) :=
  mem_ae_map_iff hf hp


theorem ae_of_ae_map {f : α → β} (hf : AEMeasurable f μ) {p : β → Prop} (h : ∀ᵐ y ∂μ.map f, p y) :
    ∀ᵐ x ∂μ, p (f x) :=
  mem_ae_of_mem_ae_map hf h


theorem ae_map_mem_range {m0 : MeasurableSpace α} (f : α → β) (hf : MeasurableSet (range f))
    (μ : Measure α) : ∀ᵐ x ∂μ.map f, x ∈ range f := by
  /-
    α : Type u_1
    β : Type u_2
    mβ : MeasurableSpace β
    m0 : MeasurableSpace α
    f : α → β
    hf : MeasurableSet (Set.range f)
    μ : MeasureTheory.Measure α
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.range f) x) (MeasureTheory.a …
  -/
  by_cases h : AEMeasurable f μ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mβ : MeasurableSpace β
      m0 : MeasurableSpace α
      f : α → β
      hf : MeasurableSet (Set.range f)
      μ : MeasureTheory.Measure α
      h : AEMeasurable f μ
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.range f) x) (MeasureTheory.a …
    -/
  · change range f ∈ ae (μ.map f)
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mβ : MeasurableSpace β
      m0 : MeasurableSpace α
      f : α → β
      hf : MeasurableSet (Set.range f)
      μ : MeasureTheory.Measure α
      h : AEMeasurable f μ
      ⊢ Membership.mem (MeasureTheory.ae (MeasureTheory.Measure.map f μ)) (Set.range …
    -/
    rw [mem_ae_map_iff h hf]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mβ : MeasurableSpace β
      m0 : MeasurableSpace α
      f : α → β
      hf : MeasurableSet (Set.range f)
      μ : MeasureTheory.Measure α
      h : AEMeasurable f μ
      ⊢ Membership.mem (MeasureTheory.ae μ) (Set.preimage f (Set.range f))
    -/
    filter_upwards using mem_range_self
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mβ : MeasurableSpace β
      m0 : MeasurableSpace α
      f : α → β
      hf : MeasurableSet (Set.range f)
      μ : MeasureTheory.Measure α
      h : Not (AEMeasurable f μ)
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.range f) x) (MeasureTheory.a …
    -/
  · simp [map_of_not_aemeasurable h]
    /-
      🎉 no goals
    -/



theorem biSup_measure_Iic [Preorder α] {s : Set α} (hsc : s.Countable)
    (hst : ∀ x : α, ∃ y ∈ s, x ≤ y) (hdir : DirectedOn (· ≤ ·) s) :
    ⨆ x ∈ s, μ (Iic x) = μ univ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Preorder α
    s : Set α
    hsc : s.Countable
    hst : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LE.le x y)
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    ⊢ Eq (iSup fun x => iSup fun h => μ (Set.Iic x)) (μ Set.univ)
  -/
  rw [← measure_biUnion_eq_iSup hsc]
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Preorder α
      s : Set α
      hsc : s.Countable
      hst : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LE.le x y)
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
      ⊢ Eq (μ (Set.iUnion fun i => Set.iUnion fun h => Set.Iic i)) (μ Set.univ)
    -/
  · congr
    /-
      case h.e_6.h
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Preorder α
      s : Set α
      hsc : s.Countable
      hst : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LE.le x y)
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
      ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Iic i) Set.univ
    -/
    simp only [← bex_def] at hst
    /-
      case h.e_6.h
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Preorder α
      s : Set α
      hsc : s.Countable
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
      hst : ∀ (x : α), Exists fun x_1 => Exists fun x_2 => LE.le x x_1
      ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Iic i) Set.univ
    -/
    exact iUnion₂_eq_univ_iff.2 hst
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Preorder α
      s : Set α
      hsc : s.Countable
      hst : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LE.le x y)
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
      ⊢ DirectedOn (Function.onFun (fun x1 x2 => HasSubset.Subset x1 x2) Set.Iic) s
    -/
  · exact directedOn_iff_directed.2 (hdir.directed_val.mono_comp _ fun x y => Iic_subset_Iic.2)
    /-
      🎉 no goals
    -/


theorem tendsto_measure_Ico_atTop [Preorder α] [NoMaxOrder α]
    [(atTop : Filter α).IsCountablyGenerated] (μ : Measure α) (a : α) :
    Tendsto (fun x => μ (Ico a x)) atTop (𝓝 (μ (Ici a))) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝² : Preorder α
    inst✝¹ : NoMaxOrder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    μ : MeasureTheory.Measure α
    a : α
    ⊢ Filter.Tendsto (fun x => μ (Set.Ico a x)) Filter.atTop (nhds (μ (Set.Ici a)))
  -/
  rw [← iUnion_Ico_right]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝² : Preorder α
    inst✝¹ : NoMaxOrder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    μ : MeasureTheory.Measure α
    a : α
    ⊢ Filter.Tendsto (fun x => μ (Set.Ico a x)) Filter.atTop (nhds (μ (Set.iUnion  …
  -/
  exact tendsto_measure_iUnion_atTop (antitone_const.Ico monotone_id)
  /-
    🎉 no goals
  -/


theorem tendsto_measure_Ioc_atBot [Preorder α] [NoMinOrder α]
    [(atBot : Filter α).IsCountablyGenerated] (μ : Measure α) (a : α) :
    Tendsto (fun x => μ (Ioc x a)) atBot (𝓝 (μ (Iic a))) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝² : Preorder α
    inst✝¹ : NoMinOrder α
    inst✝ : Filter.atBot.IsCountablyGenerated
    μ : MeasureTheory.Measure α
    a : α
    ⊢ Filter.Tendsto (fun x => μ (Set.Ioc x a)) Filter.atBot (nhds (μ (Set.Iic a)))
  -/
  rw [← iUnion_Ioc_left]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝² : Preorder α
    inst✝¹ : NoMinOrder α
    inst✝ : Filter.atBot.IsCountablyGenerated
    μ : MeasureTheory.Measure α
    a : α
    ⊢ Filter.Tendsto (fun x => μ (Set.Ioc x a)) Filter.atBot (nhds (μ (Set.iUnion  …
  -/
  exact tendsto_measure_iUnion_atBot (monotone_id.Ioc antitone_const)
  /-
    🎉 no goals
  -/


theorem tendsto_measure_Iic_atTop [Preorder α] [(atTop : Filter α).IsCountablyGenerated]
    (μ : Measure α) : Tendsto (fun x => μ (Iic x)) atTop (𝓝 (μ univ)) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝¹ : Preorder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    μ : MeasureTheory.Measure α
    ⊢ Filter.Tendsto (fun x => μ (Set.Iic x)) Filter.atTop (nhds (μ Set.univ))
  -/
  rw [← iUnion_Iic]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝¹ : Preorder α
    inst✝ : Filter.atTop.IsCountablyGenerated
    μ : MeasureTheory.Measure α
    ⊢ Filter.Tendsto (fun x => μ (Set.Iic x)) Filter.atTop (nhds (μ (Set.iUnion fu …
  -/
  exact tendsto_measure_iUnion_atTop monotone_Iic
  /-
    🎉 no goals
  -/


theorem tendsto_measure_Ici_atBot [Preorder α] [(atBot : Filter α).IsCountablyGenerated]
    (μ : Measure α) : Tendsto (fun x => μ (Ici x)) atBot (𝓝 (μ univ)) :=
  tendsto_measure_Iic_atTop (α := αᵒᵈ) μ


theorem Iio_ae_eq_Iic' (ha : μ {a} = 0) : Iio a =ᵐ[μ] Iic a := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : PartialOrder α
    a : α
    ha : Eq (μ (Singleton.singleton a)) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.Iio a) (Set.Iic a)
  -/
  rw [← Iic_diff_right, diff_ae_eq_self, measure_mono_null Set.inter_subset_right ha]
  /-
    🎉 no goals
  -/


theorem Ioi_ae_eq_Ici' (ha : μ {a} = 0) : Ioi a =ᵐ[μ] Ici a :=
  Iio_ae_eq_Iic' (α := αᵒᵈ) ha


theorem Ioo_ae_eq_Ioc' (hb : μ {b} = 0) : Ioo a b =ᵐ[μ] Ioc a b :=
  (ae_eq_refl _).inter (Iio_ae_eq_Iic' hb)


theorem Ioc_ae_eq_Icc' (ha : μ {a} = 0) : Ioc a b =ᵐ[μ] Icc a b :=
  (Ioi_ae_eq_Ici' ha).inter (ae_eq_refl _)


theorem Ioo_ae_eq_Ico' (ha : μ {a} = 0) : Ioo a b =ᵐ[μ] Ico a b :=
  (Ioi_ae_eq_Ici' ha).inter (ae_eq_refl _)


theorem Ioo_ae_eq_Icc' (ha : μ {a} = 0) (hb : μ {b} = 0) : Ioo a b =ᵐ[μ] Icc a b :=
  (Ioi_ae_eq_Ici' ha).inter (Iio_ae_eq_Iic' hb)


theorem Ico_ae_eq_Icc' (hb : μ {b} = 0) : Ico a b =ᵐ[μ] Icc a b :=
  (ae_eq_refl _).inter (Iio_ae_eq_Iic' hb)


theorem Ico_ae_eq_Ioc' (ha : μ {a} = 0) (hb : μ {b} = 0) : Ico a b =ᵐ[μ] Ioc a b :=
  (Ioo_ae_eq_Ico' ha).symm.trans (Ioo_ae_eq_Ioc' hb)


nonrec theorem map_apply (hf : MeasurableEmbedding f) (μ : Measure α) (s : Set β) :
    μ.map f s = μ (f ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ : MeasureTheory.Measure α
    s : Set β
    ⊢ Eq ((MeasureTheory.Measure.map f μ) s) (μ (Set.preimage f s))
  -/
  refine le_antisymm ?_ (le_map_apply hf.measurable.aemeasurable s)
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ : MeasureTheory.Measure α
    s : Set β
    ⊢ LE.le ((MeasureTheory.Measure.map f μ) s) (μ (Set.preimage f s))
  -/
  set t := f '' toMeasurable μ (f ⁻¹' s) ∪ (range f)ᶜ
  have htm : MeasurableSet t :=
    (hf.measurableSet_image.2 <| measurableSet_toMeasurable _ _).union
      hf.measurableSet_range.compl
  have hst : s ⊆ t := by
    rw [subset_union_compl_iff_inter_subset, ← image_preimage_eq_inter_range]
    exact image_subset _ (subset_toMeasurable _ _)
  have hft : f ⁻¹' t = toMeasurable μ (f ⁻¹' s) := by
    rw [preimage_union, preimage_compl, preimage_range, compl_univ, union_empty,
      hf.injective.preimage_image]
  calc
    μ.map f s ≤ μ.map f t := by gcongr
    _ = μ (f ⁻¹' s) := by rw [map_apply hf.measurable htm, hft, measure_toMeasurable]


lemma absolutelyContinuous_map (hf : MeasurableEmbedding f) (hμν : μ ≪ ν) :
    μ.map f ≪ ν.map f := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    μ ν : MeasureTheory.Measure α
    hf : MeasurableEmbedding f
    hμν : μ.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.Measure.map f μ).AbsolutelyContinuous (MeasureTheory.Measure. …
  -/
  intro t ht
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    μ ν : MeasureTheory.Measure α
    hf : MeasurableEmbedding f
    hμν : μ.AbsolutelyContinuous ν
    t : Set β
    ht : Eq ((MeasureTheory.Measure.map f ν) t) 0
    ⊢ Eq ((MeasureTheory.Measure.map f μ) t) 0
  -/
  rw [hf.map_apply] at ht ⊢
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    μ ν : MeasureTheory.Measure α
    hf : MeasurableEmbedding f
    hμν : μ.AbsolutelyContinuous ν
    t : Set β
    ht : Eq (ν (Set.preimage f t)) 0
    ⊢ Eq (μ (Set.preimage f t)) 0
  -/
  exact hμν ht
  /-
    🎉 no goals
  -/


/-- If we map a measure along a measurable equivalence, we can compute the measure on all sets
  (not just the measurable ones). -/
protected theorem map_apply (f : α ≃ᵐ β) (s : Set β) : μ.map f s = μ (f ⁻¹' s) :=
  f.measurableEmbedding.map_apply _ _


@[simp]
theorem map_symm_map (e : α ≃ᵐ β) : (μ.map e).map e.symm = μ := by
  /-
    α : Type u_1
    β : Type u_2
    x✝ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    e : MeasurableEquiv α β
    ⊢ Eq (MeasureTheory.Measure.map (⇑e.symm) (MeasureTheory.Measure.map (⇑e) μ)) μ
  -/
  simp [map_map e.symm.measurable e.measurable]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_map_symm (e : α ≃ᵐ β) : (ν.map e.symm).map e = ν := by
  /-
    α : Type u_1
    β : Type u_2
    x✝ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    e : MeasurableEquiv α β
    ⊢ Eq (MeasureTheory.Measure.map (⇑e) (MeasureTheory.Measure.map (⇑e.symm) ν)) ν
  -/
  simp [map_map e.measurable e.symm.measurable]
  /-
    🎉 no goals
  -/


theorem map_measurableEquiv_injective (e : α ≃ᵐ β) : Injective (Measure.map e) := by
  /-
    α : Type u_1
    β : Type u_2
    x✝ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    e : MeasurableEquiv α β
    ⊢ Function.Injective (MeasureTheory.Measure.map ⇑e)
  -/
  intro μ₁ μ₂ hμ
  /-
    α : Type u_1
    β : Type u_2
    x✝ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    e : MeasurableEquiv α β
    μ₁ μ₂ : MeasureTheory.Measure α
    hμ : Eq (MeasureTheory.Measure.map (⇑e) μ₁) (MeasureTheory.Measure.map (⇑e) μ₂)
    ⊢ Eq μ₁ μ₂
  -/
  apply_fun Measure.map e.symm at hμ
  /-
    α : Type u_1
    β : Type u_2
    x✝ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    e : MeasurableEquiv α β
    μ₁ μ₂ : MeasureTheory.Measure α
    hμ : Eq (MeasureTheory.Measure.map (⇑e.symm) (MeasureTheory.Measure.map (⇑e) μ …
    ⊢ Eq μ₁ μ₂
  -/
  simpa [map_symm_map e] using hμ
  /-
    🎉 no goals
  -/


theorem map_apply_eq_iff_map_symm_apply_eq (e : α ≃ᵐ β) : μ.map e = ν ↔ ν.map e.symm = μ := by
  /-
    α : Type u_1
    β : Type u_2
    x✝ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    e : MeasurableEquiv α β
    ⊢ Iff (Eq (MeasureTheory.Measure.map (⇑e) μ) ν) (Eq (MeasureTheory.Measure.map …
  -/
  rw [← (map_measurableEquiv_injective e).eq_iff, map_map_symm, eq_comm]
  /-
    🎉 no goals
  -/


theorem map_ae (f : α ≃ᵐ β) (μ : Measure α) : Filter.map f (ae μ) = ae (map f μ) := by
  /-
    α : Type u_1
    β : Type u_2
    x✝ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : MeasurableEquiv α β
    μ : MeasureTheory.Measure α
    ⊢ Eq (Filter.map (⇑f) (MeasureTheory.ae μ)) (MeasureTheory.ae (MeasureTheory.M …
  -/
  ext s
  /-
    case h
    α : Type u_1
    β : Type u_2
    x✝ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : MeasurableEquiv α β
    μ : MeasureTheory.Measure α
    s : Set β
    ⊢ Iff (Membership.mem (Filter.map (⇑f) (MeasureTheory.ae μ)) s) (Membership.me …
  -/
  simp_rw [mem_map, mem_ae_iff, ← preimage_compl, f.map_apply]
  /-
    🎉 no goals
  -/


theorem quasiMeasurePreserving_symm (μ : Measure α) (e : α ≃ᵐ β) :
    QuasiMeasurePreserving e.symm (map e μ) μ :=
                         /-
                           α : Type u_1
                           β : Type u_2
                           x✝ : MeasurableSpace α
                           inst✝ : MeasurableSpace β
                           μ : MeasureTheory.Measure α
                           e : MeasurableEquiv α β
                           ⊢ (MeasureTheory.Measure.map (⇑e.symm) (MeasureTheory.Measure.map (⇑e) μ)).Abs …
                         -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  ⟨e.symm.measurable, by rw [Measure.map_map, e.symm_comp_self, Measure.map_id] <;> measurability⟩
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


