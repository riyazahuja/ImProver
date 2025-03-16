/-- Restrict a measure `μ` to a set `s` as an `ℝ≥0∞`-linear map. -/
noncomputable def restrictₗ {m0 : MeasurableSpace α} (s : Set α) : Measure α →ₗ[ℝ≥0∞] Measure α :=
  liftLinear (OuterMeasure.restrict s) fun μ s' hs' t => by
    suffices μ (s ∩ t) = μ (s ∩ t ∩ s') + μ ((s ∩ t) \ s') by
      simpa [← Set.inter_assoc, Set.inter_comm _ s, ← inter_diff_assoc]
    /-
      R : Type u_1
      α : Type u_2
      β : Type u_3
      δ : Type u_4
      γ : Type u_5
      ι : Type u_6
      m0✝ : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      μ✝ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
      s✝ s'✝ t✝ : Set α
      m0 : MeasurableSpace α
      s : Set α
      μ : MeasureTheory.Measure α
      s' : Set α
      hs' : MeasurableSet s'
      t : Set α
      ⊢ Eq (μ (Inter.inter s t)) (HAdd.hAdd (μ (Inter.inter (Inter.inter s t) s')) ( …
    -/
    exact le_toOuterMeasure_caratheodory _ _ hs' _
    /-
      🎉 no goals
    -/


/-- Restrict a measure `μ` to a set `s`. -/
noncomputable def restrict {_m0 : MeasurableSpace α} (μ : Measure α) (s : Set α) : Measure α :=
  restrictₗ s μ


@[simp]
theorem restrictₗ_apply {_m0 : MeasurableSpace α} (s : Set α) (μ : Measure α) :
    restrictₗ s μ = μ.restrict s :=
  rfl


/-- This lemma shows that `restrict` and `toOuterMeasure` commute. Note that the LHS has a
restrict on measures and the RHS has a restrict on outer measures. -/
theorem restrict_toOuterMeasure_eq_toOuterMeasure_restrict (h : MeasurableSet s) :
    (μ.restrict s).toOuterMeasure = OuterMeasure.restrict s μ.toOuterMeasure := by
  simp_rw [restrict, restrictₗ, liftLinear, LinearMap.coe_mk, AddHom.coe_mk,
    toMeasure_toOuterMeasure, OuterMeasure.restrict_trim h, μ.trimmed]


theorem restrict_apply₀ (ht : NullMeasurableSet t (μ.restrict s)) : μ.restrict s t = μ (t ∩ s) := by
  rw [← restrictₗ_apply, restrictₗ, liftLinear_apply₀ _ ht, OuterMeasure.restrict_apply,
    coe_toOuterMeasure]


/-- If `t` is a measurable set, then the measure of `t` with respect to the restriction of
  the measure to `s` equals the outer measure of `t ∩ s`. An alternate version requiring that `s`
  be measurable instead of `t` exists as `Measure.restrict_apply'`. -/
@[simp]
theorem restrict_apply (ht : MeasurableSet t) : μ.restrict s t = μ (t ∩ s) :=
  restrict_apply₀ ht.nullMeasurableSet


/-- Restriction of a measure to a subset is monotone both in set and in measure. -/
theorem restrict_mono' {_m0 : MeasurableSpace α} ⦃s s' : Set α⦄ ⦃μ ν : Measure α⦄ (hs : s ≤ᵐ[μ] s')
    (hμν : μ ≤ ν) : μ.restrict s ≤ ν.restrict s' :=
  Measure.le_iff.2 fun t ht => calc
    μ.restrict s t = μ (t ∩ s) := restrict_apply ht
    _ ≤ μ (t ∩ s') := (measure_mono_ae <| hs.mono fun _x hx ⟨hxt, hxs⟩ => ⟨hxt, hx hxs⟩)
    _ ≤ ν (t ∩ s') := le_iff'.1 hμν (t ∩ s')
    _ = ν.restrict s' t := (restrict_apply ht).symm


/-- Restriction of a measure to a subset is monotone both in set and in measure. -/
@[mono, gcongr]
theorem restrict_mono {_m0 : MeasurableSpace α} ⦃s s' : Set α⦄ (hs : s ⊆ s') ⦃μ ν : Measure α⦄
    (hμν : μ ≤ ν) : μ.restrict s ≤ ν.restrict s' :=
  restrict_mono' (ae_of_all _ hs) hμν


@[gcongr]
theorem restrict_mono_measure {_ : MeasurableSpace α} {μ ν : Measure α} (h : μ ≤ ν) (s : Set α) :
    μ.restrict s ≤ ν.restrict s :=
  restrict_mono subset_rfl h


@[gcongr]
theorem restrict_mono_set {_ : MeasurableSpace α} (μ : Measure α) {s t : Set α} (h : s ⊆ t) :
    μ.restrict s ≤ μ.restrict t :=
  restrict_mono h le_rfl


theorem restrict_mono_ae (h : s ≤ᵐ[μ] t) : μ.restrict s ≤ μ.restrict t :=
  restrict_mono' h (le_refl μ)


theorem restrict_congr_set (h : s =ᵐ[μ] t) : μ.restrict s = μ.restrict t :=
  le_antisymm (restrict_mono_ae h.le) (restrict_mono_ae h.symm.le)


/-- If `s` is a measurable set, then the outer measure of `t` with respect to the restriction of
the measure to `s` equals the outer measure of `t ∩ s`. This is an alternate version of
`Measure.restrict_apply`, requiring that `s` is measurable instead of `t`. -/
@[simp]
theorem restrict_apply' (hs : MeasurableSet s) : μ.restrict s t = μ (t ∩ s) := by
  rw [← toOuterMeasure_apply,
    Measure.restrict_toOuterMeasure_eq_toOuterMeasure_restrict hs,
    OuterMeasure.restrict_apply s t _, toOuterMeasure_apply]


theorem restrict_apply₀' (hs : NullMeasurableSet s μ) : μ.restrict s t = μ (t ∩ s) := by
  rw [← restrict_congr_set hs.toMeasurable_ae_eq,
    restrict_apply' (measurableSet_toMeasurable _ _),
    measure_congr ((ae_eq_refl t).inter hs.toMeasurable_ae_eq)]


theorem restrict_le_self : μ.restrict s ≤ μ :=
  Measure.le_iff.2 fun t ht => calc
    μ.restrict s t = μ (t ∩ s) := restrict_apply ht
    _ ≤ μ t := measure_mono inter_subset_left


theorem restrict_eq_self (h : s ⊆ t) : μ.restrict t s = μ s :=
  (le_iff'.1 restrict_le_self s).antisymm <|
    calc
      μ s ≤ μ (toMeasurable (μ.restrict t) s ∩ t) :=
        measure_mono (subset_inter (subset_toMeasurable _ _) h)
      _ = μ.restrict t s := by
        /-
          α : Type u_2
          m0 : MeasurableSpace α
          μ : MeasureTheory.Measure α
          s t : Set α
          h : HasSubset.Subset s t
          ⊢ Eq (μ (Inter.inter (MeasureTheory.toMeasurable (μ.restrict t) s) t)) ((μ.res …
        -/
        rw [← restrict_apply (measurableSet_toMeasurable _ _), measure_toMeasurable]
        /-
          🎉 no goals
        -/


@[simp]
theorem restrict_apply_self (s : Set α) : (μ.restrict s) s = μ s :=
  restrict_eq_self μ Subset.rfl


theorem restrict_apply_univ (s : Set α) : μ.restrict s univ = μ s := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Eq ((μ.restrict s) Set.univ) (μ s)
  -/
  rw [restrict_apply MeasurableSet.univ, Set.univ_inter]
  /-
    🎉 no goals
  -/


theorem le_restrict_apply (s t : Set α) : μ (t ∩ s) ≤ μ.restrict s t :=
  calc
    μ (t ∩ s) = μ.restrict s (t ∩ s) := (restrict_eq_self μ inter_subset_right).symm
    _ ≤ μ.restrict s t := measure_mono inter_subset_left


theorem restrict_apply_le (s t : Set α) : μ.restrict s t ≤ μ t :=
  Measure.le_iff'.1 restrict_le_self _


theorem restrict_apply_superset (h : s ⊆ t) : μ.restrict s t = μ s :=
  ((measure_mono (subset_univ _)).trans_eq <| restrict_apply_univ _).antisymm
    ((restrict_apply_self μ s).symm.trans_le <| measure_mono h)


@[simp]
theorem restrict_add {_m0 : MeasurableSpace α} (μ ν : Measure α) (s : Set α) :
    (μ + ν).restrict s = μ.restrict s + ν.restrict s :=
  (restrictₗ s).map_add μ ν


@[simp]
theorem restrict_zero {_m0 : MeasurableSpace α} (s : Set α) : (0 : Measure α).restrict s = 0 :=
  (restrictₗ s).map_zero


@[simp]
theorem restrict_smul {_m0 : MeasurableSpace α} (c : ℝ≥0∞) (μ : Measure α) (s : Set α) :
    (c • μ).restrict s = c • μ.restrict s :=
  (restrictₗ s).map_smul c μ


theorem restrict_restrict₀ (hs : NullMeasurableSet s (μ.restrict t)) :
    (μ.restrict t).restrict s = μ.restrict (s ∩ t) :=
  ext fun u hu => by
    simp only [Set.inter_assoc, restrict_apply hu,
      restrict_apply₀ (hu.nullMeasurableSet.inter hs)]


@[simp]
theorem restrict_restrict (hs : MeasurableSet s) : (μ.restrict t).restrict s = μ.restrict (s ∩ t) :=
  restrict_restrict₀ hs.nullMeasurableSet


theorem restrict_restrict_of_subset (h : s ⊆ t) : (μ.restrict t).restrict s = μ.restrict s := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h : HasSubset.Subset s t
    ⊢ Eq ((μ.restrict t).restrict s) (μ.restrict s)
  -/
  ext1 u hu
  /-
    case h
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h : HasSubset.Subset s t
    u : Set α
    hu : MeasurableSet u
    ⊢ Eq (((μ.restrict t).restrict s) u) ((μ.restrict s) u)
  -/
  rw [restrict_apply hu, restrict_apply hu, restrict_eq_self]
  /-
    case h.h
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h : HasSubset.Subset s t
    u : Set α
    hu : MeasurableSet u
    ⊢ HasSubset.Subset (Inter.inter u s) t
  -/
  exact inter_subset_right.trans h
  /-
    🎉 no goals
  -/


theorem restrict_restrict₀' (ht : NullMeasurableSet t μ) :
    (μ.restrict t).restrict s = μ.restrict (s ∩ t) :=
                     /-
                       α : Type u_2
                       m0 : MeasurableSpace α
                       μ : MeasureTheory.Measure α
                       s t : Set α
                       ht : MeasureTheory.NullMeasurableSet t μ
                       u : Set α
                       hu : MeasurableSet u
                       ⊢ Eq (((μ.restrict t).restrict s) u) ((μ.restrict (Inter.inter s t)) u)
                     -/
  ext fun u hu => by simp only [restrict_apply hu, restrict_apply₀' ht, inter_assoc]
                     /-
                       🎉 no goals
                     -/


theorem restrict_restrict' (ht : MeasurableSet t) :
    (μ.restrict t).restrict s = μ.restrict (s ∩ t) :=
  restrict_restrict₀' ht.nullMeasurableSet


theorem restrict_comm (hs : MeasurableSet s) :
    (μ.restrict t).restrict s = (μ.restrict s).restrict t := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.restrict t).restrict s) ((μ.restrict s).restrict t)
  -/
  rw [restrict_restrict hs, restrict_restrict' hs, inter_comm]
  /-
    🎉 no goals
  -/


theorem restrict_apply_eq_zero (ht : MeasurableSet t) : μ.restrict s t = 0 ↔ μ (t ∩ s) = 0 := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    ht : MeasurableSet t
    ⊢ Iff (Eq ((μ.restrict s) t) 0) (Eq (μ (Inter.inter t s)) 0)
  -/
  rw [restrict_apply ht]
  /-
    🎉 no goals
  -/


theorem measure_inter_eq_zero_of_restrict (h : μ.restrict s t = 0) : μ (t ∩ s) = 0 :=
  nonpos_iff_eq_zero.1 (h ▸ le_restrict_apply _ _)


theorem restrict_apply_eq_zero' (hs : MeasurableSet s) : μ.restrict s t = 0 ↔ μ (t ∩ s) = 0 := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    ⊢ Iff (Eq ((μ.restrict s) t) 0) (Eq (μ (Inter.inter t s)) 0)
  -/
  rw [restrict_apply' hs]
  /-
    🎉 no goals
  -/


@[simp]
theorem restrict_eq_zero : μ.restrict s = 0 ↔ μ s = 0 := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Iff (Eq (μ.restrict s) 0) (Eq (μ s) 0)
  -/
  rw [← measure_univ_eq_zero, restrict_apply_univ]
  /-
    🎉 no goals
  -/


/-- If `μ s ≠ 0`, then `μ.restrict s ≠ 0`, in terms of `NeZero` instances. -/
instance restrict.neZero [NeZero (μ s)] : NeZero (μ.restrict s) :=
  ⟨mt restrict_eq_zero.mp <| NeZero.ne _⟩


theorem restrict_zero_set {s : Set α} (h : μ s = 0) : μ.restrict s = 0 :=
  restrict_eq_zero.2 h


@[simp]
theorem restrict_empty : μ.restrict ∅ = 0 :=
  restrict_zero_set measure_empty


@[simp]
theorem restrict_univ : μ.restrict univ = μ :=
                     /-
                       α : Type u_2
                       m0 : MeasurableSpace α
                       μ : MeasureTheory.Measure α
                       s : Set α
                       hs : MeasurableSet s
                       ⊢ Eq ((μ.restrict Set.univ) s) (μ s)
                     -/
  ext fun s hs => by simp [hs]
                     /-
                       🎉 no goals
                     -/


theorem restrict_inter_add_diff₀ (s : Set α) (ht : NullMeasurableSet t μ) :
    μ.restrict (s ∩ t) + μ.restrict (s \ t) = μ.restrict s := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t s : Set α
    ht : MeasureTheory.NullMeasurableSet t μ
    ⊢ Eq (HAdd.hAdd (μ.restrict (Inter.inter s t)) (μ.restrict (SDiff.sdiff s t))) …
  -/
  ext1 u hu
  /-
    case h
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t s : Set α
    ht : MeasureTheory.NullMeasurableSet t μ
    u : Set α
    hu : MeasurableSet u
    ⊢ Eq ((HAdd.hAdd (μ.restrict (Inter.inter s t)) (μ.restrict (SDiff.sdiff s t)) …
  -/
  simp only [add_apply, restrict_apply hu, ← inter_assoc, diff_eq]
  /-
    case h
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t s : Set α
    ht : MeasureTheory.NullMeasurableSet t μ
    u : Set α
    hu : MeasurableSet u
    ⊢ Eq (HAdd.hAdd (μ (Inter.inter (Inter.inter u s) t)) (μ (Inter.inter (Inter.i …
  -/
  exact measure_inter_add_diff₀ (u ∩ s) ht
  /-
    🎉 no goals
  -/


theorem restrict_inter_add_diff (s : Set α) (ht : MeasurableSet t) :
    μ.restrict (s ∩ t) + μ.restrict (s \ t) = μ.restrict s :=
  restrict_inter_add_diff₀ s ht.nullMeasurableSet


theorem restrict_union_add_inter₀ (s : Set α) (ht : NullMeasurableSet t μ) :
    μ.restrict (s ∪ t) + μ.restrict (s ∩ t) = μ.restrict s + μ.restrict t := by
  rw [← restrict_inter_add_diff₀ (s ∪ t) ht, union_inter_cancel_right, union_diff_right, ←
    restrict_inter_add_diff₀ s ht, add_comm, ← add_assoc, add_right_comm]


theorem restrict_union_add_inter (s : Set α) (ht : MeasurableSet t) :
    μ.restrict (s ∪ t) + μ.restrict (s ∩ t) = μ.restrict s + μ.restrict t :=
  restrict_union_add_inter₀ s ht.nullMeasurableSet


theorem restrict_union_add_inter' (hs : MeasurableSet s) (t : Set α) :
    μ.restrict (s ∪ t) + μ.restrict (s ∩ t) = μ.restrict s + μ.restrict t := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    t : Set α
    ⊢ Eq (HAdd.hAdd (μ.restrict (Union.union s t)) (μ.restrict (Inter.inter s t))) …
  -/
  simpa only [union_comm, inter_comm, add_comm] using restrict_union_add_inter t hs
  /-
    🎉 no goals
  -/


theorem restrict_union₀ (h : AEDisjoint μ s t) (ht : NullMeasurableSet t μ) :
    μ.restrict (s ∪ t) = μ.restrict s + μ.restrict t := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    ⊢ Eq (μ.restrict (Union.union s t)) (HAdd.hAdd (μ.restrict s) (μ.restrict t))
  -/
  simp [← restrict_union_add_inter₀ s ht, restrict_zero_set h]
  /-
    🎉 no goals
  -/


theorem restrict_union (h : Disjoint s t) (ht : MeasurableSet t) :
    μ.restrict (s ∪ t) = μ.restrict s + μ.restrict t :=
  restrict_union₀ h.aedisjoint ht.nullMeasurableSet


theorem restrict_union' (h : Disjoint s t) (hs : MeasurableSet s) :
    μ.restrict (s ∪ t) = μ.restrict s + μ.restrict t := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    h : Disjoint s t
    hs : MeasurableSet s
    ⊢ Eq (μ.restrict (Union.union s t)) (HAdd.hAdd (μ.restrict s) (μ.restrict t))
  -/
  rw [union_comm, restrict_union h.symm hs, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem restrict_add_restrict_compl (hs : MeasurableSet s) :
    μ.restrict s + μ.restrict sᶜ = μ := by
  rw [← restrict_union (@disjoint_compl_right (Set α) _ _) hs.compl, union_compl_self,
    restrict_univ]


@[simp]
theorem restrict_compl_add_restrict (hs : MeasurableSet s) : μ.restrict sᶜ + μ.restrict s = μ := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (HAdd.hAdd (μ.restrict (HasCompl.compl s)) (μ.restrict s)) μ
  -/
  rw [add_comm, restrict_add_restrict_compl hs]
  /-
    🎉 no goals
  -/


theorem restrict_union_le (s s' : Set α) : μ.restrict (s ∪ s') ≤ μ.restrict s + μ.restrict s' :=
  le_iff.2 fun t ht ↦ by
    /-
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s s' t : Set α
      ht : MeasurableSet t
      ⊢ LE.le ((μ.restrict (Union.union s s')) t) ((HAdd.hAdd (μ.restrict s) (μ.rest …
    -/
    simpa [ht, inter_union_distrib_left] using measure_union_le (t ∩ s) (t ∩ s')
    /-
      🎉 no goals
    -/


theorem restrict_iUnion_apply_ae [Countable ι] {s : ι → Set α} (hd : Pairwise (AEDisjoint μ on s))
    (hm : ∀ i, NullMeasurableSet (s i) μ) {t : Set α} (ht : MeasurableSet t) :
    μ.restrict (⋃ i, s i) t = ∑' i, μ.restrict (s i) t := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
    hm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq ((μ.restrict (Set.iUnion fun i => s i)) t) (tsum fun i => (μ.restrict (s  …
  -/
  simp only [restrict_apply, ht, inter_iUnion]
  exact
    measure_iUnion₀ (hd.mono fun i j h => h.mono inter_subset_right inter_subset_right)
      fun i => ht.nullMeasurableSet.inter (hm i)


theorem restrict_iUnion_apply [Countable ι] {s : ι → Set α} (hd : Pairwise (Disjoint on s))
    (hm : ∀ i, MeasurableSet (s i)) {t : Set α} (ht : MeasurableSet t) :
    μ.restrict (⋃ i, s i) t = ∑' i, μ.restrict (s i) t :=
  restrict_iUnion_apply_ae hd.aedisjoint (fun i => (hm i).nullMeasurableSet) ht


theorem restrict_iUnion_apply_eq_iSup [Countable ι] {s : ι → Set α} (hd : Directed (· ⊆ ·) s)
    {t : Set α} (ht : MeasurableSet t) : μ.restrict (⋃ i, s i) t = ⨆ i, μ.restrict (s i) t := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq ((μ.restrict (Set.iUnion fun i => s i)) t) (iSup fun i => (μ.restrict (s  …
  -/
  simp only [restrict_apply ht, inter_iUnion]
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq (μ (Set.iUnion fun i => Inter.inter t (s i))) (iSup fun i => μ (Inter.int …
  -/
  rw [Directed.measure_iUnion]
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    t : Set α
    ht : MeasurableSet t
    ⊢ Directed (fun x1 x2 => HasSubset.Subset x1 x2) fun i => Inter.inter t (s i)
  -/
  exacts [hd.mono_comp _ fun s₁ s₂ => inter_subset_inter_right _]
  /-
    🎉 no goals
  -/


/-- The restriction of the pushforward measure is the pushforward of the restriction. For a version
assuming only `AEMeasurable`, see `restrict_map_of_aemeasurable`. -/
theorem restrict_map {f : α → β} (hf : Measurable f) {s : Set β} (hs : MeasurableSet s) :
    (μ.map f).restrict s = (μ.restrict <| f ⁻¹' s).map f :=
                     /-
                       α : Type u_2
                       β : Type u_3
                       m0 : MeasurableSpace α
                       inst✝ : MeasurableSpace β
                       μ : MeasureTheory.Measure α
                       f : α → β
                       hf : Measurable f
                       s : Set β
                       hs : MeasurableSet s
                       t : Set β
                       ht : MeasurableSet t
                       ⊢ Eq (((MeasureTheory.Measure.map f μ).restrict s) t) ((MeasureTheory.Measure. …
                     -/
  ext fun t ht => by simp [*, hf ht]
                     /-
                       🎉 no goals
                     -/


theorem restrict_toMeasurable (h : μ s ≠ ∞) : μ.restrict (toMeasurable μ s) = μ.restrict s :=
  ext fun t ht => by
    rw [restrict_apply ht, restrict_apply ht, inter_comm, measure_toMeasurable_inter ht h,
      inter_comm]


theorem restrict_eq_self_of_ae_mem {_m0 : MeasurableSpace α} ⦃s : Set α⦄ ⦃μ : Measure α⦄
    (hs : ∀ᵐ x ∂μ, x ∈ s) : μ.restrict s = μ :=
  calc
    μ.restrict s = μ.restrict univ := restrict_congr_set (eventuallyEq_univ.mpr hs)
    _ = μ := restrict_univ


theorem restrict_congr_meas (hs : MeasurableSet s) :
    μ.restrict s = ν.restrict s ↔ ∀ t ⊆ s, MeasurableSet t → μ t = ν t :=
  ⟨fun H t hts ht => by
    /-
      α : Type u_2
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      H : Eq (μ.restrict s) (ν.restrict s)
      t : Set α
      hts : HasSubset.Subset t s
      ht : MeasurableSet t
      ⊢ Eq (μ t) (ν t)
    -/
    rw [← inter_eq_self_of_subset_left hts, ← restrict_apply ht, H, restrict_apply ht], fun H =>
    /-
      🎉 no goals
    -/
    ext fun t ht => by
      /-
        α : Type u_2
        m0 : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        hs : MeasurableSet s
        H : ∀ (t : Set α), HasSubset.Subset t s → MeasurableSet t → Eq (μ t) (ν t)
        t : Set α
        ht : MeasurableSet t
        ⊢ Eq ((μ.restrict s) t) ((ν.restrict s) t)
      -/
      rw [restrict_apply ht, restrict_apply ht, H _ inter_subset_right (ht.inter hs)]⟩
      /-
        🎉 no goals
      -/


theorem restrict_congr_mono (hs : s ⊆ t) (h : μ.restrict t = ν.restrict t) :
    μ.restrict s = ν.restrict s := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : HasSubset.Subset s t
    h : Eq (μ.restrict t) (ν.restrict t)
    ⊢ Eq (μ.restrict s) (ν.restrict s)
  -/
  rw [← restrict_restrict_of_subset hs, h, restrict_restrict_of_subset hs]
  /-
    🎉 no goals
  -/


/-- If two measures agree on all measurable subsets of `s` and `t`, then they agree on all
measurable subsets of `s ∪ t`. -/
theorem restrict_union_congr :
    μ.restrict (s ∪ t) = ν.restrict (s ∪ t) ↔
      μ.restrict s = ν.restrict s ∧ μ.restrict t = ν.restrict t := by
  refine ⟨fun h ↦ ⟨restrict_congr_mono subset_union_left h,
    restrict_congr_mono subset_union_right h⟩, ?_⟩
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    ⊢ And (Eq (μ.restrict s) (ν.restrict s)) (Eq (μ.restrict t) (ν.restrict t)) →  …
  -/
  rintro ⟨hs, ht⟩
  /-
    case intro
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : Eq (μ.restrict s) (ν.restrict s)
    ht : Eq (μ.restrict t) (ν.restrict t)
    ⊢ Eq (μ.restrict (Union.union s t)) (ν.restrict (Union.union s t))
  -/
  ext1 u hu
  /-
    case intro.h
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : Eq (μ.restrict s) (ν.restrict s)
    ht : Eq (μ.restrict t) (ν.restrict t)
    u : Set α
    hu : MeasurableSet u
    ⊢ Eq ((μ.restrict (Union.union s t)) u) ((ν.restrict (Union.union s t)) u)
  -/
  simp only [restrict_apply hu, inter_union_distrib_left]
  /-
    case intro.h
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : Eq (μ.restrict s) (ν.restrict s)
    ht : Eq (μ.restrict t) (ν.restrict t)
    u : Set α
    hu : MeasurableSet u
    ⊢ Eq (μ (Union.union (Inter.inter u s) (Inter.inter u t))) (ν (Union.union (In …
  -/
  rcases exists_measurable_superset₂ μ ν (u ∩ s) with ⟨US, hsub, hm, hμ, hν⟩
  calc
    μ (u ∩ s ∪ u ∩ t) = μ (US ∪ u ∩ t) :=
      measure_union_congr_of_subset hsub hμ.le Subset.rfl le_rfl
    _ = μ US + μ ((u ∩ t) \ US) := (measure_add_diff hm.nullMeasurableSet _).symm
    _ = restrict μ s u + restrict μ t (u \ US) := by
      simp only [restrict_apply, hu, hu.diff hm, hμ, ← inter_comm t, inter_diff_assoc]
    _ = restrict ν s u + restrict ν t (u \ US) := by rw [hs, ht]
    _ = ν US + ν ((u ∩ t) \ US) := by
      simp only [restrict_apply, hu, hu.diff hm, hν, ← inter_comm t, inter_diff_assoc]
    _ = ν (US ∪ u ∩ t) := measure_add_diff hm.nullMeasurableSet _
    _ = ν (u ∩ s ∪ u ∩ t) := .symm <| measure_union_congr_of_subset hsub hν.le Subset.rfl le_rfl


theorem restrict_finset_biUnion_congr {s : Finset ι} {t : ι → Set α} :
    μ.restrict (⋃ i ∈ s, t i) = ν.restrict (⋃ i ∈ s, t i) ↔
      ∀ i ∈ s, μ.restrict (t i) = ν.restrict (t i) := by
  classical
  induction' s using Finset.induction_on with i s _ hs; · simp
  simp only [forall_eq_or_imp, iUnion_iUnion_eq_or_left, Finset.mem_insert]
  rw [restrict_union_congr, ← hs]


theorem restrict_iUnion_congr [Countable ι] {s : ι → Set α} :
    μ.restrict (⋃ i, s i) = ν.restrict (⋃ i, s i) ↔ ∀ i, μ.restrict (s i) = ν.restrict (s i) := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    ⊢ Iff (Eq (μ.restrict (Set.iUnion fun i => s i)) (ν.restrict (Set.iUnion fun i …
  -/
  refine ⟨fun h i => restrict_congr_mono (subset_iUnion _ _) h, fun h => ?_⟩
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    h : ∀ (i : ι), Eq (μ.restrict (s i)) (ν.restrict (s i))
    ⊢ Eq (μ.restrict (Set.iUnion fun i => s i)) (ν.restrict (Set.iUnion fun i => s …
  -/
  ext1 t ht
  have D : Directed (· ⊆ ·) fun t : Finset ι => ⋃ i ∈ t, s i :=
    Monotone.directed_le fun t₁ t₂ ht => biUnion_subset_biUnion_left ht
  /-
    case h
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    h : ∀ (i : ι), Eq (μ.restrict (s i)) (ν.restrict (s i))
    t : Set α
    ht : MeasurableSet t
    D : Directed (fun x1 x2 => HasSubset.Subset x1 x2) fun t => Set.iUnion fun i = …
    ⊢ Eq ((μ.restrict (Set.iUnion fun i => s i)) t) ((ν.restrict (Set.iUnion fun i …
  -/
  rw [iUnion_eq_iUnion_finset]
  /-
    case h
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    h : ∀ (i : ι), Eq (μ.restrict (s i)) (ν.restrict (s i))
    t : Set α
    ht : MeasurableSet t
    D : Directed (fun x1 x2 => HasSubset.Subset x1 x2) fun t => Set.iUnion fun i = …
    ⊢ Eq ((μ.restrict (Set.iUnion fun t => Set.iUnion fun i => Set.iUnion fun h => …
  -/
  simp only [restrict_iUnion_apply_eq_iSup D ht, restrict_finset_biUnion_congr.2 fun i _ => h i]
  /-
    🎉 no goals
  -/


theorem restrict_biUnion_congr {s : Set ι} {t : ι → Set α} (hc : s.Countable) :
    μ.restrict (⋃ i ∈ s, t i) = ν.restrict (⋃ i ∈ s, t i) ↔
      ∀ i ∈ s, μ.restrict (t i) = ν.restrict (t i) := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set ι
    t : ι → Set α
    hc : s.Countable
    ⊢ Iff (Eq (μ.restrict (Set.iUnion fun i => Set.iUnion fun h => t i)) (ν.restri …
  -/
  haveI := hc.toEncodable
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set ι
    t : ι → Set α
    hc : s.Countable
    this : Encodable ↑s
    ⊢ Iff (Eq (μ.restrict (Set.iUnion fun i => Set.iUnion fun h => t i)) (ν.restri …
  -/
  simp only [biUnion_eq_iUnion, SetCoe.forall', restrict_iUnion_congr]
  /-
    🎉 no goals
  -/


theorem restrict_sUnion_congr {S : Set (Set α)} (hc : S.Countable) :
    μ.restrict (⋃₀ S) = ν.restrict (⋃₀ S) ↔ ∀ s ∈ S, μ.restrict s = ν.restrict s := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    S : Set (Set α)
    hc : S.Countable
    ⊢ Iff (Eq (μ.restrict S.sUnion) (ν.restrict S.sUnion)) (∀ (s : Set α), Members …
  -/
  rw [sUnion_eq_biUnion, restrict_biUnion_congr hc]
  /-
    🎉 no goals
  -/


/-- This lemma shows that `Inf` and `restrict` commute for measures. -/
theorem restrict_sInf_eq_sInf_restrict {m0 : MeasurableSpace α} {m : Set (Measure α)}
    (hm : m.Nonempty) (ht : MeasurableSet t) :
    (sInf m).restrict t = sInf ((fun μ : Measure α => μ.restrict t) '' m) := by
  /-
    α : Type u_2
    t : Set α
    m0 : MeasurableSpace α
    m : Set (MeasureTheory.Measure α)
    hm : m.Nonempty
    ht : MeasurableSet t
    ⊢ Eq ((InfSet.sInf m).restrict t) (InfSet.sInf (Set.image (fun μ => μ.restrict …
  -/
  ext1 s hs
  simp_rw [sInf_apply hs, restrict_apply hs, sInf_apply (MeasurableSet.inter hs ht),
    Set.image_image, restrict_toOuterMeasure_eq_toOuterMeasure_restrict ht, ←
    Set.image_image _ toOuterMeasure, ← OuterMeasure.restrict_sInf_eq_sInf_restrict _ (hm.image _),
    OuterMeasure.restrict_apply]


theorem exists_mem_of_measure_ne_zero_of_ae (hs : μ s ≠ 0) {p : α → Prop}
    (hp : ∀ᵐ x ∂μ.restrict s, p x) : ∃ x, x ∈ s ∧ p x := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Ne (μ s) 0
    p : α → Prop
    hp : Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))
    ⊢ Exists fun x => And (Membership.mem s x) (p x)
  -/
  rw [← μ.restrict_apply_self, ← frequently_ae_mem_iff] at hs
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Filter.Frequently (fun a => Membership.mem s a) (MeasureTheory.ae (μ.rest …
    p : α → Prop
    hp : Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))
    ⊢ Exists fun x => And (Membership.mem s x) (p x)
  -/
  exact (hs.and_eventually hp).exists
  /-
    🎉 no goals
  -/


/-- If a quasi measure preserving map `f` maps a set `s` to a set `t`,
then it is quasi measure preserving with respect to the restrictions of the measures. -/
theorem QuasiMeasurePreserving.restrict {ν : Measure β} {f : α → β}
    (hf : QuasiMeasurePreserving f μ ν) {t : Set β} (hmaps : MapsTo f s t) :
    QuasiMeasurePreserving f (μ.restrict s) (ν.restrict t) where
  measurable := hf.measurable
  absolutelyContinuous := by
    /-
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ν : MeasureTheory.Measure β
      f : α → β
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
      t : Set β
      hmaps : Set.MapsTo f s t
      ⊢ (MeasureTheory.Measure.map f (μ.restrict s)).AbsolutelyContinuous (ν.restric …
    -/
    refine AbsolutelyContinuous.mk fun u hum ↦ ?_
    /-
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ν : MeasureTheory.Measure β
      f : α → β
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
      t : Set β
      hmaps : Set.MapsTo f s t
      u : Set β
      hum : MeasurableSet u
      ⊢ Eq ((ν.restrict t) u) 0 → Eq ((MeasureTheory.Measure.map f (μ.restrict s)) u …
    -/
    suffices ν (u ∩ t) = 0 → μ (f ⁻¹' u ∩ s) = 0 by simpa [hum, hf.measurable, hf.measurable hum]
    /-
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ν : MeasureTheory.Measure β
      f : α → β
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
      t : Set β
      hmaps : Set.MapsTo f s t
      u : Set β
      hum : MeasurableSet u
      ⊢ Eq (ν (Inter.inter u t)) 0 → Eq (μ (Inter.inter (Set.preimage f u) s)) 0
    -/
    refine fun hu ↦ measure_mono_null ?_ (hf.preimage_null hu)
    /-
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ν : MeasureTheory.Measure β
      f : α → β
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
      t : Set β
      hmaps : Set.MapsTo f s t
      u : Set β
      hum : MeasurableSet u
      hu : Eq (ν (Inter.inter u t)) 0
      ⊢ HasSubset.Subset (Inter.inter (Set.preimage f u) s) (Set.preimage f (Inter.i …
    -/
    rw [preimage_inter]
    /-
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ν : MeasureTheory.Measure β
      f : α → β
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
      t : Set β
      hmaps : Set.MapsTo f s t
      u : Set β
      hum : MeasurableSet u
      hu : Eq (ν (Inter.inter u t)) 0
      ⊢ HasSubset.Subset (Inter.inter (Set.preimage f u) s) (Inter.inter (Set.preima …
    -/
    gcongr
    /-
      case H
      α : Type u_2
      β : Type u_3
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ν : MeasureTheory.Measure β
      f : α → β
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
      t : Set β
      hmaps : Set.MapsTo f s t
      u : Set β
      hum : MeasurableSet u
      hu : Eq (ν (Inter.inter u t)) 0
      ⊢ HasSubset.Subset s (Set.preimage f t)
    -/
    assumption
    /-
      🎉 no goals
    -/


/-- Two measures are equal if they have equal restrictions on a spanning collection of sets
  (formulated using `Union`). -/
theorem ext_iff_of_iUnion_eq_univ [Countable ι] {s : ι → Set α} (hs : ⋃ i, s i = univ) :
    μ = ν ↔ ∀ i, μ.restrict (s i) = ν.restrict (s i) := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hs : Eq (Set.iUnion fun i => s i) Set.univ
    ⊢ Iff (Eq μ ν) (∀ (i : ι), Eq (μ.restrict (s i)) (ν.restrict (s i)))
  -/
  rw [← restrict_iUnion_congr, hs, restrict_univ, restrict_univ]
  /-
    🎉 no goals
  -/


alias ⟨_, ext_of_iUnion_eq_univ⟩ := ext_iff_of_iUnion_eq_univ


/-- Two measures are equal if they have equal restrictions on a spanning collection of sets
  (formulated using `biUnion`). -/
theorem ext_iff_of_biUnion_eq_univ {S : Set ι} {s : ι → Set α} (hc : S.Countable)
    (hs : ⋃ i ∈ S, s i = univ) : μ = ν ↔ ∀ i ∈ S, μ.restrict (s i) = ν.restrict (s i) := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    S : Set ι
    s : ι → Set α
    hc : S.Countable
    hs : Eq (Set.iUnion fun i => Set.iUnion fun h => s i) Set.univ
    ⊢ Iff (Eq μ ν) (∀ (i : ι), Membership.mem S i → Eq (μ.restrict (s i)) (ν.restr …
  -/
  rw [← restrict_biUnion_congr hc, hs, restrict_univ, restrict_univ]
  /-
    🎉 no goals
  -/


alias ⟨_, ext_of_biUnion_eq_univ⟩ := ext_iff_of_biUnion_eq_univ


/-- Two measures are equal if they have equal restrictions on a spanning collection of sets
  (formulated using `sUnion`). -/
theorem ext_iff_of_sUnion_eq_univ {S : Set (Set α)} (hc : S.Countable) (hs : ⋃₀ S = univ) :
    μ = ν ↔ ∀ s ∈ S, μ.restrict s = ν.restrict s :=
                                      /-
                                        α : Type u_2
                                        m0 : MeasurableSpace α
                                        μ ν : MeasureTheory.Measure α
                                        S : Set (Set α)
                                        hc : S.Countable
                                        hs : Eq S.sUnion Set.univ
                                        ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => i) Set.univ
                                      -/
  ext_iff_of_biUnion_eq_univ hc <| by rwa [← sUnion_eq_biUnion]
                                      /-
                                        🎉 no goals
                                      -/


alias ⟨_, ext_of_sUnion_eq_univ⟩ := ext_iff_of_sUnion_eq_univ


theorem ext_of_generateFrom_of_cover {S T : Set (Set α)} (h_gen : ‹_› = generateFrom S)
    (hc : T.Countable) (h_inter : IsPiSystem S) (hU : ⋃₀ T = univ) (htop : ∀ t ∈ T, μ t ≠ ∞)
    (ST_eq : ∀ t ∈ T, ∀ s ∈ S, μ (s ∩ t) = ν (s ∩ t)) (T_eq : ∀ t ∈ T, μ t = ν t) : μ = ν := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    S T : Set (Set α)
    h_gen : Eq m0 (MeasurableSpace.generateFrom S)
    hc : T.Countable
    h_inter : IsPiSystem S
    hU : Eq T.sUnion Set.univ
    htop : ∀ (t : Set α), Membership.mem T t → Ne (μ t) Top.top
    ST_eq : ∀ (t : Set α), Membership.mem T t → ∀ (s : Set α), Membership.mem S s  …
    T_eq : ∀ (t : Set α), Membership.mem T t → Eq (μ t) (ν t)
    ⊢ Eq μ ν
  -/
  refine ext_of_sUnion_eq_univ hc hU fun t ht => ?_
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    S T : Set (Set α)
    h_gen : Eq m0 (MeasurableSpace.generateFrom S)
    hc : T.Countable
    h_inter : IsPiSystem S
    hU : Eq T.sUnion Set.univ
    htop : ∀ (t : Set α), Membership.mem T t → Ne (μ t) Top.top
    ST_eq : ∀ (t : Set α), Membership.mem T t → ∀ (s : Set α), Membership.mem S s  …
    T_eq : ∀ (t : Set α), Membership.mem T t → Eq (μ t) (ν t)
    t : Set α
    ht : Membership.mem T t
    ⊢ Eq (μ.restrict t) (ν.restrict t)
  -/
  ext1 u hu
  /-
    case h
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    S T : Set (Set α)
    h_gen : Eq m0 (MeasurableSpace.generateFrom S)
    hc : T.Countable
    h_inter : IsPiSystem S
    hU : Eq T.sUnion Set.univ
    htop : ∀ (t : Set α), Membership.mem T t → Ne (μ t) Top.top
    ST_eq : ∀ (t : Set α), Membership.mem T t → ∀ (s : Set α), Membership.mem S s  …
    T_eq : ∀ (t : Set α), Membership.mem T t → Eq (μ t) (ν t)
    t : Set α
    ht : Membership.mem T t
    u : Set α
    hu : MeasurableSet u
    ⊢ Eq ((μ.restrict t) u) ((ν.restrict t) u)
  -/
  simp only [restrict_apply hu]
  induction u, hu using induction_on_inter h_gen h_inter with
  | empty => simp only [Set.empty_inter, measure_empty]
  | basic u hu => exact ST_eq _ ht _ hu
  | compl u hu ihu =>
    have := T_eq t ht
    rw [Set.inter_comm] at ihu ⊢
    rwa [← measure_inter_add_diff t hu, ← measure_inter_add_diff t hu, ← ihu,
      ENNReal.add_right_inj] at this
    exact ne_top_of_le_ne_top (htop t ht) (measure_mono Set.inter_subset_left)
  | iUnion f hfd hfm ihf =>
    simp only [← restrict_apply (hfm _), ← restrict_apply (MeasurableSet.iUnion hfm)] at ihf ⊢
    simp only [measure_iUnion hfd hfm, ihf]


/-- Two measures are equal if they are equal on the π-system generating the σ-algebra,
  and they are both finite on an increasing spanning sequence of sets in the π-system.
  This lemma is formulated using `sUnion`. -/
theorem ext_of_generateFrom_of_cover_subset {S T : Set (Set α)} (h_gen : ‹_› = generateFrom S)
    (h_inter : IsPiSystem S) (h_sub : T ⊆ S) (hc : T.Countable) (hU : ⋃₀ T = univ)
    (htop : ∀ s ∈ T, μ s ≠ ∞) (h_eq : ∀ s ∈ S, μ s = ν s) : μ = ν := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    S T : Set (Set α)
    h_gen : Eq m0 (MeasurableSpace.generateFrom S)
    h_inter : IsPiSystem S
    h_sub : HasSubset.Subset T S
    hc : T.Countable
    hU : Eq T.sUnion Set.univ
    htop : ∀ (s : Set α), Membership.mem T s → Ne (μ s) Top.top
    h_eq : ∀ (s : Set α), Membership.mem S s → Eq (μ s) (ν s)
    ⊢ Eq μ ν
  -/
  refine ext_of_generateFrom_of_cover h_gen hc h_inter hU htop ?_ fun t ht => h_eq t (h_sub ht)
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    S T : Set (Set α)
    h_gen : Eq m0 (MeasurableSpace.generateFrom S)
    h_inter : IsPiSystem S
    h_sub : HasSubset.Subset T S
    hc : T.Countable
    hU : Eq T.sUnion Set.univ
    htop : ∀ (s : Set α), Membership.mem T s → Ne (μ s) Top.top
    h_eq : ∀ (s : Set α), Membership.mem S s → Eq (μ s) (ν s)
    ⊢ ∀ (t : Set α), Membership.mem T t → ∀ (s : Set α), Membership.mem S s → Eq ( …
  -/
  intro t ht s hs; rcases (s ∩ t).eq_empty_or_nonempty with H | H
    /-
      case inl
      α : Type u_2
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      S T : Set (Set α)
      h_gen : Eq m0 (MeasurableSpace.generateFrom S)
      h_inter : IsPiSystem S
      h_sub : HasSubset.Subset T S
      hc : T.Countable
      hU : Eq T.sUnion Set.univ
      htop : ∀ (s : Set α), Membership.mem T s → Ne (μ s) Top.top
      h_eq : ∀ (s : Set α), Membership.mem S s → Eq (μ s) (ν s)
      t : Set α
      ht : Membership.mem T t
      s : Set α
      hs : Membership.mem S s
      H : Eq (Inter.inter s t) EmptyCollection.emptyCollection
      ⊢ Eq (μ (Inter.inter s t)) (ν (Inter.inter s t))
    -/
  · simp only [H, measure_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      S T : Set (Set α)
      h_gen : Eq m0 (MeasurableSpace.generateFrom S)
      h_inter : IsPiSystem S
      h_sub : HasSubset.Subset T S
      hc : T.Countable
      hU : Eq T.sUnion Set.univ
      htop : ∀ (s : Set α), Membership.mem T s → Ne (μ s) Top.top
      h_eq : ∀ (s : Set α), Membership.mem S s → Eq (μ s) (ν s)
      t : Set α
      ht : Membership.mem T t
      s : Set α
      hs : Membership.mem S s
      H : (Inter.inter s t).Nonempty
      ⊢ Eq (μ (Inter.inter s t)) (ν (Inter.inter s t))
    -/
  · exact h_eq _ (h_inter _ hs _ (h_sub ht) H)
    /-
      🎉 no goals
    -/


/-- Two measures are equal if they are equal on the π-system generating the σ-algebra,
  and they are both finite on an increasing spanning sequence of sets in the π-system.
  This lemma is formulated using `iUnion`.
  `FiniteSpanningSetsIn.ext` is a reformulation of this lemma. -/
theorem ext_of_generateFrom_of_iUnion (C : Set (Set α)) (B : ℕ → Set α) (hA : ‹_› = generateFrom C)
    (hC : IsPiSystem C) (h1B : ⋃ i, B i = univ) (h2B : ∀ i, B i ∈ C) (hμB : ∀ i, μ (B i) ≠ ∞)
    (h_eq : ∀ s ∈ C, μ s = ν s) : μ = ν := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    C : Set (Set α)
    B : Nat → Set α
    hA : Eq m0 (MeasurableSpace.generateFrom C)
    hC : IsPiSystem C
    h1B : Eq (Set.iUnion fun i => B i) Set.univ
    h2B : ∀ (i : Nat), Membership.mem C (B i)
    hμB : ∀ (i : Nat), Ne (μ (B i)) Top.top
    h_eq : ∀ (s : Set α), Membership.mem C s → Eq (μ s) (ν s)
    ⊢ Eq μ ν
  -/
  refine ext_of_generateFrom_of_cover_subset hA hC ?_ (countable_range B) h1B ?_ h_eq
    /-
      case refine_1
      α : Type u_2
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      C : Set (Set α)
      B : Nat → Set α
      hA : Eq m0 (MeasurableSpace.generateFrom C)
      hC : IsPiSystem C
      h1B : Eq (Set.iUnion fun i => B i) Set.univ
      h2B : ∀ (i : Nat), Membership.mem C (B i)
      hμB : ∀ (i : Nat), Ne (μ (B i)) Top.top
      h_eq : ∀ (s : Set α), Membership.mem C s → Eq (μ s) (ν s)
      ⊢ HasSubset.Subset (Set.range B) C
    -/
  · rintro _ ⟨i, rfl⟩
    /-
      case refine_1.intro
      α : Type u_2
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      C : Set (Set α)
      B : Nat → Set α
      hA : Eq m0 (MeasurableSpace.generateFrom C)
      hC : IsPiSystem C
      h1B : Eq (Set.iUnion fun i => B i) Set.univ
      h2B : ∀ (i : Nat), Membership.mem C (B i)
      hμB : ∀ (i : Nat), Ne (μ (B i)) Top.top
      h_eq : ∀ (s : Set α), Membership.mem C s → Eq (μ s) (ν s)
      i : Nat
      ⊢ Membership.mem C (B i)
    -/
    apply h2B
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      C : Set (Set α)
      B : Nat → Set α
      hA : Eq m0 (MeasurableSpace.generateFrom C)
      hC : IsPiSystem C
      h1B : Eq (Set.iUnion fun i => B i) Set.univ
      h2B : ∀ (i : Nat), Membership.mem C (B i)
      hμB : ∀ (i : Nat), Ne (μ (B i)) Top.top
      h_eq : ∀ (s : Set α), Membership.mem C s → Eq (μ s) (ν s)
      ⊢ ∀ (s : Set α), Membership.mem (Set.range B) s → Ne (μ s) Top.top
    -/
  · rintro _ ⟨i, rfl⟩
    /-
      case refine_2.intro
      α : Type u_2
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      C : Set (Set α)
      B : Nat → Set α
      hA : Eq m0 (MeasurableSpace.generateFrom C)
      hC : IsPiSystem C
      h1B : Eq (Set.iUnion fun i => B i) Set.univ
      h2B : ∀ (i : Nat), Membership.mem C (B i)
      hμB : ∀ (i : Nat), Ne (μ (B i)) Top.top
      h_eq : ∀ (s : Set α), Membership.mem C s → Eq (μ s) (ν s)
      i : Nat
      ⊢ Ne (μ (B i)) Top.top
    -/
    apply hμB
    /-
      🎉 no goals
    -/


@[simp]
theorem restrict_sum (μ : ι → Measure α) {s : Set α} (hs : MeasurableSet s) :
    (sum μ).restrict s = sum fun i => (μ i).restrict s :=
                     /-
                       α : Type u_2
                       ι : Type u_6
                       m0 : MeasurableSpace α
                       μ : ι → MeasureTheory.Measure α
                       s : Set α
                       hs : MeasurableSet s
                       t : Set α
                       ht : MeasurableSet t
                       ⊢ Eq (((MeasureTheory.Measure.sum μ).restrict s) t) ((MeasureTheory.Measure.su …
                     -/
  ext fun t ht => by simp only [sum_apply, restrict_apply, ht, ht.inter hs]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem restrict_sum_of_countable [Countable ι] (μ : ι → Measure α) (s : Set α) :
    (sum μ).restrict s = sum fun i => (μ i).restrict s := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure α
    s : Set α
    ⊢ Eq ((MeasureTheory.Measure.sum μ).restrict s) (MeasureTheory.Measure.sum fun …
  -/
  ext t ht
  /-
    case h
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure α
    s t : Set α
    ht : MeasurableSet t
    ⊢ Eq (((MeasureTheory.Measure.sum μ).restrict s) t) ((MeasureTheory.Measure.su …
  -/
  simp_rw [sum_apply _ ht, restrict_apply ht, sum_apply_of_countable]
  /-
    🎉 no goals
  -/


lemma AbsolutelyContinuous.restrict (h : μ ≪ ν) (s : Set α) : μ.restrict s ≪ ν.restrict s := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous ν
    s : Set α
    ⊢ (μ.restrict s).AbsolutelyContinuous (ν.restrict s)
  -/
  refine Measure.AbsolutelyContinuous.mk (fun t ht htν ↦ ?_)
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous ν
    s t : Set α
    ht : MeasurableSet t
    htν : Eq ((ν.restrict s) t) 0
    ⊢ Eq ((μ.restrict s) t) 0
  -/
  rw [restrict_apply ht] at htν ⊢
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous ν
    s t : Set α
    ht : MeasurableSet t
    htν : Eq (ν (Inter.inter t s)) 0
    ⊢ Eq (μ (Inter.inter t s)) 0
  -/
  exact h htν
  /-
    🎉 no goals
  -/


theorem restrict_iUnion_ae [Countable ι] {s : ι → Set α} (hd : Pairwise (AEDisjoint μ on s))
    (hm : ∀ i, NullMeasurableSet (s i) μ) : μ.restrict (⋃ i, s i) = sum fun i => μ.restrict (s i) :=
                     /-
                       α : Type u_2
                       ι : Type u_6
                       m0 : MeasurableSpace α
                       μ : MeasureTheory.Measure α
                       inst✝ : Countable ι
                       s : ι → Set α
                       hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
                       hm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
                       t : Set α
                       ht : MeasurableSet t
                       ⊢ Eq ((μ.restrict (Set.iUnion fun i => s i)) t) ((MeasureTheory.Measure.sum fu …
                     -/
  ext fun t ht => by simp only [sum_apply _ ht, restrict_iUnion_apply_ae hd hm ht]
                     /-
                       🎉 no goals
                     -/


theorem restrict_iUnion [Countable ι] {s : ι → Set α} (hd : Pairwise (Disjoint on s))
    (hm : ∀ i, MeasurableSet (s i)) : μ.restrict (⋃ i, s i) = sum fun i => μ.restrict (s i) :=
  restrict_iUnion_ae hd.aedisjoint fun i => (hm i).nullMeasurableSet


theorem restrict_iUnion_le [Countable ι] {s : ι → Set α} :
    μ.restrict (⋃ i, s i) ≤ sum fun i => μ.restrict (s i) :=
                         /-
                           α : Type u_2
                           ι : Type u_6
                           m0 : MeasurableSpace α
                           μ : MeasureTheory.Measure α
                           inst✝ : Countable ι
                           s : ι → Set α
                           t : Set α
                           ht : MeasurableSet t
                           ⊢ LE.le ((μ.restrict (Set.iUnion fun i => s i)) t) ((MeasureTheory.Measure.sum …
                         -/
  le_iff.2 fun t ht ↦ by simpa [ht, inter_iUnion] using measure_iUnion_le (t ∩ s ·)
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem ae_restrict_iUnion_eq [Countable ι] (s : ι → Set α) :
    ae (μ.restrict (⋃ i, s i)) = ⨆ i, ae (μ.restrict (s i)) :=
  le_antisymm ((ae_sum_eq fun i => μ.restrict (s i)) ▸ ae_mono restrict_iUnion_le) <|
    iSup_le fun i => ae_mono <| restrict_mono (subset_iUnion s i) le_rfl


@[simp]
theorem ae_restrict_union_eq (s t : Set α) :
    ae (μ.restrict (s ∪ t)) = ae (μ.restrict s) ⊔ ae (μ.restrict t) := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    ⊢ Eq (MeasureTheory.ae (μ.restrict (Union.union s t))) (Max.max (MeasureTheory …
  -/
  simp [union_eq_iUnion, iSup_bool_eq]
  /-
    🎉 no goals
  -/


theorem ae_restrict_biUnion_eq (s : ι → Set α) {t : Set ι} (ht : t.Countable) :
    ae (μ.restrict (⋃ i ∈ t, s i)) = ⨆ i ∈ t, ae (μ.restrict (s i)) := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    t : Set ι
    ht : t.Countable
    ⊢ Eq (MeasureTheory.ae (μ.restrict (Set.iUnion fun i => Set.iUnion fun h => s  …
  -/
  haveI := ht.to_subtype
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    t : Set ι
    ht : t.Countable
    this : Countable ↑t
    ⊢ Eq (MeasureTheory.ae (μ.restrict (Set.iUnion fun i => Set.iUnion fun h => s  …
  -/
  rw [biUnion_eq_iUnion, ae_restrict_iUnion_eq, ← iSup_subtype'']
  /-
    🎉 no goals
  -/


theorem ae_restrict_biUnion_finset_eq (s : ι → Set α) (t : Finset ι) :
    ae (μ.restrict (⋃ i ∈ t, s i)) = ⨆ i ∈ t, ae (μ.restrict (s i)) :=
  ae_restrict_biUnion_eq s t.countable_toSet


theorem ae_restrict_iUnion_iff [Countable ι] (s : ι → Set α) (p : α → Prop) :
                                                                                /-
                                                                                  α : Type u_2
                                                                                  ι : Type u_6
                                                                                  m0 : MeasurableSpace α
                                                                                  μ : MeasureTheory.Measure α
                                                                                  inst✝ : Countable ι
                                                                                  s : ι → Set α
                                                                                  p : α → Prop
                                                                                  ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Set.iUn …
                                                                                -/
    (∀ᵐ x ∂μ.restrict (⋃ i, s i), p x) ↔ ∀ i, ∀ᵐ x ∂μ.restrict (s i), p x := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem ae_restrict_union_iff (s t : Set α) (p : α → Prop) :
                                                                                                /-
                                                                                                  α : Type u_2
                                                                                                  m0 : MeasurableSpace α
                                                                                                  μ : MeasureTheory.Measure α
                                                                                                  s t : Set α
                                                                                                  p : α → Prop
                                                                                                  ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Union.u …
                                                                                                -/
    (∀ᵐ x ∂μ.restrict (s ∪ t), p x) ↔ (∀ᵐ x ∂μ.restrict s, p x) ∧ ∀ᵐ x ∂μ.restrict t, p x := by simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem ae_restrict_biUnion_iff (s : ι → Set α) {t : Set ι} (ht : t.Countable) (p : α → Prop) :
    (∀ᵐ x ∂μ.restrict (⋃ i ∈ t, s i), p x) ↔ ∀ i ∈ t, ∀ᵐ x ∂μ.restrict (s i), p x := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    t : Set ι
    ht : t.Countable
    p : α → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Set.iUn …
  -/
  simp_rw [Filter.Eventually, ae_restrict_biUnion_eq s ht, mem_iSup]
  /-
    🎉 no goals
  -/


@[simp]
theorem ae_restrict_biUnion_finset_iff (s : ι → Set α) (t : Finset ι) (p : α → Prop) :
    (∀ᵐ x ∂μ.restrict (⋃ i ∈ t, s i), p x) ↔ ∀ i ∈ t, ∀ᵐ x ∂μ.restrict (s i), p x := by
  /-
    α : Type u_2
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    t : Finset ι
    p : α → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Set.iUn …
  -/
  simp_rw [Filter.Eventually, ae_restrict_biUnion_finset_eq s, mem_iSup]
  /-
    🎉 no goals
  -/


theorem ae_eq_restrict_iUnion_iff [Countable ι] (s : ι → Set α) (f g : α → δ) :
    f =ᵐ[μ.restrict (⋃ i, s i)] g ↔ ∀ i, f =ᵐ[μ.restrict (s i)] g := by
  /-
    α : Type u_2
    δ : Type u_4
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    f g : α → δ
    ⊢ Iff ((MeasureTheory.ae (μ.restrict (Set.iUnion fun i => s i))).EventuallyEq  …
  -/
  simp_rw [EventuallyEq, ae_restrict_iUnion_eq, eventually_iSup]
  /-
    🎉 no goals
  -/


theorem ae_eq_restrict_biUnion_iff (s : ι → Set α) {t : Set ι} (ht : t.Countable) (f g : α → δ) :
    f =ᵐ[μ.restrict (⋃ i ∈ t, s i)] g ↔ ∀ i ∈ t, f =ᵐ[μ.restrict (s i)] g := by
  /-
    α : Type u_2
    δ : Type u_4
    ι : Type u_6
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : ι → Set α
    t : Set ι
    ht : t.Countable
    f g : α → δ
    ⊢ Iff ((MeasureTheory.ae (μ.restrict (Set.iUnion fun i => Set.iUnion fun h =>  …
  -/
  simp_rw [ae_restrict_biUnion_eq s ht, EventuallyEq, eventually_iSup]
  /-
    🎉 no goals
  -/


theorem ae_eq_restrict_biUnion_finset_iff (s : ι → Set α) (t : Finset ι) (f g : α → δ) :
    f =ᵐ[μ.restrict (⋃ i ∈ t, s i)] g ↔ ∀ i ∈ t, f =ᵐ[μ.restrict (s i)] g :=
  ae_eq_restrict_biUnion_iff s t.countable_toSet f g


theorem ae_restrict_uIoc_eq [LinearOrder α] (a b : α) :
    ae (μ.restrict (Ι a b)) = ae (μ.restrict (Ioc a b)) ⊔ ae (μ.restrict (Ioc b a)) := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder α
    a b : α
    ⊢ Eq (MeasureTheory.ae (μ.restrict (Set.uIoc a b))) (Max.max (MeasureTheory.ae …
  -/
  simp only [uIoc_eq_union, ae_restrict_union_eq]
  /-
    🎉 no goals
  -/


/-- See also `MeasureTheory.ae_uIoc_iff`. -/
theorem ae_restrict_uIoc_iff [LinearOrder α] {a b : α} {P : α → Prop} :
    (∀ᵐ x ∂μ.restrict (Ι a b), P x) ↔
      (∀ᵐ x ∂μ.restrict (Ioc a b), P x) ∧ ∀ᵐ x ∂μ.restrict (Ioc b a), P x := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder α
    a b : α
    P : α → Prop
    ⊢ Iff (Filter.Eventually (fun x => P x) (MeasureTheory.ae (μ.restrict (Set.uIo …
  -/
  rw [ae_restrict_uIoc_eq, eventually_sup]
  /-
    🎉 no goals
  -/


theorem ae_restrict_iff₀ {p : α → Prop} (hp : NullMeasurableSet { x | p x } (μ.restrict s)) :
    (∀ᵐ x ∂μ.restrict s, p x) ↔ ∀ᵐ x ∂μ, x ∈ s → p x := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    p : α → Prop
    hp : MeasureTheory.NullMeasurableSet (setOf fun x => p x) (μ.restrict s)
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))) (Fi …
  -/
  simp only [ae_iff, ← compl_setOf, Measure.restrict_apply₀ hp.compl]
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    p : α → Prop
    hp : MeasureTheory.NullMeasurableSet (setOf fun x => p x) (μ.restrict s)
    ⊢ Iff (Eq (μ (Inter.inter (HasCompl.compl (setOf fun x => p x)) s)) 0) (Eq (μ  …
  -/
  rw [iff_iff_eq]; congr with x; simp [and_comm]
                                 /-
                                   🎉 no goals
                                 -/


theorem ae_restrict_iff {p : α → Prop} (hp : MeasurableSet { x | p x }) :
    (∀ᵐ x ∂μ.restrict s, p x) ↔ ∀ᵐ x ∂μ, x ∈ s → p x :=
  ae_restrict_iff₀ hp.nullMeasurableSet


theorem ae_imp_of_ae_restrict {s : Set α} {p : α → Prop} (h : ∀ᵐ x ∂μ.restrict s, p x) :
    ∀ᵐ x ∂μ, x ∈ s → p x := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    p : α → Prop
    h : Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))
    ⊢ Filter.Eventually (fun x => Membership.mem s x → p x) (MeasureTheory.ae μ)
  -/
  simp only [ae_iff] at h ⊢
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    p : α → Prop
    h : Eq ((μ.restrict s) (setOf fun a => Not (p a))) 0
    ⊢ Eq (μ (setOf fun a => Not (Membership.mem s a → p a))) 0
  -/
  simpa [setOf_and, inter_comm] using measure_inter_eq_zero_of_restrict h
  /-
    🎉 no goals
  -/


theorem ae_restrict_iff'₀ {p : α → Prop} (hs : NullMeasurableSet s μ) :
    (∀ᵐ x ∂μ.restrict s, p x) ↔ ∀ᵐ x ∂μ, x ∈ s → p x := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    p : α → Prop
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))) (Fi …
  -/
  simp only [ae_iff, ← compl_setOf, restrict_apply₀' hs]
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    p : α → Prop
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ Iff (Eq (μ (Inter.inter (HasCompl.compl (setOf fun x => p x)) s)) 0) (Eq (μ  …
  -/
  rw [iff_iff_eq]; congr with x; simp [and_comm]
                                 /-
                                   🎉 no goals
                                 -/


theorem ae_restrict_iff' {p : α → Prop} (hs : MeasurableSet s) :
    (∀ᵐ x ∂μ.restrict s, p x) ↔ ∀ᵐ x ∂μ, x ∈ s → p x :=
  ae_restrict_iff'₀ hs.nullMeasurableSet


theorem _root_.Filter.EventuallyEq.restrict {f g : α → δ} {s : Set α} (hfg : f =ᵐ[μ] g) :
    f =ᵐ[μ.restrict s] g := by
  -- note that we cannot use `ae_restrict_iff` since we do not require measurability
  /-
    α : Type u_2
    δ : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → δ
    s : Set α
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq f g
  -/
  refine hfg.filter_mono ?_
  /-
    α : Type u_2
    δ : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → δ
    s : Set α
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ LE.le (MeasureTheory.ae (μ.restrict s)) (MeasureTheory.ae μ)
  -/
  rw [Measure.ae_le_iff_absolutelyContinuous]
  /-
    α : Type u_2
    δ : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → δ
    s : Set α
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ (μ.restrict s).AbsolutelyContinuous μ
  -/
  exact Measure.absolutelyContinuous_of_le Measure.restrict_le_self
  /-
    🎉 no goals
  -/


theorem ae_restrict_mem₀ (hs : NullMeasurableSet s μ) : ∀ᵐ x ∂μ.restrict s, x ∈ s :=
  (ae_restrict_iff'₀ hs).2 (Filter.Eventually.of_forall fun _ => id)


theorem ae_restrict_mem (hs : MeasurableSet s) : ∀ᵐ x ∂μ.restrict s, x ∈ s :=
  ae_restrict_mem₀ hs.nullMeasurableSet


theorem ae_restrict_of_forall_mem {μ : Measure α} {s : Set α}
    (hs : MeasurableSet s) {p : α → Prop} (h : ∀ x ∈ s, p x) : ∀ᵐ (x : α) ∂μ.restrict s, p x :=
  (ae_restrict_mem hs).mono h


theorem ae_restrict_of_ae {s : Set α} {p : α → Prop} (h : ∀ᵐ x ∂μ, p x) : ∀ᵐ x ∂μ.restrict s, p x :=
  h.filter_mono (ae_mono Measure.restrict_le_self)


theorem ae_restrict_of_ae_restrict_of_subset {s t : Set α} {p : α → Prop} (hst : s ⊆ t)
    (h : ∀ᵐ x ∂μ.restrict t, p x) : ∀ᵐ x ∂μ.restrict s, p x :=
  h.filter_mono (ae_mono <| Measure.restrict_mono hst (le_refl μ))


theorem ae_of_ae_restrict_of_ae_restrict_compl (t : Set α) {p : α → Prop}
    (ht : ∀ᵐ x ∂μ.restrict t, p x) (htc : ∀ᵐ x ∂μ.restrict tᶜ, p x) : ∀ᵐ x ∂μ, p x :=
  nonpos_iff_eq_zero.1 <|
    calc
      μ { x | ¬p x } ≤ μ ({ x | ¬p x } ∩ t) + μ ({ x | ¬p x } ∩ tᶜ) :=
        measure_le_inter_add_diff _ _ _
      _ ≤ μ.restrict t { x | ¬p x } + μ.restrict tᶜ { x | ¬p x } :=
        add_le_add (le_restrict_apply _ _) (le_restrict_apply _ _)
                  /-
                    α : Type u_2
                    m0 : MeasurableSpace α
                    μ : MeasureTheory.Measure α
                    t : Set α
                    p : α → Prop
                    ht : Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict t))
                    htc : Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (HasCompl …
                    ⊢ Eq (HAdd.hAdd ((μ.restrict t) (setOf fun x => Not (p x))) ((μ.restrict (HasC …
                  -/
      _ = 0 := by rw [ae_iff.1 ht, ae_iff.1 htc, zero_add]
                  /-
                    🎉 no goals
                  -/


theorem mem_map_restrict_ae_iff {β} {s : Set α} {t : Set β} {f : α → β} (hs : MeasurableSet s) :
    t ∈ Filter.map f (ae (μ.restrict s)) ↔ μ ((f ⁻¹' t)ᶜ ∩ s) = 0 := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    s : Set α
    t : Set β
    f : α → β
    hs : MeasurableSet s
    ⊢ Iff (Membership.mem (Filter.map f (MeasureTheory.ae (μ.restrict s))) t) (Eq  …
  -/
  rw [mem_map, mem_ae_iff, Measure.restrict_apply' hs]
  /-
    🎉 no goals
  -/


theorem ae_add_measure_iff {p : α → Prop} {ν} :
    (∀ᵐ x ∂μ + ν, p x) ↔ (∀ᵐ x ∂μ, p x) ∧ ∀ᵐ x ∂ν, p x :=
  add_eq_zero


theorem ae_eq_comp' {ν : Measure β} {f : α → β} {g g' : β → δ} (hf : AEMeasurable f μ)
    (h : g =ᵐ[ν] g') (h2 : μ.map f ≪ ν) : g ∘ f =ᵐ[μ] g' ∘ f :=
  (tendsto_ae_map hf).mono_right h2.ae_le h


theorem Measure.QuasiMeasurePreserving.ae_eq_comp {ν : Measure β} {f : α → β} {g g' : β → δ}
    (hf : QuasiMeasurePreserving f μ ν) (h : g =ᵐ[ν] g') : g ∘ f =ᵐ[μ] g' ∘ f :=
  ae_eq_comp' hf.aemeasurable h hf.absolutelyContinuous


theorem ae_eq_comp {f : α → β} {g g' : β → δ} (hf : AEMeasurable f μ) (h : g =ᵐ[μ.map f] g') :
    g ∘ f =ᵐ[μ] g' ∘ f :=
  ae_eq_comp' hf h AbsolutelyContinuous.rfl


@[to_additive]
theorem div_ae_eq_one {β} [Group β] (f g : α → β) : f / g =ᵐ[μ] 1 ↔ f =ᵐ[μ] g := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : Group β
    f g : α → β
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (HDiv.hDiv f g) 1) ((MeasureTheory.ae …
  -/
  refine ⟨fun h ↦ h.mono fun x hx ↦ ?_, fun h ↦ h.mono fun x hx ↦ ?_⟩
    /-
      case refine_1
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : Group β
      f g : α → β
      h : (MeasureTheory.ae μ).EventuallyEq (HDiv.hDiv f g) 1
      x : α
      hx : Eq (HDiv.hDiv f g x) (1 x)
      ⊢ Eq (f x) (g x)
    -/
  · rwa [Pi.div_apply, Pi.one_apply, div_eq_one] at hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : Group β
      f g : α → β
      h : (MeasureTheory.ae μ).EventuallyEq f g
      x : α
      hx : Eq (f x) (g x)
      ⊢ Eq (HDiv.hDiv f g x) (1 x)
    -/
  · rwa [Pi.div_apply, Pi.one_apply, div_eq_one]
    /-
      🎉 no goals
    -/


@[to_additive sub_nonneg_ae]
lemma one_le_div_ae {β : Type*} [Group β] [LE β] [MulRightMono β] (f g : α → β) :
    1 ≤ᵐ[μ] g / f ↔ f ≤ᵐ[μ] g := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝² : Group β
    inst✝¹ : LE β
    inst✝ : MulRightMono β
    f g : α → β
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyLE 1 (HDiv.hDiv g f)) ((MeasureTheory.ae …
  -/
  refine ⟨fun h ↦ h.mono fun a ha ↦ ?_, fun h ↦ h.mono fun a ha ↦ ?_⟩
    /-
      case refine_1
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝² : Group β
      inst✝¹ : LE β
      inst✝ : MulRightMono β
      f g : α → β
      h : (MeasureTheory.ae μ).EventuallyLE 1 (HDiv.hDiv g f)
      a : α
      ha : LE.le (1 a) (HDiv.hDiv g f a)
      ⊢ LE.le (f a) (g a)
    -/
  · rwa [Pi.one_apply, Pi.div_apply, one_le_div'] at ha
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝² : Group β
      inst✝¹ : LE β
      inst✝ : MulRightMono β
      f g : α → β
      h : (MeasureTheory.ae μ).EventuallyLE f g
      a : α
      ha : LE.le (f a) (g a)
      ⊢ LE.le (1 a) (HDiv.hDiv g f a)
    -/
  · rwa [Pi.one_apply, Pi.div_apply, one_le_div']
    /-
      🎉 no goals
    -/


theorem le_ae_restrict : ae μ ⊓ 𝓟 s ≤ ae (μ.restrict s) := fun _s hs =>
  eventually_inf_principal.2 (ae_imp_of_ae_restrict hs)


@[simp]
theorem ae_restrict_eq (hs : MeasurableSet s) : ae (μ.restrict s) = ae μ ⊓ 𝓟 s := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.ae (μ.restrict s)) (Min.min (MeasureTheory.ae μ) (Filter.p …
  -/
  ext t
  simp only [mem_inf_principal, mem_ae_iff, restrict_apply_eq_zero' hs, compl_setOf,
    Classical.not_imp, fun a => and_comm (a := a ∈ s) (b := ¬a ∈ t)]
  /-
    case h
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    t : Set α
    ⊢ Iff (Eq (μ (Inter.inter (HasCompl.compl t) s)) 0) (Eq (μ (setOf fun a => And …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma ae_restrict_le (hs : MeasurableSet s) : ae (μ.restrict s) ≤ ae μ :=
  ae_restrict_eq hs ▸ inf_le_left


theorem ae_restrict_eq_bot {s} : ae (μ.restrict s) = ⊥ ↔ μ s = 0 :=
  ae_eq_bot.trans restrict_eq_zero


theorem ae_restrict_neBot {s} : (ae <| μ.restrict s).NeBot ↔ μ s ≠ 0 :=
  neBot_iff.trans ae_restrict_eq_bot.not


theorem self_mem_ae_restrict {s} (hs : MeasurableSet s) : s ∈ ae (μ.restrict s) := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Membership.mem (MeasureTheory.ae (μ.restrict s)) s
  -/
  simp only [ae_restrict_eq hs, exists_prop, mem_principal, mem_inf_iff]
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Exists fun t₁ => And (Membership.mem (MeasureTheory.ae μ) t₁) (Exists fun t₂ …
  -/
  exact ⟨_, univ_mem, s, Subset.rfl, (univ_inter s).symm⟩
  /-
    🎉 no goals
  -/


/-- If two measurable sets are ae_eq then any proposition that is almost everywhere true on one
is almost everywhere true on the other -/
theorem ae_restrict_of_ae_eq_of_ae_restrict {s t} (hst : s =ᵐ[μ] t) {p : α → Prop} :
                                                              /-
                                                                α : Type u_2
                                                                m0 : MeasurableSpace α
                                                                μ : MeasureTheory.Measure α
                                                                s t : α → Prop
                                                                hst : (MeasureTheory.ae μ).EventuallyEq s t
                                                                p : α → Prop
                                                                ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s)) → Filter. …
                                                              -/
    (∀ᵐ x ∂μ.restrict s, p x) → ∀ᵐ x ∂μ.restrict t, p x := by simp [Measure.restrict_congr_set hst]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- If two measurable sets are ae_eq then any proposition that is almost everywhere true on one
is almost everywhere true on the other -/
theorem ae_restrict_congr_set {s t} (hst : s =ᵐ[μ] t) {p : α → Prop} :
    (∀ᵐ x ∂μ.restrict s, p x) ↔ ∀ᵐ x ∂μ.restrict t, p x :=
  ⟨ae_restrict_of_ae_eq_of_ae_restrict hst, ae_restrict_of_ae_eq_of_ae_restrict hst.symm⟩


lemma NullMeasurable.measure_preimage_eq_measure_restrict_preimage_of_ae_compl_eq_const
    {β : Type*} [MeasurableSpace β] {b : β} {f : α → β} {s : Set α}
    (f_mble : NullMeasurable f (μ.restrict s)) (hs : f =ᵐ[Measure.restrict μ sᶜ] (fun _ ↦ b))
    {t : Set β} (t_mble : MeasurableSet t) (ht : b ∉ t) :
    μ (f ⁻¹' t) = μ.restrict s (f ⁻¹' t) := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    b : β
    f : α → β
    s : Set α
    f_mble : MeasureTheory.NullMeasurable f (μ.restrict s)
    hs : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq f fun x = …
    t : Set β
    t_mble : MeasurableSet t
    ht : Not (Membership.mem t b)
    ⊢ Eq (μ (Set.preimage f t)) ((μ.restrict s) (Set.preimage f t))
  -/
  rw [Measure.restrict_apply₀ (f_mble t_mble)]
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    b : β
    f : α → β
    s : Set α
    f_mble : MeasureTheory.NullMeasurable f (μ.restrict s)
    hs : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq f fun x = …
    t : Set β
    t_mble : MeasurableSet t
    ht : Not (Membership.mem t b)
    ⊢ Eq (μ (Set.preimage f t)) (μ (Inter.inter (Set.preimage f t) s))
  -/
  rw [EventuallyEq, ae_iff, Measure.restrict_apply₀] at hs
    /-
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : MeasurableSpace β
      b : β
      f : α → β
      s : Set α
      f_mble : MeasureTheory.NullMeasurable f (μ.restrict s)
      hs : Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) b)) (HasCompl.compl s))) 0
      t : Set β
      t_mble : MeasurableSet t
      ht : Not (Membership.mem t b)
      ⊢ Eq (μ (Set.preimage f t)) (μ (Inter.inter (Set.preimage f t) s))
    -/
  · apply le_antisymm _ (measure_mono inter_subset_left)
    /-
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : MeasurableSpace β
      b : β
      f : α → β
      s : Set α
      f_mble : MeasureTheory.NullMeasurable f (μ.restrict s)
      hs : Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) b)) (HasCompl.compl s))) 0
      t : Set β
      t_mble : MeasurableSet t
      ht : Not (Membership.mem t b)
      ⊢ LE.le (μ (Set.preimage f t)) (μ (Inter.inter (Set.preimage f t) s))
    -/
    apply (measure_mono (Eq.symm (inter_union_compl (f ⁻¹' t) s)).le).trans
    /-
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : MeasurableSpace β
      b : β
      f : α → β
      s : Set α
      f_mble : MeasureTheory.NullMeasurable f (μ.restrict s)
      hs : Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) b)) (HasCompl.compl s))) 0
      t : Set β
      t_mble : MeasurableSet t
      ht : Not (Membership.mem t b)
      ⊢ LE.le (μ (Union.union (Inter.inter (Set.preimage f t) s) (Inter.inter (Set.p …
    -/
    apply (measure_union_le _ _).trans
    have obs : μ ((f ⁻¹' t) ∩ sᶜ) = 0 := by
      apply le_antisymm _ (zero_le _)
      rw [← hs]
      apply measure_mono (inter_subset_inter_left _ _)
      intro x hx hfx
      simp only [mem_preimage, mem_setOf_eq] at hx hfx
      exact ht (hfx ▸ hx)
    /-
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : MeasurableSpace β
      b : β
      f : α → β
      s : Set α
      f_mble : MeasureTheory.NullMeasurable f (μ.restrict s)
      hs : Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) b)) (HasCompl.compl s))) 0
      t : Set β
      t_mble : MeasurableSet t
      ht : Not (Membership.mem t b)
      obs : Eq (μ (Inter.inter (Set.preimage f t) (HasCompl.compl s))) 0
      ⊢ LE.le (HAdd.hAdd (μ (Inter.inter (Set.preimage f t) s)) (μ (Inter.inter (Set …
    -/
    simp only [obs, add_zero, le_refl]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : MeasurableSpace β
      b : β
      f : α → β
      s : Set α
      f_mble : MeasureTheory.NullMeasurable f (μ.restrict s)
      hs : Eq ((μ.restrict (HasCompl.compl s)) (setOf fun a => Not (Eq (f a) b))) 0
      t : Set β
      t_mble : MeasurableSet t
      ht : Not (Membership.mem t b)
      ⊢ MeasureTheory.NullMeasurableSet (setOf fun a => Not (Eq (f a) b)) (μ.restric …
    -/
  · exact NullMeasurableSet.of_null hs
    /-
      🎉 no goals
    -/


theorem MeasurableSet.nullMeasurableSet_subtype_coe {t : Set s} (hs : NullMeasurableSet s μ)
    (ht : MeasurableSet t) : NullMeasurableSet ((↑) '' t) μ := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    t : Set ↑s
    hs : MeasureTheory.NullMeasurableSet s μ
    ht : MeasurableSet t
    ⊢ MeasureTheory.NullMeasurableSet (Set.image Subtype.val t) μ
  -/
  rw [Subtype.instMeasurableSpace, comap_eq_generateFrom] at ht
  induction t, ht using generateFrom_induction with
  | hC t' ht' =>
    obtain ⟨s', hs', rfl⟩ := ht'
    rw [Subtype.image_preimage_coe]
    exact hs.inter (hs'.nullMeasurableSet)
  | empty => simp only [image_empty, nullMeasurableSet_empty]
  | compl t' _ ht' =>
    simp only [← range_diff_image Subtype.coe_injective, Subtype.range_coe_subtype, setOf_mem_eq]
    exact hs.diff ht'
  | iUnion f _ hf =>
    dsimp only []
    rw [image_iUnion]
    exact .iUnion hf


theorem NullMeasurableSet.subtype_coe {t : Set s} (hs : NullMeasurableSet s μ)
    (ht : NullMeasurableSet t (μ.comap Subtype.val)) : NullMeasurableSet (((↑) : s → α) '' t) μ :=
  NullMeasurableSet.image _ μ Subtype.coe_injective
    (fun _ => MeasurableSet.nullMeasurableSet_subtype_coe hs) ht


theorem measure_subtype_coe_le_comap (hs : NullMeasurableSet s μ) (t : Set s) :
    μ (((↑) : s → α) '' t) ≤ μ.comap Subtype.val t :=
  le_comap_apply _ _ Subtype.coe_injective (fun _ =>
    MeasurableSet.nullMeasurableSet_subtype_coe hs) _


theorem measure_subtype_coe_eq_zero_of_comap_eq_zero (hs : NullMeasurableSet s μ) {t : Set s}
    (ht : μ.comap Subtype.val t = 0) : μ (((↑) : s → α) '' t) = 0 :=
  eq_bot_iff.mpr <| (measure_subtype_coe_le_comap hs t).trans ht.le


/-- In a measure space, one can restrict the measure to a subtype to get a new measure space.
Not registered as an instance, as there are other natural choices such as the normalized restriction
for a probability measure, or the subspace measure when restricting to a vector subspace. Enable
locally if needed with `attribute [local instance] Measure.Subtype.measureSpace`. -/
noncomputable def Subtype.measureSpace : MeasureSpace (Subtype p) where
  volume := Measure.comap Subtype.val volume


theorem Subtype.volume_def : (volume : Measure u) = volume.comap Subtype.val :=
  rfl


                                  /-
                                    R : Type u_1
                                    α : Type u_2
                                    β : Type u_3
                                    δ : Type u_4
                                    γ : Type u_5
                                    ι : Type u_6
                                    m0 : MeasurableSpace α
                                    inst✝² : MeasurableSpace β
                                    inst✝¹ : MeasurableSpace γ
                                    μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
                                    s s' t : Set α
                                    u : Set δ
                                    inst✝ : MeasureTheory.MeasureSpace δ
                                    p : δ → Prop
                                    ⊢ MeasureTheory.Measure δ
                                  -/
theorem Subtype.volume_univ (hu : NullMeasurableSet u) : volume (univ : Set u) = volume u := by
                                  /-
                                    🎉 no goals
                                  -/
  /-
    δ : Type u_4
    u : Set δ
    inst✝ : MeasureTheory.MeasureSpace δ
    hu : MeasureTheory.NullMeasurableSet u MeasureTheory.MeasureSpace.volume
    ⊢ Eq (MeasureTheory.MeasureSpace.volume Set.univ) (MeasureTheory.MeasureSpace. …
  -/
  rw [Subtype.volume_def, comap_apply₀ _ _ _ _ MeasurableSet.univ.nullMeasurableSet]
    /-
      δ : Type u_4
      u : Set δ
      inst✝ : MeasureTheory.MeasureSpace δ
      hu : MeasureTheory.NullMeasurableSet u MeasureTheory.MeasureSpace.volume
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.image Subtype.val Set.univ)) (Mea …
    -/
  · congr
    /-
      case h.e_6.h
      δ : Type u_4
      u : Set δ
      inst✝ : MeasureTheory.MeasureSpace δ
      hu : MeasureTheory.NullMeasurableSet u MeasureTheory.MeasureSpace.volume
      ⊢ Eq (Set.image Subtype.val Set.univ) u
    -/
    simp only [image_univ, Subtype.range_coe_subtype, setOf_mem_eq]
    /-
      🎉 no goals
    -/
    /-
      δ : Type u_4
      u : Set δ
      inst✝ : MeasureTheory.MeasureSpace δ
      hu : MeasureTheory.NullMeasurableSet u MeasureTheory.MeasureSpace.volume
      ⊢ Function.Injective Subtype.val
    -/
  · exact Subtype.coe_injective
    /-
      🎉 no goals
    -/
    /-
      δ : Type u_4
      u : Set δ
      inst✝ : MeasureTheory.MeasureSpace δ
      hu : MeasureTheory.NullMeasurableSet u MeasureTheory.MeasureSpace.volume
      ⊢ ∀ (s : Set (Subtype fun x => Membership.mem u x)), MeasurableSet s → Measure …
    -/
  · exact fun t => MeasurableSet.nullMeasurableSet_subtype_coe hu
    /-
      🎉 no goals
    -/


                                           /-
                                             R : Type u_1
                                             α : Type u_2
                                             β : Type u_3
                                             δ : Type u_4
                                             γ : Type u_5
                                             ι : Type u_6
                                             m0 : MeasurableSpace α
                                             inst✝² : MeasurableSpace β
                                             inst✝¹ : MeasurableSpace γ
                                             μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
                                             s s' t : Set α
                                             u : Set δ
                                             inst✝ : MeasureTheory.MeasureSpace δ
                                             p : δ → Prop
                                             ⊢ MeasureTheory.Measure δ
                                           -/
theorem volume_subtype_coe_le_volume (hu : NullMeasurableSet u) (t : Set u) :
                                           /-
                                             🎉 no goals
                                           -/
    volume (((↑) : u → δ) '' t) ≤ volume t :=
  measure_subtype_coe_le_comap hu t


                                                           /-
                                                             R : Type u_1
                                                             α : Type u_2
                                                             β : Type u_3
                                                             δ : Type u_4
                                                             γ : Type u_5
                                                             ι : Type u_6
                                                             m0 : MeasurableSpace α
                                                             inst✝² : MeasurableSpace β
                                                             inst✝¹ : MeasurableSpace γ
                                                             μ μ₁ μ₂ μ₃ ν ν' ν₁ ν₂ : MeasureTheory.Measure α
                                                             s s' t : Set α
                                                             u : Set δ
                                                             inst✝ : MeasureTheory.MeasureSpace δ
                                                             p : δ → Prop
                                                             ⊢ MeasureTheory.Measure δ
                                                           -/
theorem volume_subtype_coe_eq_zero_of_volume_eq_zero (hu : NullMeasurableSet u) {t : Set u}
                                                           /-
                                                             🎉 no goals
                                                           -/
    (ht : volume t = 0) : volume (((↑) : u → δ) '' t) = 0 :=
  measure_subtype_coe_eq_zero_of_comap_eq_zero hu ht


theorem map_comap (μ : Measure β) : (comap f μ).map f = μ.restrict (range f) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ : MeasureTheory.Measure β
    ⊢ Eq (MeasureTheory.Measure.map f (MeasureTheory.Measure.comap f μ)) (μ.restri …
  -/
  ext1 t ht
  rw [hf.map_apply, comap_apply f hf.injective hf.measurableSet_image' _ (hf.measurable ht),
    image_preimage_eq_inter_range, Measure.restrict_apply ht]


theorem comap_apply (μ : Measure β) (s : Set α) : comap f μ s = μ (f '' s) :=
  calc
                                                   /-
                                                     α : Type u_2
                                                     β : Type u_3
                                                     m0 : MeasurableSpace α
                                                     m1 : MeasurableSpace β
                                                     f : α → β
                                                     hf : MeasurableEmbedding f
                                                     μ : MeasureTheory.Measure β
                                                     s : Set α
                                                     ⊢ Eq ((MeasureTheory.Measure.comap f μ) s) ((MeasureTheory.Measure.comap f μ)  …
                                                   -/
    comap f μ s = comap f μ (f ⁻¹' (f '' s)) := by rw [hf.injective.preimage_image]
                                                   /-
                                                     🎉 no goals
                                                   -/
    _ = (comap f μ).map f (f '' s) := (hf.map_apply _ _).symm
    _ = μ (f '' s) := by
      rw [hf.map_comap, restrict_apply' hf.measurableSet_range,
        inter_eq_self_of_subset_left (image_subset_range _ _)]


theorem comap_map (μ : Measure α) : (map f μ).comap f = μ := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.comap f (MeasureTheory.Measure.map f μ)) μ
  -/
  ext t _
  /-
    case h
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ : MeasureTheory.Measure α
    t : Set α
    a✝ : MeasurableSet t
    ⊢ Eq ((MeasureTheory.Measure.comap f (MeasureTheory.Measure.map f μ)) t) (μ t)
  -/
  rw [hf.comap_apply, hf.map_apply, preimage_image_eq _ hf.injective]
  /-
    🎉 no goals
  -/


theorem ae_map_iff {p : β → Prop} {μ : Measure α} : (∀ᵐ x ∂μ.map f, p x) ↔ ∀ᵐ x ∂μ, p (f x) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    p : β → Prop
    μ : MeasureTheory.Measure α
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (MeasureTheory.Measu …
  -/
  simp only [ae_iff, hf.map_apply, preimage_setOf_eq]
  /-
    🎉 no goals
  -/


theorem restrict_map (μ : Measure α) (s : Set β) :
    (μ.map f).restrict s = (μ.restrict <| f ⁻¹' s).map f :=
                             /-
                               α : Type u_2
                               β : Type u_3
                               m0 : MeasurableSpace α
                               m1 : MeasurableSpace β
                               f : α → β
                               hf : MeasurableEmbedding f
                               μ : MeasureTheory.Measure α
                               s t : Set β
                               ht : MeasurableSet t
                               ⊢ Eq (((MeasureTheory.Measure.map f μ).restrict s) t) ((MeasureTheory.Measure. …
                             -/
  Measure.ext fun t ht => by simp [hf.map_apply, ht, hf.measurable ht]
                             /-
                               🎉 no goals
                             -/


protected theorem comap_preimage (μ : Measure β) (s : Set β) :
    μ.comap f (f ⁻¹' s) = μ (s ∩ range f) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ : MeasureTheory.Measure β
    s : Set β
    ⊢ Eq ((MeasureTheory.Measure.comap f μ) (Set.preimage f s)) (μ (Inter.inter s  …
  -/
  rw [← hf.map_apply, hf.map_comap, restrict_apply' hf.measurableSet_range]
  /-
    🎉 no goals
  -/


lemma comap_restrict (μ : Measure β) (s : Set β) :
    (μ.restrict s).comap f = (μ.comap f).restrict (f ⁻¹' s) := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ : MeasureTheory.Measure β
    s : Set β
    ⊢ Eq (MeasureTheory.Measure.comap f (μ.restrict s)) ((MeasureTheory.Measure.co …
  -/
  ext t ht
  rw [Measure.restrict_apply ht, comap_apply hf, comap_apply hf,
    Measure.restrict_apply (hf.measurableSet_image.2 ht), image_inter_preimage]


lemma restrict_comap (μ : Measure β) (s : Set α) :
    (μ.comap f).restrict s = (μ.restrict (f '' s)).comap f := by
  /-
    α : Type u_2
    β : Type u_3
    m0 : MeasurableSpace α
    m1 : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ : MeasureTheory.Measure β
    s : Set α
    ⊢ Eq ((MeasureTheory.Measure.comap f μ).restrict s) (MeasureTheory.Measure.com …
  -/
  rw [comap_restrict hf, preimage_image_eq _ hf.injective]
  /-
    🎉 no goals
  -/


theorem _root_.MeasurableEquiv.restrict_map (e : α ≃ᵐ β) (μ : Measure α) (s : Set β) :
    (μ.map e).restrict s = (μ.restrict <| e ⁻¹' s).map e :=
  e.measurableEmbedding.restrict_map _ _


theorem comap_subtype_coe_apply {_m0 : MeasurableSpace α} {s : Set α} (hs : MeasurableSet s)
    (μ : Measure α) (t : Set s) : comap (↑) μ t = μ ((↑) '' t) :=
  (MeasurableEmbedding.subtype_coe hs).comap_apply _ _


theorem map_comap_subtype_coe {m0 : MeasurableSpace α} {s : Set α} (hs : MeasurableSet s)
    (μ : Measure α) : (comap (↑) μ).map ((↑) : s → α) = μ.restrict s := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.Measure.map Subtype.val (MeasureTheory.Measure.comap Subty …
  -/
  rw [(MeasurableEmbedding.subtype_coe hs).map_comap, Subtype.range_coe]
  /-
    🎉 no goals
  -/


theorem ae_restrict_iff_subtype {m0 : MeasurableSpace α} {μ : Measure α} {s : Set α}
    (hs : MeasurableSet s) {p : α → Prop} :
    (∀ᵐ x ∂μ.restrict s, p x) ↔ ∀ᵐ (x : s) ∂comap ((↑) : s → α) μ, p x := by
  /-
    α : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    p : α → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))) (Fi …
  -/
  rw [← map_comap_subtype_coe hs, (MeasurableEmbedding.subtype_coe hs).ae_map_iff]
  /-
    🎉 no goals
  -/


theorem volume_set_coe_def (s : Set α) : (volume : Measure s) = comap ((↑) : s → α) volume :=
  rfl


theorem MeasurableSet.map_coe_volume {s : Set α} (hs : MeasurableSet s) :
    volume.map ((↑) : s → α) = restrict volume s := by
  /-
    α : Type u_2
    inst✝ : MeasureTheory.MeasureSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.Measure.map Subtype.val MeasureTheory.MeasureSpace.volume) …
  -/
  rw [volume_set_coe_def, (MeasurableEmbedding.subtype_coe hs).map_comap volume, Subtype.range_coe]
  /-
    🎉 no goals
  -/


theorem volume_image_subtype_coe {s : Set α} (hs : MeasurableSet s) (t : Set s) :
    volume ((↑) '' t : Set α) = volume t :=
  (comap_subtype_coe_apply hs volume t).symm


@[simp]
                                  /-
                                    R : Type u_1
                                    α : Type u_2
                                    β : Type u_3
                                    δ : Type u_4
                                    γ : Type u_5
                                    ι : Type u_6
                                    inst✝ : MeasureTheory.MeasureSpace α
                                    s t : Set α
                                    ⊢ MeasureTheory.Measure α
                                  -/
theorem volume_preimage_coe (hs : NullMeasurableSet s) (ht : MeasurableSet t) :
                                  /-
                                    🎉 no goals
                                  -/
    volume (((↑) : s → α) ⁻¹' t) = volume (t ∩ s) := by
  rw [volume_set_coe_def,
    comap_apply₀ _ _ Subtype.coe_injective
      (fun h => MeasurableSet.nullMeasurableSet_subtype_coe hs)
      (measurable_subtype_coe ht).nullMeasurableSet,
    image_preimage_eq_inter_range, Subtype.range_coe]


theorem piecewise_ae_eq_restrict [DecidablePred (· ∈ s)] (hs : MeasurableSet s) :
    piecewise s f g =ᵐ[μ.restrict s] f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → β
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (s.piecewise f g) f
  -/
  rw [ae_restrict_eq hs]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → β
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    ⊢ (Min.min (MeasureTheory.ae μ) (Filter.principal s)).EventuallyEq (s.piecewis …
  -/
  exact (piecewise_eqOn s f g).eventuallyEq.filter_mono inf_le_right
  /-
    🎉 no goals
  -/


theorem piecewise_ae_eq_restrict_compl [DecidablePred (· ∈ s)] (hs : MeasurableSet s) :
    piecewise s f g =ᵐ[μ.restrict sᶜ] g := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → β
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq (s.piecewise …
  -/
  rw [ae_restrict_eq hs.compl]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → β
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    ⊢ (Min.min (MeasureTheory.ae μ) (Filter.principal (HasCompl.compl s))).Eventua …
  -/
  exact (piecewise_eqOn_compl s f g).eventuallyEq.filter_mono inf_le_right
  /-
    🎉 no goals
  -/


theorem piecewise_ae_eq_of_ae_eq_set [DecidablePred (· ∈ s)] [DecidablePred (· ∈ t)]
    (hst : s =ᵐ[μ] t) : s.piecewise f g =ᵐ[μ] t.piecewise f g :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    inst✝² : MeasurableSpace α
                                    μ : MeasureTheory.Measure α
                                    s t : Set α
                                    f g : α → β
                                    inst✝¹ : DecidablePred fun x => Membership.mem s x
                                    inst✝ : DecidablePred fun x => Membership.mem t x
                                    hst : (MeasureTheory.ae μ).EventuallyEq s t
                                    x : α
                                    hx : Iff (Membership.mem s x) (Membership.mem t x)
                                    ⊢ Eq (s.piecewise f g x) (t.piecewise f g x)
                                  -/
  hst.mem_iff.mono fun x hx => by simp [piecewise, hx]
                                  /-
                                    🎉 no goals
                                  -/


theorem mem_map_indicator_ae_iff_mem_map_restrict_ae_of_zero_mem [Zero β] {t : Set β}
    (ht : (0 : β) ∈ t) (hs : MeasurableSet s) :
    t ∈ Filter.map (s.indicator f) (ae μ) ↔ t ∈ Filter.map f (ae <| μ.restrict s) := by
  classical
  simp_rw [mem_map, mem_ae_iff]
  rw [Measure.restrict_apply' hs, Set.indicator_preimage, Set.ite]
  simp_rw [Set.compl_union, Set.compl_inter]
  change μ (((f ⁻¹' t)ᶜ ∪ sᶜ) ∩ ((fun _ => (0 : β)) ⁻¹' t \ s)ᶜ) = 0 ↔ μ ((f ⁻¹' t)ᶜ ∩ s) = 0
  simp only [ht, ← Set.compl_eq_univ_diff, compl_compl, Set.compl_union, if_true,
    Set.preimage_const]
  simp_rw [Set.union_inter_distrib_right, Set.compl_inter_self s, Set.union_empty]


theorem mem_map_indicator_ae_iff_of_zero_nmem [Zero β] {t : Set β} (ht : (0 : β) ∉ t) :
    t ∈ Filter.map (s.indicator f) (ae μ) ↔ μ ((f ⁻¹' t)ᶜ ∪ sᶜ) = 0 := by
  classical
  rw [mem_map, mem_ae_iff, Set.indicator_preimage, Set.ite, Set.compl_union, Set.compl_inter]
  change μ (((f ⁻¹' t)ᶜ ∪ sᶜ) ∩ ((fun _ => (0 : β)) ⁻¹' t \ s)ᶜ) = 0 ↔ μ ((f ⁻¹' t)ᶜ ∪ sᶜ) = 0
  simp only [ht, if_false, Set.compl_empty, Set.empty_diff, Set.inter_univ, Set.preimage_const]


theorem map_restrict_ae_le_map_indicator_ae [Zero β] (hs : MeasurableSet s) :
    Filter.map f (ae <| μ.restrict s) ≤ Filter.map (s.indicator f) (ae μ) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    ⊢ LE.le (Filter.map f (MeasureTheory.ae (μ.restrict s))) (Filter.map (s.indica …
  -/
  intro t
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    t : Set β
    ⊢ Membership.mem (Filter.map (s.indicator f) (MeasureTheory.ae μ)) t → Members …
  -/
  by_cases ht : (0 : β) ∈ t
    /-
      case pos
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      hs : MeasurableSet s
      t : Set β
      ht : Membership.mem t 0
      ⊢ Membership.mem (Filter.map (s.indicator f) (MeasureTheory.ae μ)) t → Members …
    -/
  · rw [mem_map_indicator_ae_iff_mem_map_restrict_ae_of_zero_mem ht hs]
    /-
      case pos
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      hs : MeasurableSet s
      t : Set β
      ht : Membership.mem t 0
      ⊢ Membership.mem (Filter.map f (MeasureTheory.ae (μ.restrict s))) t → Membersh …
    -/
    exact id
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    t : Set β
    ht : Not (Membership.mem t 0)
    ⊢ Membership.mem (Filter.map (s.indicator f) (MeasureTheory.ae μ)) t → Members …
  -/
  rw [mem_map_indicator_ae_iff_of_zero_nmem ht, mem_map_restrict_ae_iff hs]
  /-
    case neg
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    t : Set β
    ht : Not (Membership.mem t 0)
    ⊢ Eq (μ (Union.union (HasCompl.compl (Set.preimage f t)) (HasCompl.compl s)))  …
  -/
  exact fun h => measure_mono_null (Set.inter_subset_left.trans Set.subset_union_left) h
  /-
    🎉 no goals
  -/


theorem indicator_ae_eq_restrict (hs : MeasurableSet s) : indicator s f =ᵐ[μ.restrict s] f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (s.indicator f) f
  -/
  classical exact piecewise_ae_eq_restrict hs
  /-
    🎉 no goals
  -/


theorem indicator_ae_eq_restrict_compl (hs : MeasurableSet s) :
    indicator s f =ᵐ[μ.restrict sᶜ] 0 := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    ⊢ (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq (s.indicator …
  -/
  classical exact piecewise_ae_eq_restrict_compl hs
  /-
    🎉 no goals
  -/


theorem indicator_ae_eq_of_restrict_compl_ae_eq_zero (hs : MeasurableSet s)
    (hf : f =ᵐ[μ.restrict sᶜ] 0) : s.indicator f =ᵐ[μ] f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq f 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator f) f
  -/
  rw [Filter.EventuallyEq, ae_restrict_iff' hs.compl] at hf
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    hf : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (f x …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator f) f
  -/
  filter_upwards [hf] with x hx
  /-
    case h
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    hf : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (f x …
    x : α
    hx : Membership.mem (HasCompl.compl s) x → Eq (f x) (0 x)
    ⊢ Eq (s.indicator f x) (f x)
  -/
  by_cases hxs : x ∈ s
    /-
      case pos
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      hs : MeasurableSet s
      hf : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (f x …
      x : α
      hx : Membership.mem (HasCompl.compl s) x → Eq (f x) (0 x)
      hxs : Membership.mem s x
      ⊢ Eq (s.indicator f x) (f x)
    -/
  · simp only [hxs, Set.indicator_of_mem]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      hs : MeasurableSet s
      hf : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (f x …
      x : α
      hx : Membership.mem (HasCompl.compl s) x → Eq (f x) (0 x)
      hxs : Not (Membership.mem s x)
      ⊢ Eq (s.indicator f x) (f x)
    -/
  · simp only [hx hxs, Pi.zero_apply, Set.indicator_apply_eq_zero, eq_self_iff_true, imp_true_iff]
    /-
      🎉 no goals
    -/


theorem indicator_ae_eq_zero_of_restrict_ae_eq_zero (hs : MeasurableSet s)
    (hf : f =ᵐ[μ.restrict s] 0) : s.indicator f =ᵐ[μ] 0 := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator f) 0
  -/
  rw [Filter.EventuallyEq, ae_restrict_iff' hs] at hf
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    hf : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (0 x)) (Measure …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator f) 0
  -/
  filter_upwards [hf] with x hx
  /-
    case h
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    hs : MeasurableSet s
    hf : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (0 x)) (Measure …
    x : α
    hx : Membership.mem s x → Eq (f x) (0 x)
    ⊢ Eq (s.indicator f x) (0 x)
  -/
  by_cases hxs : x ∈ s
    /-
      case pos
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      hs : MeasurableSet s
      hf : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (0 x)) (Measure …
      x : α
      hx : Membership.mem s x → Eq (f x) (0 x)
      hxs : Membership.mem s x
      ⊢ Eq (s.indicator f x) (0 x)
    -/
  · simp only [hxs, hx hxs, Set.indicator_of_mem]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      hs : MeasurableSet s
      hf : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (0 x)) (Measure …
      x : α
      hx : Membership.mem s x → Eq (f x) (0 x)
      hxs : Not (Membership.mem s x)
      ⊢ Eq (s.indicator f x) (0 x)
    -/
  · simp [hx, hxs]
    /-
      🎉 no goals
    -/


theorem indicator_ae_eq_of_ae_eq_set (hst : s =ᵐ[μ] t) : s.indicator f =ᵐ[μ] t.indicator f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    f : α → β
    inst✝ : Zero β
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator f) (t.indicator f)
  -/
  classical exact piecewise_ae_eq_of_ae_eq_set hst
  /-
    🎉 no goals
  -/


theorem indicator_meas_zero (hs : μ s = 0) : indicator s f =ᵐ[μ] 0 :=
  indicator_empty' f ▸ indicator_ae_eq_of_ae_eq_set (ae_eq_empty.2 hs)


theorem ae_eq_restrict_iff_indicator_ae_eq {g : α → β} (hs : MeasurableSet s) :
    f =ᵐ[μ.restrict s] g ↔ s.indicator f =ᵐ[μ] s.indicator g := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    g : α → β
    hs : MeasurableSet s
    ⊢ Iff ((MeasureTheory.ae (μ.restrict s)).EventuallyEq f g) ((MeasureTheory.ae  …
  -/
  rw [Filter.EventuallyEq, ae_restrict_iff' hs]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → β
    inst✝ : Zero β
    g : α → β
    hs : MeasurableSet s
    ⊢ Iff (Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (Measu …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩ <;> filter_upwards [h] with x hx
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      g : α → β
      hs : MeasurableSet s
      h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (MeasureT …
      x : α
      hx : Membership.mem s x → Eq (f x) (g x)
      ⊢ Eq (s.indicator f x) (s.indicator g x)
    -/
  · by_cases hxs : x ∈ s
      /-
        case pos
        α : Type u_2
        β : Type u_3
        inst✝¹ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        f : α → β
        inst✝ : Zero β
        g : α → β
        hs : MeasurableSet s
        h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (MeasureT …
        x : α
        hx : Membership.mem s x → Eq (f x) (g x)
        hxs : Membership.mem s x
        ⊢ Eq (s.indicator f x) (s.indicator g x)
      -/
    · simp [hxs, hx hxs]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_2
        β : Type u_3
        inst✝¹ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        f : α → β
        inst✝ : Zero β
        g : α → β
        hs : MeasurableSet s
        h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (MeasureT …
        x : α
        hx : Membership.mem s x → Eq (f x) (g x)
        hxs : Not (Membership.mem s x)
        ⊢ Eq (s.indicator f x) (s.indicator g x)
      -/
    · simp [hxs]
      /-
        🎉 no goals
      -/
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      g : α → β
      hs : MeasurableSet s
      h : (MeasureTheory.ae μ).EventuallyEq (s.indicator f) (s.indicator g)
      x : α
      hx : Eq (s.indicator f x) (s.indicator g x)
      ⊢ Membership.mem s x → Eq (f x) (g x)
    -/
  · intro hxs
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → β
      inst✝ : Zero β
      g : α → β
      hs : MeasurableSet s
      h : (MeasureTheory.ae μ).EventuallyEq (s.indicator f) (s.indicator g)
      x : α
      hx : Eq (s.indicator f x) (s.indicator g x)
      hxs : Membership.mem s x
      ⊢ Eq (f x) (g x)
    -/
    simpa [hxs] using hx
    /-
      🎉 no goals
    -/


