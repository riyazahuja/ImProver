/-- Two measures `μ`, `ν` are said to be mutually singular if there exists a measurable set `s`
such that `μ s = 0` and `ν sᶜ = 0`. -/
def MutuallySingular {_ : MeasurableSpace α} (μ ν : Measure α) : Prop :=
  ∃ s : Set α, MeasurableSet s ∧ μ s = 0 ∧ ν sᶜ = 0


@[inherit_doc MeasureTheory.Measure.MutuallySingular]
scoped[MeasureTheory] infixl:60 " ⟂ₘ " => MeasureTheory.Measure.MutuallySingular


theorem mk {s t : Set α} (hs : μ s = 0) (ht : ν t = 0) (hst : univ ⊆ s ∪ t) :
    MutuallySingular μ ν := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : Eq (μ s) 0
    ht : Eq (ν t) 0
    hst : HasSubset.Subset Set.univ (Union.union s t)
    ⊢ μ.MutuallySingular ν
  -/
  use toMeasurable μ s, measurableSet_toMeasurable _ _, (measure_toMeasurable _).trans hs
  /-
    case right
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : Eq (μ s) 0
    ht : Eq (ν t) 0
    hst : HasSubset.Subset Set.univ (Union.union s t)
    ⊢ Eq (ν (HasCompl.compl (MeasureTheory.toMeasurable μ s))) 0
  -/
  refine measure_mono_null (fun x hx => (hst trivial).resolve_left fun hxs => hx ?_) ht
  /-
    case right
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    hs : Eq (μ s) 0
    ht : Eq (ν t) 0
    hst : HasSubset.Subset Set.univ (Union.union s t)
    x : α
    hx : Membership.mem (HasCompl.compl (MeasureTheory.toMeasurable μ s)) x
    hxs : Membership.mem s x
    ⊢ Membership.mem (MeasureTheory.toMeasurable μ s) x
  -/
  exact subset_toMeasurable _ _ hxs
  /-
    🎉 no goals
  -/


/-- A set such that `μ h.nullSet = 0` and `ν h.nullSetᶜ = 0`. -/
def nullSet (h : μ ⟂ₘ ν) : Set α := h.choose


lemma measurableSet_nullSet (h : μ ⟂ₘ ν) : MeasurableSet h.nullSet := h.choose_spec.1


@[simp]
lemma measure_nullSet (h : μ ⟂ₘ ν) : μ h.nullSet = 0 := h.choose_spec.2.1


@[simp]
lemma measure_compl_nullSet (h : μ ⟂ₘ ν) : ν h.nullSetᶜ = 0 := h.choose_spec.2.2

-- TODO: this is proved by simp, but is not simplified in other contexts without the @[simp]
-- attribute. Also, the linter does not complain about that attribute.

@[simp]
                                                                     /-
                                                                       α : Type u_1
                                                                       m0 : MeasurableSpace α
                                                                       μ ν : MeasureTheory.Measure α
                                                                       h : μ.MutuallySingular ν
                                                                       ⊢ Eq (μ.restrict h.nullSet) 0
                                                                     -/
lemma restrict_nullSet (h : μ ⟂ₘ ν) : μ.restrict h.nullSet = 0 := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/

-- TODO: this is proved by simp, but is not simplified in other contexts without the @[simp]
-- attribute. Also, the linter does not complain about that attribute.

@[simp]
                                                                            /-
                                                                              α : Type u_1
                                                                              m0 : MeasurableSpace α
                                                                              μ ν : MeasureTheory.Measure α
                                                                              h : μ.MutuallySingular ν
                                                                              ⊢ Eq (ν.restrict (HasCompl.compl h.nullSet)) 0
                                                                            -/
lemma restrict_compl_nullSet (h : μ ⟂ₘ ν) : ν.restrict h.nullSetᶜ = 0 := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
theorem zero_right : μ ⟂ₘ 0 :=
  ⟨∅, MeasurableSet.empty, measure_empty, rfl⟩


@[symm]
theorem symm (h : ν ⟂ₘ μ) : μ ⟂ₘ ν :=
  let ⟨i, hi, his, hit⟩ := h
  ⟨iᶜ, hi.compl, hit, (compl_compl i).symm ▸ his⟩


theorem comm : μ ⟂ₘ ν ↔ ν ⟂ₘ μ :=
  ⟨fun h => h.symm, fun h => h.symm⟩


@[simp]
theorem zero_left : 0 ⟂ₘ μ :=
  zero_right.symm


theorem mono_ac (h : μ₁ ⟂ₘ ν₁) (hμ : μ₂ ≪ μ₁) (hν : ν₂ ≪ ν₁) : μ₂ ⟂ₘ ν₂ :=
  let ⟨s, hs, h₁, h₂⟩ := h
  ⟨s, hs, hμ h₁, hν h₂⟩


lemma congr_ac (hμμ₂ : μ ≪ μ₂) (hμ₂μ : μ₂ ≪ μ) (hνν₂ : ν ≪ ν₂) (hν₂ν : ν₂ ≪ ν) :
    μ ⟂ₘ ν ↔ μ₂ ⟂ₘ ν₂ :=
  ⟨fun h ↦ h.mono_ac hμ₂μ hν₂ν, fun h ↦ h.mono_ac hμμ₂ hνν₂⟩


theorem mono (h : μ₁ ⟂ₘ ν₁) (hμ : μ₂ ≤ μ₁) (hν : ν₂ ≤ ν₁) : μ₂ ⟂ₘ ν₂ :=
  h.mono_ac hμ.absolutelyContinuous hν.absolutelyContinuous


@[simp]
lemma self_iff (μ : Measure α) : μ ⟂ₘ μ ↔ μ = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Iff (μ.MutuallySingular μ) (Eq μ 0)
  -/
  refine ⟨?_, fun h ↦ by (rw [h]; exact zero_left)⟩
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ μ.MutuallySingular μ → Eq μ 0
  -/
  rintro ⟨s, hs, hμs, hμs_compl⟩
  /-
    case intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    hμs : Eq (μ s) 0
    hμs_compl : Eq (μ (HasCompl.compl s)) 0
    ⊢ Eq μ 0
  -/
  suffices μ Set.univ = 0 by rwa [measure_univ_eq_zero] at this
  rw [← Set.union_compl_self s, measure_union disjoint_compl_right hs.compl, hμs, hμs_compl,
    add_zero]


@[simp]
theorem sum_left {ι : Type*} [Countable ι] {μ : ι → Measure α} : sum μ ⟂ₘ ν ↔ ∀ i, μ i ⟂ₘ ν := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    ν : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure α
    ⊢ Iff ((MeasureTheory.Measure.sum μ).MutuallySingular ν) (∀ (i : ι), (μ i).Mut …
  -/
  refine ⟨fun h i => h.mono (le_sum _ _) le_rfl, fun H => ?_⟩
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    ν : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure α
    H : ∀ (i : ι), (μ i).MutuallySingular ν
    ⊢ (MeasureTheory.Measure.sum μ).MutuallySingular ν
  -/
  choose s hsm hsμ hsν using H
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    ν : MeasureTheory.Measure α
    ι : Type u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure α
    s : ι → Set α
    hsm : ∀ (i : ι), MeasurableSet (s i)
    hsμ : ∀ (i : ι), Eq ((μ i) (s i)) 0
    hsν : ∀ (i : ι), Eq (ν (HasCompl.compl (s i))) 0
    ⊢ (MeasureTheory.Measure.sum μ).MutuallySingular ν
  -/
  refine ⟨⋂ i, s i, MeasurableSet.iInter hsm, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      ν : MeasureTheory.Measure α
      ι : Type u_2
      inst✝ : Countable ι
      μ : ι → MeasureTheory.Measure α
      s : ι → Set α
      hsm : ∀ (i : ι), MeasurableSet (s i)
      hsμ : ∀ (i : ι), Eq ((μ i) (s i)) 0
      hsν : ∀ (i : ι), Eq (ν (HasCompl.compl (s i))) 0
      ⊢ Eq ((MeasureTheory.Measure.sum μ) (Set.iInter fun i => s i)) 0
    -/
  · rw [sum_apply _ (MeasurableSet.iInter hsm), ENNReal.tsum_eq_zero]
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      ν : MeasureTheory.Measure α
      ι : Type u_2
      inst✝ : Countable ι
      μ : ι → MeasureTheory.Measure α
      s : ι → Set α
      hsm : ∀ (i : ι), MeasurableSet (s i)
      hsμ : ∀ (i : ι), Eq ((μ i) (s i)) 0
      hsν : ∀ (i : ι), Eq (ν (HasCompl.compl (s i))) 0
      ⊢ ∀ (i : ι), Eq ((μ i) (Set.iInter fun b => s b)) 0
    -/
    exact fun i => measure_mono_null (iInter_subset _ _) (hsμ i)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      ν : MeasureTheory.Measure α
      ι : Type u_2
      inst✝ : Countable ι
      μ : ι → MeasureTheory.Measure α
      s : ι → Set α
      hsm : ∀ (i : ι), MeasurableSet (s i)
      hsμ : ∀ (i : ι), Eq ((μ i) (s i)) 0
      hsν : ∀ (i : ι), Eq (ν (HasCompl.compl (s i))) 0
      ⊢ Eq (ν (HasCompl.compl (Set.iInter fun i => s i))) 0
    -/
  · rwa [compl_iInter, measure_iUnion_null_iff]
    /-
      🎉 no goals
    -/


@[simp]
theorem sum_right {ι : Type*} [Countable ι] {ν : ι → Measure α} : μ ⟂ₘ sum ν ↔ ∀ i, μ ⟂ₘ ν i :=
  comm.trans <| sum_left.trans <| forall_congr' fun _ => comm


@[simp]
theorem add_left_iff : μ₁ + μ₂ ⟂ₘ ν ↔ μ₁ ⟂ₘ ν ∧ μ₂ ⟂ₘ ν := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ₁ μ₂ ν : MeasureTheory.Measure α
    ⊢ Iff ((HAdd.hAdd μ₁ μ₂).MutuallySingular ν) (And (μ₁.MutuallySingular ν) (μ₂. …
  -/
  rw [← sum_cond, sum_left, Bool.forall_bool, cond, cond, and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_right_iff : μ ⟂ₘ ν₁ + ν₂ ↔ μ ⟂ₘ ν₁ ∧ μ ⟂ₘ ν₂ :=
  comm.trans <| add_left_iff.trans <| and_congr comm comm


theorem add_left (h₁ : ν₁ ⟂ₘ μ) (h₂ : ν₂ ⟂ₘ μ) : ν₁ + ν₂ ⟂ₘ μ :=
  add_left_iff.2 ⟨h₁, h₂⟩


theorem add_right (h₁ : μ ⟂ₘ ν₁) (h₂ : μ ⟂ₘ ν₂) : μ ⟂ₘ ν₁ + ν₂ :=
  add_right_iff.2 ⟨h₁, h₂⟩


theorem smul (r : ℝ≥0∞) (h : ν ⟂ₘ μ) : r • ν ⟂ₘ μ :=
  h.mono_ac (AbsolutelyContinuous.rfl.smul_left r) AbsolutelyContinuous.rfl


theorem smul_nnreal (r : ℝ≥0) (h : ν ⟂ₘ μ) : r • ν ⟂ₘ μ :=
  h.smul r


lemma restrict (h : μ ⟂ₘ ν) (s : Set α) : μ.restrict s ⟂ₘ ν := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    s : Set α
    ⊢ (μ.restrict s).MutuallySingular ν
  -/
  refine ⟨h.nullSet, h.measurableSet_nullSet, ?_, h.measure_compl_nullSet⟩
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    s : Set α
    ⊢ Eq ((μ.restrict s) h.nullSet) 0
  -/
  rw [Measure.restrict_apply h.measurableSet_nullSet]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    s : Set α
    ⊢ Eq (μ (Inter.inter h.nullSet s)) 0
  -/
  exact measure_mono_null Set.inter_subset_left h.measure_nullSet
  /-
    🎉 no goals
  -/


lemma eq_zero_of_absolutelyContinuous_of_mutuallySingular {μ ν : Measure α}
    (h_ac : μ ≪ ν) (h_ms : μ ⟂ₘ ν) :
    μ = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h_ac : μ.AbsolutelyContinuous ν
    h_ms : μ.MutuallySingular ν
    ⊢ Eq μ 0
  -/
  rw [← Measure.MutuallySingular.self_iff]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h_ac : μ.AbsolutelyContinuous ν
    h_ms : μ.MutuallySingular ν
    ⊢ μ.MutuallySingular μ
  -/
  exact h_ms.mono_ac Measure.AbsolutelyContinuous.rfl h_ac
  /-
    🎉 no goals
  -/


lemma absolutelyContinuous_of_add_of_mutuallySingular {ν₁ ν₂ : Measure α}
    (h : μ ≪ ν₁ + ν₂) (h_ms : μ ⟂ₘ ν₂) : μ ≪ ν₁ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    ⊢ μ.AbsolutelyContinuous ν₁
  -/
  refine AbsolutelyContinuous.mk fun s hs hs_zero ↦ ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (ν₁ s) 0
    ⊢ Eq (μ s) 0
  -/
  let t := h_ms.nullSet
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (ν₁ s) 0
    t : Set α := h_ms.nullSet
    ⊢ Eq (μ s) 0
  -/
  have ht : MeasurableSet t := h_ms.measurableSet_nullSet
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (ν₁ s) 0
    t : Set α := h_ms.nullSet
    ht : MeasurableSet t
    ⊢ Eq (μ s) 0
  -/
  have htμ : μ t = 0 := h_ms.measure_nullSet
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (ν₁ s) 0
    t : Set α := h_ms.nullSet
    ht : MeasurableSet t
    htμ : Eq (μ t) 0
    ⊢ Eq (μ s) 0
  -/
  have htν₂ : ν₂ tᶜ = 0 := h_ms.measure_compl_nullSet
  have : μ s = μ (s ∩ tᶜ) := by
    conv_lhs => rw [← inter_union_compl s t]
    rw [measure_union, measure_inter_null_of_null_right _ htμ, zero_add]
    · exact (disjoint_compl_right.inter_right' _ ).inter_left' _
    · exact hs.inter ht.compl
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (ν₁ s) 0
    t : Set α := h_ms.nullSet
    ht : MeasurableSet t
    htμ : Eq (μ t) 0
    htν₂ : Eq (ν₂ (HasCompl.compl t)) 0
    this : Eq (μ s) (μ (Inter.inter s (HasCompl.compl t)))
    ⊢ Eq (μ s) 0
  -/
  rw [this]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (ν₁ s) 0
    t : Set α := h_ms.nullSet
    ht : MeasurableSet t
    htμ : Eq (μ t) 0
    htν₂ : Eq (ν₂ (HasCompl.compl t)) 0
    this : Eq (μ s) (μ (Inter.inter s (HasCompl.compl t)))
    ⊢ Eq (μ (Inter.inter s (HasCompl.compl t))) 0
  -/
  refine h ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (ν₁ s) 0
    t : Set α := h_ms.nullSet
    ht : MeasurableSet t
    htμ : Eq (μ t) 0
    htν₂ : Eq (ν₂ (HasCompl.compl t)) 0
    this : Eq (μ s) (μ (Inter.inter s (HasCompl.compl t)))
    ⊢ Eq ((HAdd.hAdd ν₁ ν₂) (Inter.inter s (HasCompl.compl t))) 0
  -/
  simp only [Measure.coe_add, Pi.add_apply, add_eq_zero]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    h : μ.AbsolutelyContinuous (HAdd.hAdd ν₁ ν₂)
    h_ms : μ.MutuallySingular ν₂
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (ν₁ s) 0
    t : Set α := h_ms.nullSet
    ht : MeasurableSet t
    htμ : Eq (μ t) 0
    htν₂ : Eq (ν₂ (HasCompl.compl t)) 0
    this : Eq (μ s) (μ (Inter.inter s (HasCompl.compl t)))
    ⊢ And (Eq (ν₁ (Inter.inter s (HasCompl.compl t))) 0) (Eq (ν₂ (Inter.inter s (H …
  -/
  exact ⟨measure_inter_null_of_null_left _ hs_zero, measure_inter_null_of_null_right _ htν₂⟩
  /-
    🎉 no goals
  -/


lemma _root_.MeasurableEmbedding.mutuallySingular_map {β : Type*} {_ : MeasurableSpace β}
    {f : α → β} (hf : MeasurableEmbedding f) (hμν : μ ⟂ₘ ν) :
    μ.map f ⟂ₘ ν.map f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    β : Type u_2
    x✝ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    hμν : μ.MutuallySingular ν
    ⊢ (MeasureTheory.Measure.map f μ).MutuallySingular (MeasureTheory.Measure.map  …
  -/
  refine ⟨f '' hμν.nullSet, hf.measurableSet_image' hμν.measurableSet_nullSet, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      β : Type u_2
      x✝ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      hμν : μ.MutuallySingular ν
      ⊢ Eq ((MeasureTheory.Measure.map f μ) (Set.image f hμν.nullSet)) 0
    -/
  · rw [hf.map_apply, hf.injective.preimage_image, hμν.measure_nullSet]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      β : Type u_2
      x✝ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      hμν : μ.MutuallySingular ν
      ⊢ Eq ((MeasureTheory.Measure.map f ν) (HasCompl.compl (Set.image f hμν.nullSet …
    -/
  · rw [hf.map_apply, Set.preimage_compl, hf.injective.preimage_image, hμν.measure_compl_nullSet]
    /-
      🎉 no goals
    -/


lemma exists_null_set_measure_lt_of_disjoint (h : Disjoint μ ν) {ε : ℝ≥0} (hε : 0 < ε) :
    ∃ s, μ s = 0 ∧ ν sᶜ ≤ 2 * ε := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Disjoint μ ν
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun s => And (Eq (μ s) 0) (LE.le (ν (HasCompl.compl s)) (HMul.hMul 2  …
  -/
  have h₁ : (μ ⊓ ν) univ = 0 := le_bot_iff.1 (h (inf_le_left (b := ν)) inf_le_right) ▸ rfl
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Disjoint μ ν
    ε : NNReal
    hε : LT.lt 0 ε
    h₁ : Eq ((Min.min μ ν) Set.univ) 0
    ⊢ Exists fun s => And (Eq (μ s) 0) (LE.le (ν (HasCompl.compl s)) (HMul.hMul 2  …
  -/
  simp_rw [Measure.inf_apply MeasurableSet.univ, inter_univ] at h₁
  have h₂ : ∀ n : ℕ, ∃ t, μ t + ν tᶜ < ε * (1 / 2) ^ n := by
    intro n
    obtain ⟨m, ⟨t, ht₁, rfl⟩, hm₂⟩ :
        ∃ x ∈ {m | ∃ t, m = μ t + ν tᶜ}, x < ε * (1 / 2 : ℝ≥0∞) ^ n := by
      refine exists_lt_of_csInf_lt ⟨ν univ, ∅, by simp⟩ <| h₁ ▸ ENNReal.mul_pos ?_ (by simp)
      norm_cast
      exact hε.ne.symm
    exact ⟨t, hm₂⟩
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Disjoint μ ν
    ε : NNReal
    hε : LT.lt 0 ε
    h₁ : Eq (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ t) (ν  …
    h₂ : ∀ (n : Nat), Exists fun t => LT.lt (HAdd.hAdd (μ t) (ν (HasCompl.compl t) …
    ⊢ Exists fun s => And (Eq (μ s) 0) (LE.le (ν (HasCompl.compl s)) (HMul.hMul 2  …
  -/
  choose t ht₂ using h₂
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Disjoint μ ν
    ε : NNReal
    hε : LT.lt 0 ε
    h₁ : Eq (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ t) (ν  …
    t : Nat → Set α
    ht₂ : ∀ (n : Nat), LT.lt (HAdd.hAdd (μ (t n)) (ν (HasCompl.compl (t n)))) (HMu …
    ⊢ Exists fun s => And (Eq (μ s) 0) (LE.le (ν (HasCompl.compl s)) (HMul.hMul 2  …
  -/
  refine ⟨⋂ n, t n, ?_, ?_⟩
  · refine eq_zero_of_le_mul_pow (by norm_num)
      fun n ↦ ((measure_mono <| iInter_subset_of_subset n fun _ ht ↦ ht).trans
      (le_add_right le_rfl)).trans (ht₂ n).le
  · rw [compl_iInter, (by simp [ENNReal.tsum_mul_left, mul_comm] :
      2 * (ε : ℝ≥0∞) = ∑' (n : ℕ), ε * (1 / 2 : ℝ≥0∞) ^ n)]
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Disjoint μ ν
      ε : NNReal
      hε : LT.lt 0 ε
      h₁ : Eq (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ t) (ν  …
      t : Nat → Set α
      ht₂ : ∀ (n : Nat), LT.lt (HAdd.hAdd (μ (t n)) (ν (HasCompl.compl (t n)))) (HMu …
      ⊢ LE.le (ν (Set.iUnion fun i => HasCompl.compl (t i))) (tsum fun n => HMul.hMu …
    -/
    refine (measure_iUnion_le _).trans ?_
    exact tsum_le_tsum (fun n ↦ (le_add_left le_rfl).trans (ht₂ n).le)
      ENNReal.summable ENNReal.summable


lemma mutuallySingular_of_disjoint (h : Disjoint μ ν) : μ ⟂ₘ ν := by
  have h' (n : ℕ) : ∃ s, μ s = 0 ∧ ν sᶜ ≤ (1 / 2) ^ n := by
    convert exists_null_set_measure_lt_of_disjoint h (ε := (1 / 2) ^ (n + 1))
      <| pow_pos (by simp) (n + 1)
    push_cast
    rw [pow_succ, ← mul_assoc, mul_comm, ← mul_assoc]
    norm_cast
    rw [div_mul_cancel₀, one_mul]
    · push_cast
      simp
    · simp
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Disjoint μ ν
    h' : ∀ (n : Nat), Exists fun s => And (Eq (μ s) 0) (LE.le (ν (HasCompl.compl s …
    ⊢ μ.MutuallySingular ν
  -/
  choose s hs₂ hs₃ using h'
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Disjoint μ ν
    s : Nat → Set α
    hs₂ : ∀ (n : Nat), Eq (μ (s n)) 0
    hs₃ : ∀ (n : Nat), LE.le (ν (HasCompl.compl (s n))) (HPow.hPow (1 / 2) n)
    ⊢ μ.MutuallySingular ν
  -/
  refine Measure.MutuallySingular.mk (t := (⋃ n, s n)ᶜ) (measure_iUnion_null hs₂) ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Disjoint μ ν
      s : Nat → Set α
      hs₂ : ∀ (n : Nat), Eq (μ (s n)) 0
      hs₃ : ∀ (n : Nat), LE.le (ν (HasCompl.compl (s n))) (HPow.hPow (1 / 2) n)
      ⊢ Eq (ν (HasCompl.compl (Set.iUnion fun n => s n))) 0
    -/
  · rw [compl_iUnion]
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Disjoint μ ν
      s : Nat → Set α
      hs₂ : ∀ (n : Nat), Eq (μ (s n)) 0
      hs₃ : ∀ (n : Nat), LE.le (ν (HasCompl.compl (s n))) (HPow.hPow (1 / 2) n)
      ⊢ Eq (ν (Set.iInter fun i => HasCompl.compl (s i))) 0
    -/
    refine eq_zero_of_le_mul_pow (ε := 1) (by norm_num : (1 / 2 : ℝ≥0∞) < 1) <| fun n ↦ ?_
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Disjoint μ ν
      s : Nat → Set α
      hs₂ : ∀ (n : Nat), Eq (μ (s n)) 0
      hs₃ : ∀ (n : Nat), LE.le (ν (HasCompl.compl (s n))) (HPow.hPow (1 / 2) n)
      n : Nat
      ⊢ LE.le (ν (Set.iInter fun i => HasCompl.compl (s i))) (HMul.hMul (↑1) (HPow.h …
    -/
    rw [ENNReal.coe_one, one_mul]
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Disjoint μ ν
      s : Nat → Set α
      hs₂ : ∀ (n : Nat), Eq (μ (s n)) 0
      hs₃ : ∀ (n : Nat), LE.le (ν (HasCompl.compl (s n))) (HPow.hPow (1 / 2) n)
      n : Nat
      ⊢ LE.le (ν (Set.iInter fun i => HasCompl.compl (s i))) (HPow.hPow (1 / 2) n)
    -/
    exact (measure_mono <| iInter_subset_of_subset n fun _ ht ↦ ht).trans (hs₃ n)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Disjoint μ ν
      s : Nat → Set α
      hs₂ : ∀ (n : Nat), Eq (μ (s n)) 0
      hs₃ : ∀ (n : Nat), LE.le (ν (HasCompl.compl (s n))) (HPow.hPow (1 / 2) n)
      ⊢ HasSubset.Subset Set.univ (Union.union (Set.iUnion fun i => s i) (HasCompl.c …
    -/
  · rw [union_compl_self]
    /-
      🎉 no goals
    -/


lemma MutuallySingular.disjoint (h : μ ⟂ₘ ν) : Disjoint μ ν := by
  have h_bot_iff (ξ : Measure α) : ξ ≤ ⊥ ↔ ξ = 0 := by
    rw [le_bot_iff]
    rfl
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    h_bot_iff : ∀ (ξ : MeasureTheory.Measure α), Iff (LE.le ξ Bot.bot) (Eq ξ 0)
    ⊢ Disjoint μ ν
  -/
  intro ξ hξμ hξν
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    h_bot_iff : ∀ (ξ : MeasureTheory.Measure α), Iff (LE.le ξ Bot.bot) (Eq ξ 0)
    ξ : MeasureTheory.Measure α
    hξμ : LE.le ξ μ
    hξν : LE.le ξ ν
    ⊢ LE.le ξ Bot.bot
  -/
  rw [h_bot_iff]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    h_bot_iff : ∀ (ξ : MeasureTheory.Measure α), Iff (LE.le ξ Bot.bot) (Eq ξ 0)
    ξ : MeasureTheory.Measure α
    hξμ : LE.le ξ μ
    hξν : LE.le ξ ν
    ⊢ Eq ξ 0
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    h_bot_iff : ∀ (ξ : MeasureTheory.Measure α), Iff (LE.le ξ Bot.bot) (Eq ξ 0)
    ξ : MeasureTheory.Measure α
    hξμ : LE.le ξ μ
    hξν : LE.le ξ ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (ξ s) (0 s)
  -/
  simp only [Measure.coe_zero, Pi.zero_apply]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    h_bot_iff : ∀ (ξ : MeasureTheory.Measure α), Iff (LE.le ξ Bot.bot) (Eq ξ 0)
    ξ : MeasureTheory.Measure α
    hξμ : LE.le ξ μ
    hξν : LE.le ξ ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (ξ s) 0
  -/
  rw [← inter_union_compl s h.nullSet, measure_union, add_eq_zero]
  · exact ⟨measure_inter_null_of_null_right _ <| absolutelyContinuous_of_le hξμ h.measure_nullSet,
      measure_inter_null_of_null_right _ <| absolutelyContinuous_of_le hξν h.measure_compl_nullSet⟩
    /-
      case h.hd
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : μ.MutuallySingular ν
      h_bot_iff : ∀ (ξ : MeasureTheory.Measure α), Iff (LE.le ξ Bot.bot) (Eq ξ 0)
      ξ : MeasureTheory.Measure α
      hξμ : LE.le ξ μ
      hξν : LE.le ξ ν
      s : Set α
      hs : MeasurableSet s
      ⊢ Disjoint (Inter.inter s h.nullSet) (Inter.inter s (HasCompl.compl h.nullSet))
    -/
  · exact Disjoint.mono inter_subset_right inter_subset_right disjoint_compl_right
    /-
      🎉 no goals
    -/
    /-
      case h.h
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : μ.MutuallySingular ν
      h_bot_iff : ∀ (ξ : MeasureTheory.Measure α), Iff (LE.le ξ Bot.bot) (Eq ξ 0)
      ξ : MeasureTheory.Measure α
      hξμ : LE.le ξ μ
      hξν : LE.le ξ ν
      s : Set α
      hs : MeasurableSet s
      ⊢ MeasurableSet (Inter.inter s (HasCompl.compl h.nullSet))
    -/
  · exact hs.inter h.measurableSet_nullSet.compl
    /-
      🎉 no goals
    -/


lemma MutuallySingular.disjoint_ae (h : μ ⟂ₘ ν) : Disjoint (ae μ) (ae ν) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    ⊢ Disjoint (MeasureTheory.ae μ) (MeasureTheory.ae ν)
  -/
  rw [disjoint_iff_inf_le]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    ⊢ LE.le (Min.min (MeasureTheory.ae μ) (MeasureTheory.ae ν)) Bot.bot
  -/
  intro s _
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : μ.MutuallySingular ν
    s : Set α
    a✝ : Membership.mem Bot.bot s
    ⊢ Membership.mem (Min.min (MeasureTheory.ae μ) (MeasureTheory.ae ν)) s
  -/
  refine ⟨s ∪ h.nullSetᶜ, ?_, s ∪ h.nullSet, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : μ.MutuallySingular ν
      s : Set α
      a✝ : Membership.mem Bot.bot s
      ⊢ Membership.mem (MeasureTheory.ae μ) (Union.union s (HasCompl.compl h.nullSet))
    -/
  · rw [mem_ae_iff, compl_union, compl_compl]
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : μ.MutuallySingular ν
      s : Set α
      a✝ : Membership.mem Bot.bot s
      ⊢ Eq (μ (Inter.inter (HasCompl.compl s) h.nullSet)) 0
    -/
    exact measure_inter_null_of_null_right _ h.measure_nullSet
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : μ.MutuallySingular ν
      s : Set α
      a✝ : Membership.mem Bot.bot s
      ⊢ Membership.mem (MeasureTheory.ae ν) (Union.union s h.nullSet)
    -/
  · rw [mem_ae_iff, compl_union]
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : μ.MutuallySingular ν
      s : Set α
      a✝ : Membership.mem Bot.bot s
      ⊢ Eq (ν (Inter.inter (HasCompl.compl s) (HasCompl.compl h.nullSet))) 0
    -/
    exact measure_inter_null_of_null_right _ h.measure_compl_nullSet
    /-
      🎉 no goals
    -/
  · rw [union_eq_compl_compl_inter_compl, union_eq_compl_compl_inter_compl,
      ← compl_union, compl_compl, inter_union_compl, compl_compl]


lemma disjoint_of_disjoint_ae (h : Disjoint (ae μ) (ae ν)) : Disjoint μ ν := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : Disjoint (MeasureTheory.ae μ) (MeasureTheory.ae ν)
    ⊢ Disjoint μ ν
  -/
  rw [disjoint_iff_inf_le] at h ⊢
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : LE.le (Min.min (MeasureTheory.ae μ) (MeasureTheory.ae ν)) Bot.bot
    ⊢ LE.le (Min.min μ ν) Bot.bot
  -/
  refine Measure.le_intro fun s hs _ ↦ ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : LE.le (Min.min (MeasureTheory.ae μ) (MeasureTheory.ae ν)) Bot.bot
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    ⊢ LE.le ((Min.min μ ν) s) (Bot.bot s)
  -/
  rw [Measure.inf_apply hs]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : LE.le (Min.min (MeasureTheory.ae μ) (MeasureTheory.ae ν)) Bot.bot
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    ⊢ LE.le (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter …
  -/
  have : (⊥ : Measure α) = 0 := rfl
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : LE.le (Min.min (MeasureTheory.ae μ) (MeasureTheory.ae ν)) Bot.bot
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    ⊢ LE.le (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter …
  -/
  simp only [this, Measure.coe_zero, Pi.zero_apply, nonpos_iff_eq_zero]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    h : LE.le (Min.min (MeasureTheory.ae μ) (MeasureTheory.ae ν)) Bot.bot
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    ⊢ Eq (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter.in …
  -/
  specialize h (mem_bot (s := sᶜ))
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    h : Membership.mem (Min.min (MeasureTheory.ae μ) (MeasureTheory.ae ν)) (HasCom …
    ⊢ Eq (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter.in …
  -/
  rw [mem_inf_iff] at h
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    h : Exists fun t₁ => And (Membership.mem (MeasureTheory.ae μ) t₁) (Exists fun  …
    ⊢ Eq (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter.in …
  -/
  obtain ⟨t₁, ht₁, t₂, ht₂, h_eq'⟩ := h
  have h_eq : s = t₁ᶜ ∪ t₂ᶜ := by
    rw [union_eq_compl_compl_inter_compl, compl_compl, compl_compl, ← h_eq', compl_compl]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    t₁ : Set α
    ht₁ : Membership.mem (MeasureTheory.ae μ) t₁
    t₂ : Set α
    ht₂ : Membership.mem (MeasureTheory.ae ν) t₂
    h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
    h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
    ⊢ Eq (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter.in …
  -/
  rw [mem_ae_iff] at ht₁ ht₂
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    t₁ : Set α
    ht₁ : Eq (μ (HasCompl.compl t₁)) 0
    t₂ : Set α
    ht₂ : Eq (ν (HasCompl.compl t₂)) 0
    h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
    h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
    ⊢ Eq (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter.in …
  -/
  refine le_antisymm ?_ zero_le'
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    t₁ : Set α
    ht₁ : Eq (μ (HasCompl.compl t₁)) 0
    t₂ : Set α
    ht₂ : Eq (ν (HasCompl.compl t₂)) 0
    h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
    h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
    ⊢ LE.le (InfSet.sInf (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter …
  -/
  refine sInf_le_of_le (a := 0) (b := 0) ?_ le_rfl
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    t₁ : Set α
    ht₁ : Eq (μ (HasCompl.compl t₁)) 0
    t₂ : Set α
    ht₂ : Eq (ν (HasCompl.compl t₂)) 0
    h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
    h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
    ⊢ Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter.int …
  -/
  rw [h_eq]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    t₁ : Set α
    ht₁ : Eq (μ (HasCompl.compl t₁)) 0
    t₂ : Set α
    ht₂ : Eq (ν (HasCompl.compl t₂)) 0
    h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
    h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
    ⊢ Membership.mem (setOf fun m => Exists fun t => Eq m (HAdd.hAdd (μ (Inter.int …
  -/
  refine ⟨t₁ᶜ ∩ t₂, Eq.symm ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    t₁ : Set α
    ht₁ : Eq (μ (HasCompl.compl t₁)) 0
    t₂ : Set α
    ht₂ : Eq (ν (HasCompl.compl t₂)) 0
    h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
    h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
    ⊢ Eq (HAdd.hAdd (μ (Inter.inter (Inter.inter (HasCompl.compl t₁) t₂) (Union.un …
  -/
  rw [add_eq_zero]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    x✝ : s.Nonempty
    this : Eq Bot.bot 0
    t₁ : Set α
    ht₁ : Eq (μ (HasCompl.compl t₁)) 0
    t₂ : Set α
    ht₂ : Eq (ν (HasCompl.compl t₂)) 0
    h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
    h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
    ⊢ And (Eq (μ (Inter.inter (Inter.inter (HasCompl.compl t₁) t₂) (Union.union (H …
  -/
  constructor
    /-
      case intro.intro.intro.intro.left
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      x✝ : s.Nonempty
      this : Eq Bot.bot 0
      t₁ : Set α
      ht₁ : Eq (μ (HasCompl.compl t₁)) 0
      t₂ : Set α
      ht₂ : Eq (ν (HasCompl.compl t₂)) 0
      h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
      h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
      ⊢ Eq (μ (Inter.inter (Inter.inter (HasCompl.compl t₁) t₂) (Union.union (HasCom …
    -/
  · refine measure_inter_null_of_null_left _ ?_
    /-
      case intro.intro.intro.intro.left
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      x✝ : s.Nonempty
      this : Eq Bot.bot 0
      t₁ : Set α
      ht₁ : Eq (μ (HasCompl.compl t₁)) 0
      t₂ : Set α
      ht₂ : Eq (ν (HasCompl.compl t₂)) 0
      h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
      h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
      ⊢ Eq (μ (Inter.inter (HasCompl.compl t₁) t₂)) 0
    -/
    exact measure_inter_null_of_null_left _ ht₁
    /-
      🎉 no goals
    -/
  · rw [compl_inter, compl_compl, union_eq_compl_compl_inter_compl,
      union_eq_compl_compl_inter_compl, ← compl_union, compl_compl, compl_compl, inter_comm,
      inter_comm t₁, union_comm, inter_union_compl]
    /-
      case intro.intro.intro.intro.right
      α : Type u_1
      m0 : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      x✝ : s.Nonempty
      this : Eq Bot.bot 0
      t₁ : Set α
      ht₁ : Eq (μ (HasCompl.compl t₁)) 0
      t₂ : Set α
      ht₂ : Eq (ν (HasCompl.compl t₂)) 0
      h_eq' : Eq (HasCompl.compl s) (Inter.inter t₁ t₂)
      h_eq : Eq s (Union.union (HasCompl.compl t₁) (HasCompl.compl t₂))
      ⊢ Eq (ν (HasCompl.compl t₂)) 0
    -/
    exact ht₂
    /-
      🎉 no goals
    -/


lemma mutuallySingular_tfae : List.TFAE
    [ μ ⟂ₘ ν,
      Disjoint μ ν,
      Disjoint (ae μ) (ae ν) ] := by
  tfae_have 1 → 2
  | h => h.disjoint
  tfae_have 2 → 1
  | h => mutuallySingular_of_disjoint h
  tfae_have 1 → 3
  | h => h.disjoint_ae
  tfae_have 3 → 2
  | h => disjoint_of_disjoint_ae h
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    tfae_1_to_2 : μ.MutuallySingular ν → Disjoint μ ν
    tfae_2_to_1 : Disjoint μ ν → μ.MutuallySingular ν
    tfae_1_to_3 : μ.MutuallySingular ν → Disjoint (MeasureTheory.ae μ) (MeasureThe …
    tfae_3_to_2 : Disjoint (MeasureTheory.ae μ) (MeasureTheory.ae ν) → Disjoint μ ν
    ⊢ (List.cons (μ.MutuallySingular ν) (List.cons (Disjoint μ ν) (List.cons (Disj …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma mutuallySingular_iff_disjoint : μ ⟂ₘ ν ↔ Disjoint μ ν :=
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ Eq ((List.cons (MeasureTheory.Measure.MutuallySingular ?m.47301 ?m.47302) (L …
  -/
  /-
    🎉 no goals
  -/
  mutuallySingular_tfae.out 0 1
  /-
    🎉 no goals
  -/


lemma mutuallySingular_iff_disjoint_ae : μ ⟂ₘ ν ↔ Disjoint (ae μ) (ae ν) :=
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ Eq ((List.cons (MeasureTheory.Measure.MutuallySingular ?m.47938 ?m.47939) (L …
  -/
  /-
    🎉 no goals
  -/
  mutuallySingular_tfae.out 0 2
  /-
    🎉 no goals
  -/


