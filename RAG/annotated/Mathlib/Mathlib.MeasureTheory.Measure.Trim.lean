/-- Restriction of a measure to a sub-σ-algebra.
It is common to see a measure `μ` on a measurable space structure `m0` as being also a measure on
any `m ≤ m0`. Since measures in mathlib have to be trimmed to the measurable space, `μ` itself
cannot be a measure on `m`, hence the definition of `μ.trim hm`.

This notion is related to `OuterMeasure.trim`, see the lemma
`toOuterMeasure_trim_eq_trim_toOuterMeasure`. -/
noncomputable
def Measure.trim {m m0 : MeasurableSpace α} (μ : @Measure α m0) (hm : m ≤ m0) : @Measure α m :=
  @OuterMeasure.toMeasure α m μ.toOuterMeasure (hm.trans (le_toOuterMeasure_caratheodory μ))


@[simp]
theorem trim_eq_self [MeasurableSpace α] {μ : Measure α} : μ.trim le_rfl = μ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (μ.trim ⋯) μ
  -/
  simp [Measure.trim]
  /-
    🎉 no goals
  -/


theorem toOuterMeasure_trim_eq_trim_toOuterMeasure (μ : Measure α) (hm : m ≤ m0) :
    @Measure.toOuterMeasure _ m (μ.trim hm) = @OuterMeasure.trim _ m μ.toOuterMeasure := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    ⊢ Eq (μ.trim hm).toOuterMeasure μ.trim
  -/
  rw [Measure.trim, toMeasure_toOuterMeasure (ms := m)]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_trim (hm : m ≤ m0) : (0 : Measure α).trim hm = (0 : @Measure α m) := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    hm : LE.le m m0
    ⊢ Eq (MeasureTheory.Measure.trim 0 hm) 0
  -/
  simp [Measure.trim, @OuterMeasure.toMeasure_zero _ m]
  /-
    🎉 no goals
  -/


theorem trim_measurableSet_eq (hm : m ≤ m0) (hs : @MeasurableSet α m s) : μ.trim hm s = μ s := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    ⊢ Eq ((μ.trim hm) s) (μ s)
  -/
  rw [Measure.trim, toMeasure_apply (ms := m) _ _ hs, Measure.coe_toOuterMeasure]
  /-
    🎉 no goals
  -/


theorem le_trim (hm : m ≤ m0) : μ s ≤ μ.trim hm s := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    ⊢ LE.le (μ s) ((μ.trim hm) s)
  -/
  simp_rw [Measure.trim]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    ⊢ LE.le (μ s) ((μ.toMeasure ⋯) s)
  -/
  exact @le_toMeasure_apply _ m _ _ _
  /-
    🎉 no goals
  -/


lemma trim_add {ν : Measure α} (hm : m ≤ m0) : (μ + ν).trim hm = μ.trim hm + ν.trim hm :=
                                      /-
                                        α : Type u_1
                                        m m0 : MeasurableSpace α
                                        μ ν : MeasureTheory.Measure α
                                        hm : LE.le m m0
                                        s : Set α
                                        hs : MeasurableSet s
                                        ⊢ Eq (((HAdd.hAdd μ ν).trim hm) s) ((HAdd.hAdd (μ.trim hm) (ν.trim hm)) s)
                                      -/
  @Measure.ext _ m _ _ (fun s hs ↦ by simp [trim_measurableSet_eq hm hs])
                                      /-
                                        🎉 no goals
                                      -/


theorem measure_eq_zero_of_trim_eq_zero (hm : m ≤ m0) (h : μ.trim hm s = 0) : μ s = 0 :=
  le_antisymm ((le_trim hm).trans (le_of_eq h)) (zero_le _)


theorem measure_trim_toMeasurable_eq_zero {hm : m ≤ m0} (hs : μ.trim hm s = 0) :
    μ (@toMeasurable α m (μ.trim hm) s) = 0 :=
                                         /-
                                           α : Type u_1
                                           m m0 : MeasurableSpace α
                                           μ : MeasureTheory.Measure α
                                           s : Set α
                                           hm : LE.le m m0
                                           hs : Eq ((μ.trim hm) s) 0
                                           ⊢ Eq ((μ.trim hm) (MeasureTheory.toMeasurable (μ.trim hm) s)) 0
                                         -/
  measure_eq_zero_of_trim_eq_zero hm (by rwa [@measure_toMeasurable _ m])
                                         /-
                                           🎉 no goals
                                         -/


theorem ae_of_ae_trim (hm : m ≤ m0) {μ : Measure α} {P : α → Prop} (h : ∀ᵐ x ∂μ.trim hm, P x) :
    ∀ᵐ x ∂μ, P x :=
  measure_eq_zero_of_trim_eq_zero hm h


theorem ae_eq_of_ae_eq_trim {E} {hm : m ≤ m0} {f₁ f₂ : α → E}
    (h12 : f₁ =ᵐ[μ.trim hm] f₂) : f₁ =ᵐ[μ] f₂ :=
  measure_eq_zero_of_trim_eq_zero hm h12


theorem ae_le_of_ae_le_trim {E} [LE E] {hm : m ≤ m0} {f₁ f₂ : α → E}
    (h12 : f₁ ≤ᵐ[μ.trim hm] f₂) : f₁ ≤ᵐ[μ] f₂ :=
  measure_eq_zero_of_trim_eq_zero hm h12


theorem trim_trim {m₁ m₂ : MeasurableSpace α} {hm₁₂ : m₁ ≤ m₂} {hm₂ : m₂ ≤ m0} :
    (μ.trim hm₂).trim hm₁₂ = μ.trim (hm₁₂.trans hm₂) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    m₁ m₂ : MeasurableSpace α
    hm₁₂ : LE.le m₁ m₂
    hm₂ : LE.le m₂ m0
    ⊢ Eq ((μ.trim hm₂).trim hm₁₂) (μ.trim ⋯)
  -/
  refine @Measure.ext _ m₁ _ _ (fun t ht => ?_)
  rw [trim_measurableSet_eq hm₁₂ ht, trim_measurableSet_eq (hm₁₂.trans hm₂) ht,
    trim_measurableSet_eq hm₂ (hm₁₂ t ht)]


theorem restrict_trim (hm : m ≤ m0) (μ : Measure α) (hs : @MeasurableSet α m s) :
    @Measure.restrict α m (μ.trim hm) s = (μ.restrict s).trim hm := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    s : Set α
    hm : LE.le m m0
    μ : MeasureTheory.Measure α
    hs : MeasurableSet s
    ⊢ Eq ((μ.trim hm).restrict s) ((μ.restrict s).trim hm)
  -/
  refine @Measure.ext _ m _ _ (fun t ht => ?_)
  rw [@Measure.restrict_apply α m _ _ _ ht, trim_measurableSet_eq hm ht,
    Measure.restrict_apply (hm t ht),
    trim_measurableSet_eq hm (@MeasurableSet.inter α m t s ht hs)]


instance isFiniteMeasure_trim (hm : m ≤ m0) [IsFiniteMeasure μ] : IsFiniteMeasure (μ.trim hm) where
  measure_univ_lt_top := by
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ⊢ LT.lt ((μ.trim hm) Set.univ) Top.top
    -/
    rw [trim_measurableSet_eq hm (@MeasurableSet.univ _ m)]
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ⊢ LT.lt (μ Set.univ) Top.top
    -/
    exact measure_lt_top _ _
    /-
      🎉 no goals
    -/


theorem sigmaFiniteTrim_mono {m m₂ m0 : MeasurableSpace α} {μ : Measure α} (hm : m ≤ m0)
    (hm₂ : m₂ ≤ m) [SigmaFinite (μ.trim (hm₂.trans hm))] : SigmaFinite (μ.trim hm) := by
  /-
    α : Type u_1
    m m₂ m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hm₂ : LE.le m₂ m
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    ⊢ MeasureTheory.SigmaFinite (μ.trim hm)
  -/
  refine ⟨⟨?_⟩⟩
  refine
    { set := spanningSets (μ.trim (hm₂.trans hm))
      set_mem := fun _ => Set.mem_univ _
      finite := fun i => ?_
      spanning := iUnion_spanningSets _ }
  calc
    (μ.trim hm) (spanningSets (μ.trim (hm₂.trans hm)) i) =
        ((μ.trim hm).trim hm₂) (spanningSets (μ.trim (hm₂.trans hm)) i) := by
      rw [@trim_measurableSet_eq α m₂ m (μ.trim hm) _ hm₂ (measurableSet_spanningSets _ _)]
    _ = (μ.trim (hm₂.trans hm)) (spanningSets (μ.trim (hm₂.trans hm)) i) := by
      rw [@trim_trim _ _ μ _ _ hm₂ hm]
    _ < ∞ := measure_spanningSets_lt_top _ _


theorem sigmaFinite_trim_bot_iff : SigmaFinite (μ.trim bot_le) ↔ IsFiniteMeasure μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Iff (MeasureTheory.SigmaFinite (μ.trim ⋯)) (MeasureTheory.IsFiniteMeasure μ)
  -/
  rw [sigmaFinite_bot_iff]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Iff (MeasureTheory.IsFiniteMeasure (μ.trim ⋯)) (MeasureTheory.IsFiniteMeasur …
  -/
  refine ⟨fun h => ⟨?_⟩, fun h => ⟨?_⟩⟩ <;> have h_univ := h.measure_univ_lt_top
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      h : MeasureTheory.IsFiniteMeasure (μ.trim ⋯)
      h_univ : LT.lt ((μ.trim ⋯) Set.univ) Top.top
      ⊢ LT.lt (μ Set.univ) Top.top
    -/
  · rwa [trim_measurableSet_eq bot_le MeasurableSet.univ] at h_univ
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      h : MeasureTheory.IsFiniteMeasure μ
      h_univ : LT.lt (μ Set.univ) Top.top
      ⊢ LT.lt ((μ.trim ⋯) Set.univ) Top.top
    -/
  · rwa [trim_measurableSet_eq bot_le MeasurableSet.univ]
    /-
      🎉 no goals
    -/


lemma Measure.AbsolutelyContinuous.trim {ν : Measure α} (hμν : μ ≪ ν) (hm : m ≤ m0) :
    μ.trim hm ≪ ν.trim hm := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    hm : LE.le m m0
    ⊢ (μ.trim hm).AbsolutelyContinuous (ν.trim hm)
  -/
  refine Measure.AbsolutelyContinuous.mk (fun s hs hsν ↦ ?_)
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    hm : LE.le m m0
    s : Set α
    hs : MeasurableSet s
    hsν : Eq ((ν.trim hm) s) 0
    ⊢ Eq ((μ.trim hm) s) 0
  -/
  rw [trim_measurableSet_eq hm hs] at hsν ⊢
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    hm : LE.le m m0
    s : Set α
    hs : MeasurableSet s
    hsν : Eq (ν s) 0
    ⊢ Eq (μ s) 0
  -/
  exact hμν hsν
  /-
    🎉 no goals
  -/


