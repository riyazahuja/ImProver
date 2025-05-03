/-- Auxiliary definition for `MeasureTheory.Measure.toFinite`. -/
noncomputable def Measure.toFiniteAux (μ : Measure α) [SFinite μ] : Measure α :=
  letI := Classical.dec
  if IsFiniteMeasure μ then μ else (exists_isFiniteMeasure_absolutelyContinuous μ).choose


/-- A finite measure obtained from an s-finite measure `μ`, such that
`μ = μ.toFinite.withDensity μ.densityToFinite` (see `withDensity_densitytoFinite`).
If `μ` is non-zero, this is a probability measure. -/
noncomputable def Measure.toFinite (μ : Measure α) [SFinite μ] : Measure α :=
  μ.toFiniteAux[|univ]


@[local simp]
lemma ae_toFiniteAux [SFinite μ] : ae μ.toFiniteAux = ae μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq (MeasureTheory.ae μ.toFiniteAux) (MeasureTheory.ae μ)
  -/
  rw [Measure.toFiniteAux]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq (MeasureTheory.ae (ite (MeasureTheory.IsFiniteMeasure μ) μ ⋯.choose)) (Me …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      h✝ : MeasureTheory.IsFiniteMeasure μ
      ⊢ Eq (MeasureTheory.ae μ) (MeasureTheory.ae μ)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      h✝ : Not (MeasureTheory.IsFiniteMeasure μ)
      ⊢ Eq (MeasureTheory.ae ⋯.choose) (MeasureTheory.ae μ)
    -/
  · obtain ⟨_, h₁, h₂⟩ := (exists_isFiniteMeasure_absolutelyContinuous μ).choose_spec
    /-
      case neg.intro.intro
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      h✝ : Not (MeasureTheory.IsFiniteMeasure μ)
      left✝ : MeasureTheory.IsFiniteMeasure ⋯.choose
      h₁ : μ.AbsolutelyContinuous ⋯.choose
      h₂ : ⋯.choose.AbsolutelyContinuous μ
      ⊢ Eq (MeasureTheory.ae ⋯.choose) (MeasureTheory.ae μ)
    -/
    exact h₂.ae_le.antisymm h₁.ae_le
    /-
      🎉 no goals
    -/


@[local instance]
theorem isFiniteMeasure_toFiniteAux [SFinite μ] : IsFiniteMeasure μ.toFiniteAux := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ MeasureTheory.IsFiniteMeasure μ.toFiniteAux
  -/
  rw [Measure.toFiniteAux]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ MeasureTheory.IsFiniteMeasure (ite (MeasureTheory.IsFiniteMeasure μ) μ ⋯.cho …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      h✝ : MeasureTheory.IsFiniteMeasure μ
      ⊢ MeasureTheory.IsFiniteMeasure μ
    -/
  · assumption
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      h✝ : Not (MeasureTheory.IsFiniteMeasure μ)
      ⊢ MeasureTheory.IsFiniteMeasure ⋯.choose
    -/
  · exact (exists_isFiniteMeasure_absolutelyContinuous μ).choose_spec.1
    /-
      🎉 no goals
    -/


@[simp]
lemma ae_toFinite [SFinite μ] : ae μ.toFinite = ae μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq (MeasureTheory.ae μ.toFinite) (MeasureTheory.ae μ)
  -/
  simp [Measure.toFinite, ProbabilityTheory.cond]
  /-
    🎉 no goals
  -/


@[simp]
lemma toFinite_apply_eq_zero_iff [SFinite μ] {s : Set α} : μ.toFinite s = 0 ↔ μ s = 0 := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    ⊢ Iff (Eq (μ.toFinite s) 0) (Eq (μ s) 0)
  -/
  simp only [← compl_mem_ae_iff, ae_toFinite]
  /-
    🎉 no goals
  -/


@[simp]
lemma toFinite_eq_zero_iff [SFinite μ] : μ.toFinite = 0 ↔ μ = 0 := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Iff (Eq μ.toFinite 0) (Eq μ 0)
  -/
  simp_rw [← Measure.measure_univ_eq_zero, toFinite_apply_eq_zero_iff]
  /-
    🎉 no goals
  -/


@[simp]
                                                                 /-
                                                                   α : Type u_1
                                                                   mα : MeasurableSpace α
                                                                   ⊢ Eq (MeasureTheory.Measure.toFinite 0) 0
                                                                 -/
lemma toFinite_zero : Measure.toFinite (0 : Measure α) = 0 := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma toFinite_eq_self [IsProbabilityMeasure μ] : μ.toFinite = μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq μ.toFinite μ
  -/
  rw [Measure.toFinite, Measure.toFiniteAux, if_pos, ProbabilityTheory.cond_univ]
  /-
    case hc
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    ⊢ MeasureTheory.IsFiniteMeasure μ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [SFinite μ] : IsFiniteMeasure μ.toFinite := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ MeasureTheory.IsFiniteMeasure μ.toFinite
  -/
  rw [Measure.toFinite]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ MeasureTheory.IsFiniteMeasure (ProbabilityTheory.cond μ.toFiniteAux Set.univ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [SFinite μ] [NeZero μ] : IsProbabilityMeasure μ.toFinite := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : NeZero μ
    ⊢ MeasureTheory.IsProbabilityMeasure μ.toFinite
  -/
  apply ProbabilityTheory.cond_isProbabilityMeasure
  /-
    case hcs
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : NeZero μ
    ⊢ Ne (μ.toFiniteAux Set.univ) 0
  -/
  simp [ne_eq, ← compl_mem_ae_iff, ae_toFiniteAux]
  /-
    🎉 no goals
  -/


lemma absolutelyContinuous_toFinite (μ : Measure α) [SFinite μ] : μ ≪ μ.toFinite :=
  Measure.ae_le_iff_absolutelyContinuous.mp ae_toFinite.ge


lemma sfiniteSeq_absolutelyContinuous_toFinite (μ : Measure α) [SFinite μ] (n : ℕ) :
    sfiniteSeq μ n ≪ μ.toFinite :=
  (sfiniteSeq_le μ n).absolutelyContinuous.trans (absolutelyContinuous_toFinite μ)


@[deprecated (since := "2024-10-11")]
alias sFiniteSeq_absolutelyContinuous_toFinite := sfiniteSeq_absolutelyContinuous_toFinite


lemma toFinite_absolutelyContinuous (μ : Measure α) [SFinite μ] : μ.toFinite ≪ μ :=
  Measure.ae_le_iff_absolutelyContinuous.mp ae_toFinite.le


/-- A measurable function such that `μ.toFinite.withDensity μ.densityToFinite = μ`.
See `withDensity_densitytoFinite`. -/
@[deprecated rnDeriv (since := "2024-10-04")]
noncomputable def Measure.densityToFinite (μ : Measure α) [SFinite μ] (a : α) : ℝ≥0∞ :=
  μ.rnDeriv μ.toFinite a


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-10-04")]
lemma densityToFinite_def (μ : Measure α) [SFinite μ] :
    μ.densityToFinite = μ.rnDeriv μ.toFinite :=
  rfl


set_option linter.deprecated false in
@[deprecated Measure.measurable_rnDeriv (since := "2024-10-04")]
lemma measurable_densityToFinite (μ : Measure α) [SFinite μ] : Measurable μ.densityToFinite :=
  Measure.measurable_rnDeriv _ _


set_option linter.deprecated false in
@[deprecated Measure.withDensity_rnDeriv_eq (since := "2024-10-04")]
theorem withDensity_densitytoFinite (μ : Measure α) [SFinite μ] :
    μ.toFinite.withDensity μ.densityToFinite = μ :=
  Measure.withDensity_rnDeriv_eq _ _ (absolutelyContinuous_toFinite _)


set_option linter.deprecated false in
@[deprecated Measure.rnDeriv_lt_top (since := "2024-10-04")]
lemma densityToFinite_ae_lt_top (μ : Measure α) [SigmaFinite μ] :
    ∀ᵐ x ∂μ, μ.densityToFinite x < ∞ :=
  (absolutelyContinuous_toFinite μ).ae_le <| Measure.rnDeriv_lt_top _ _


set_option linter.deprecated false in
@[deprecated Measure.rnDeriv_ne_top (since := "2024-10-04")]
lemma densityToFinite_ae_ne_top (μ : Measure α) [SigmaFinite μ] :
    ∀ᵐ x ∂μ, μ.densityToFinite x ≠ ∞ :=
  (densityToFinite_ae_lt_top μ).mono (fun _ hx ↦ hx.ne)


lemma restrict_compl_sigmaFiniteSet [SFinite μ] :
    μ.restrict μ.sigmaFiniteSetᶜ = ∞ • μ.toFinite.restrict μ.sigmaFiniteSetᶜ := by
  rw [Measure.sigmaFiniteSet,
    restrict_compl_sigmaFiniteSetWRT (Measure.AbsolutelyContinuous.refl μ)]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq (HSMul.hSMul Top.top (μ.restrict (HasCompl.compl (μ.sigmaFiniteSetWRT μ)) …
  -/
  ext t ht
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq ((HSMul.hSMul Top.top (μ.restrict (HasCompl.compl (μ.sigmaFiniteSetWRT μ) …
  -/
  simp only [Measure.smul_apply, smul_eq_mul]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq (HMul.hMul Top.top ((μ.restrict (HasCompl.compl (μ.sigmaFiniteSetWRT μ))) …
  -/
  rw [Measure.restrict_apply ht, Measure.restrict_apply ht]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq (HMul.hMul Top.top (μ (Inter.inter t (HasCompl.compl (μ.sigmaFiniteSetWRT …
  -/
  by_cases hμt : μ (t ∩ (μ.sigmaFiniteSetWRT μ)ᶜ) = 0
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      t : Set α
      ht : MeasurableSet t
      hμt : Eq (μ (Inter.inter t (HasCompl.compl (μ.sigmaFiniteSetWRT μ)))) 0
      ⊢ Eq (HMul.hMul Top.top (μ (Inter.inter t (HasCompl.compl (μ.sigmaFiniteSetWRT …
    -/
  · rw [hμt, toFinite_absolutelyContinuous μ hμt]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      t : Set α
      ht : MeasurableSet t
      hμt : Not (Eq (μ (Inter.inter t (HasCompl.compl (μ.sigmaFiniteSetWRT μ)))) 0)
      ⊢ Eq (HMul.hMul Top.top (μ (Inter.inter t (HasCompl.compl (μ.sigmaFiniteSetWRT …
    -/
  · rw [ENNReal.top_mul hμt, ENNReal.top_mul]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      t : Set α
      ht : MeasurableSet t
      hμt : Not (Eq (μ (Inter.inter t (HasCompl.compl (μ.sigmaFiniteSetWRT μ)))) 0)
      ⊢ Ne (μ.toFinite (Inter.inter t (HasCompl.compl (μ.sigmaFiniteSetWRT μ)))) 0
    -/
    exact fun h ↦ hμt (absolutelyContinuous_toFinite μ h)
    /-
      🎉 no goals
    -/


