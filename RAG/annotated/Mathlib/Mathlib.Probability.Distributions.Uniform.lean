/-- A random variable `X` has uniform distribution on `s` if its push-forward measure is
`(μ s)⁻¹ • μ.restrict s`. -/
def IsUniform (X : Ω → E) (s : Set E) (ℙ : Measure Ω) (μ : Measure E := by volume_tac) :=
  map X ℙ = ProbabilityTheory.cond μ s


theorem aemeasurable {X : Ω → E} {s : Set E} (hns : μ s ≠ 0) (hnt : μ s ≠ ∞)
    (hu : IsUniform X s ℙ μ) : AEMeasurable X ℙ := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hns : Ne (μ s) 0
    hnt : Ne (μ s) Top.top
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    ⊢ AEMeasurable X ℙ
  -/
  dsimp [IsUniform, ProbabilityTheory.cond] at hu
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hns : Ne (μ s) 0
    hnt : Ne (μ s) Top.top
    hu : Eq (MeasureTheory.Measure.map X ℙ) (HSMul.hSMul (Inv.inv (μ s)) (μ.restri …
    ⊢ AEMeasurable X ℙ
  -/
  by_contra h
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hns : Ne (μ s) 0
    hnt : Ne (μ s) Top.top
    hu : Eq (MeasureTheory.Measure.map X ℙ) (HSMul.hSMul (Inv.inv (μ s)) (μ.restri …
    h : Not (AEMeasurable X ℙ)
    ⊢ False
  -/
  rw [map_of_not_aemeasurable h] at hu
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hns : Ne (μ s) 0
    hnt : Ne (μ s) Top.top
    hu : Eq 0 (HSMul.hSMul (Inv.inv (μ s)) (μ.restrict s))
    h : Not (AEMeasurable X ℙ)
    ⊢ False
  -/
  apply zero_ne_one' ℝ≥0∞
  calc
    0 = (0 : Measure E) Set.univ := rfl
    _ = _ := by rw [hu, smul_apply, restrict_apply MeasurableSet.univ,
      Set.univ_inter, smul_eq_mul, ENNReal.inv_mul_cancel hns hnt]


theorem absolutelyContinuous {X : Ω → E} {s : Set E} (hu : IsUniform X s ℙ μ) : map X ℙ ≪ μ := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    ⊢ (MeasureTheory.Measure.map X ℙ).AbsolutelyContinuous μ
  -/
  rw [hu]; exact ProbabilityTheory.cond_absolutelyContinuous
           /-
             🎉 no goals
           -/


theorem measure_preimage {X : Ω → E} {s : Set E} (hns : μ s ≠ 0) (hnt : μ s ≠ ∞)
    (hu : IsUniform X s ℙ μ) {A : Set E} (hA : MeasurableSet A) :
    ℙ (X ⁻¹' A) = μ (s ∩ A) / μ s := by
  rwa [← map_apply_of_aemeasurable (hu.aemeasurable hns hnt) hA, hu, ProbabilityTheory.cond_apply',
    ENNReal.div_eq_inv_mul]


theorem isProbabilityMeasure {X : Ω → E} {s : Set E} (hns : μ s ≠ 0) (hnt : μ s ≠ ∞)
    (hu : IsUniform X s ℙ μ) : IsProbabilityMeasure ℙ :=
  ⟨by
    /-
      E : Type u_1
      inst✝ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → E
      s : Set E
      hns : Ne (μ s) 0
      hnt : Ne (μ s) Top.top
      hu : MeasureTheory.pdf.IsUniform X s ℙ μ
      ⊢ Eq (ℙ Set.univ) 1
    -/
    have : X ⁻¹' Set.univ = Set.univ := Set.preimage_univ
    rw [← this, hu.measure_preimage hns hnt MeasurableSet.univ, Set.inter_univ,
      ENNReal.div_self hns hnt]⟩


theorem toMeasurable_iff {X : Ω → E} {s : Set E} :
    IsUniform X (toMeasurable μ s) ℙ μ ↔ IsUniform X s ℙ μ := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    ⊢ Iff (MeasureTheory.pdf.IsUniform X (MeasureTheory.toMeasurable μ s) ℙ μ) (Me …
  -/
  unfold IsUniform
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    ⊢ Iff (Eq (MeasureTheory.Measure.map X ℙ) (ProbabilityTheory.cond μ (MeasureTh …
  -/
  rw [ProbabilityTheory.cond_toMeasurable_eq]
  /-
    🎉 no goals
  -/


protected theorem toMeasurable {X : Ω → E} {s : Set E} (hu : IsUniform X s ℙ μ) :
    IsUniform X (toMeasurable μ s) ℙ μ := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    ⊢ MeasureTheory.pdf.IsUniform X (MeasureTheory.toMeasurable μ s) ℙ μ
  -/
  unfold IsUniform at *
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hu : Eq (MeasureTheory.Measure.map X ℙ) (ProbabilityTheory.cond μ s)
    ⊢ Eq (MeasureTheory.Measure.map X ℙ) (ProbabilityTheory.cond μ (MeasureTheory. …
  -/
  rwa [ProbabilityTheory.cond_toMeasurable_eq]
  /-
    🎉 no goals
  -/


theorem hasPDF {X : Ω → E} {s : Set E} (hns : μ s ≠ 0) (hnt : μ s ≠ ∞)
    (hu : IsUniform X s ℙ μ) : HasPDF X ℙ μ := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hns : Ne (μ s) 0
    hnt : Ne (μ s) Top.top
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    ⊢ MeasureTheory.HasPDF X ℙ μ
  -/
  let t := toMeasurable μ s
  apply hasPDF_of_map_eq_withDensity (hu.aemeasurable hns hnt) (t.indicator ((μ t)⁻¹ • 1)) <|
    (measurable_one.aemeasurable.const_smul (μ t)⁻¹).indicator (measurableSet_toMeasurable μ s)
  rw [hu, withDensity_indicator (measurableSet_toMeasurable μ s), withDensity_smul _ measurable_one,
    withDensity_one, restrict_toMeasurable hnt, measure_toMeasurable, ProbabilityTheory.cond]


theorem pdf_eq_zero_of_measure_eq_zero_or_top {X : Ω → E} {s : Set E}
    (hu : IsUniform X s ℙ μ) (hμs : μ s = 0 ∨ μ s = ∞) : pdf X ℙ μ =ᵐ[μ] 0 := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    hμs : Or (Eq (μ s) 0) (Eq (μ s) Top.top)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0
  -/
  rcases hμs with H|H
  · simp only [IsUniform, ProbabilityTheory.cond, H, ENNReal.inv_zero, restrict_eq_zero.mpr H,
    smul_zero] at hu
    /-
      case inl
      E : Type u_1
      inst✝ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → E
      s : Set E
      H : Eq (μ s) 0
      hu : Eq (MeasureTheory.Measure.map X ℙ) 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0
    -/
    simp [pdf, hu]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → E
      s : Set E
      hu : MeasureTheory.pdf.IsUniform X s ℙ μ
      H : Eq (μ s) Top.top
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0
    -/
  · simp only [IsUniform, ProbabilityTheory.cond, H, ENNReal.inv_top, zero_smul] at hu
    /-
      case inr
      E : Type u_1
      inst✝ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → E
      s : Set E
      H : Eq (μ s) Top.top
      hu : Eq (MeasureTheory.Measure.map X ℙ) 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0
    -/
    simp [pdf, hu]
    /-
      🎉 no goals
    -/


theorem pdf_eq {X : Ω → E} {s : Set E} (hms : MeasurableSet s)
    (hu : IsUniform X s ℙ μ) : pdf X ℙ μ =ᵐ[μ] s.indicator ((μ s)⁻¹ • (1 : E → ℝ≥0∞)) := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hms : MeasurableSet s
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) (s.indicator (HS …
  -/
  by_cases hnt : μ s = ∞
    /-
      case pos
      E : Type u_1
      inst✝ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → E
      s : Set E
      hms : MeasurableSet s
      hu : MeasureTheory.pdf.IsUniform X s ℙ μ
      hnt : Eq (μ s) Top.top
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) (s.indicator (HS …
    -/
  · simp [pdf_eq_zero_of_measure_eq_zero_or_top hu (Or.inr hnt), hnt]
    /-
      🎉 no goals
    -/
  /-
    case neg
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hms : MeasurableSet s
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    hnt : Not (Eq (μ s) Top.top)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) (s.indicator (HS …
  -/
  by_cases hns : μ s = 0
  · filter_upwards [measure_zero_iff_ae_nmem.mp hns,
      pdf_eq_zero_of_measure_eq_zero_or_top hu (Or.inl hns)] with x hx h'x
    /-
      case h
      E : Type u_1
      inst✝ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → E
      s : Set E
      hms : MeasurableSet s
      hu : MeasureTheory.pdf.IsUniform X s ℙ μ
      hnt : Not (Eq (μ s) Top.top)
      hns : Eq (μ s) 0
      x : E
      hx : Not (Membership.mem s x)
      h'x : Eq (MeasureTheory.pdf X ℙ μ x) (0 x)
      ⊢ Eq (MeasureTheory.pdf X ℙ μ x) (s.indicator (HSMul.hSMul (Inv.inv (μ s)) 1) x)
    -/
    simp [hx, h'x, hns]
    /-
      🎉 no goals
    -/
  /-
    case neg
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hms : MeasurableSet s
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    hnt : Not (Eq (μ s) Top.top)
    hns : Not (Eq (μ s) 0)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) (s.indicator (HS …
  -/
  have : HasPDF X ℙ μ := hasPDF hns hnt hu
  /-
    case neg
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hms : MeasurableSet s
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    hnt : Not (Eq (μ s) Top.top)
    hns : Not (Eq (μ s) 0)
    this : MeasureTheory.HasPDF X ℙ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) (s.indicator (HS …
  -/
  have : IsProbabilityMeasure ℙ := isProbabilityMeasure hns hnt hu
  /-
    case neg
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hms : MeasurableSet s
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    hnt : Not (Eq (μ s) Top.top)
    hns : Not (Eq (μ s) 0)
    this✝ : MeasureTheory.HasPDF X ℙ μ
    this : MeasureTheory.IsProbabilityMeasure ℙ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) (s.indicator (HS …
  -/
  apply (eq_of_map_eq_withDensity _ _).mp
  · rw [hu, withDensity_indicator hms, withDensity_smul _ measurable_one, withDensity_one,
      ProbabilityTheory.cond]
    /-
      E : Type u_1
      inst✝ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → E
      s : Set E
      hms : MeasurableSet s
      hu : MeasureTheory.pdf.IsUniform X s ℙ μ
      hnt : Not (Eq (μ s) Top.top)
      hns : Not (Eq (μ s) 0)
      this✝ : MeasureTheory.HasPDF X ℙ μ
      this : MeasureTheory.IsProbabilityMeasure ℙ
      ⊢ AEMeasurable (s.indicator (HSMul.hSMul (Inv.inv (μ s)) 1)) μ
    -/
  · exact (measurable_one.aemeasurable.const_smul (μ s)⁻¹).indicator hms
    /-
      🎉 no goals
    -/


theorem pdf_toReal_ae_eq {X : Ω → E} {s : Set E} (hms : MeasurableSet s)
    (hX : IsUniform X s ℙ μ) :
    (fun x => (pdf X ℙ μ x).toReal) =ᵐ[μ] fun x =>
      (s.indicator ((μ s)⁻¹ • (1 : E → ℝ≥0∞)) x).toReal :=
  Filter.EventuallyEq.fun_comp (pdf_eq hms hX) ENNReal.toReal


                                                      /-
                                                        E : Type u_1
                                                        inst✝ : MeasurableSpace E
                                                        μ : MeasureTheory.Measure E
                                                        Ω : Type u_2
                                                        x✝ : MeasurableSpace Ω
                                                        ℙ : MeasureTheory.Measure Ω
                                                        X : Ω → Real
                                                        s : Set Real
                                                        hcs : IsCompact s
                                                        ⊢ MeasureTheory.Measure Real
                                                      -/
theorem mul_pdf_integrable (hcs : IsCompact s) (huX : IsUniform X s ℙ) :
                                                      /-
                                                        🎉 no goals
                                                      -/
    /-
      E : Type u_1
      inst✝ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → Real
      s : Set Real
      hcs : IsCompact s
      huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
      ⊢ MeasureTheory.Measure Real
    -/
    Integrable fun x : ℝ => x * (pdf X ℙ volume x).toReal := by
    /-
      🎉 no goals
    -/
  /-
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    hcs : IsCompact s
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul x (MeasureTheory.pdf X ℙ Measur …
  -/
  by_cases hnt : volume s = 0 ∨ volume s = ∞
    /-
      case pos
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → Real
      s : Set Real
      hcs : IsCompact s
      huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
      hnt : Or (Eq (MeasureTheory.MeasureSpace.volume s) 0) (Eq (MeasureTheory.Measu …
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul x (MeasureTheory.pdf X ℙ Measur …
    -/
  · have I : Integrable (fun x ↦ x * ENNReal.toReal (0)) := by simp
    /-
      case pos
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → Real
      s : Set Real
      hcs : IsCompact s
      huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
      hnt : Or (Eq (MeasureTheory.MeasureSpace.volume s) 0) (Eq (MeasureTheory.Measu …
      I : MeasureTheory.Integrable (fun x => HMul.hMul x (ENNReal.toReal 0)) Measure …
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul x (MeasureTheory.pdf X ℙ Measur …
    -/
    apply I.congr
    /-
      case pos
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → Real
      s : Set Real
      hcs : IsCompact s
      huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
      hnt : Or (Eq (MeasureTheory.MeasureSpace.volume s) 0) (Eq (MeasureTheory.Measu …
      I : MeasureTheory.Integrable (fun x => HMul.hMul x (ENNReal.toReal 0)) Measure …
      ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (fun x =>  …
    -/
    filter_upwards [pdf_eq_zero_of_measure_eq_zero_or_top huX hnt] with x hx
    /-
      case h
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → Real
      s : Set Real
      hcs : IsCompact s
      huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
      hnt : Or (Eq (MeasureTheory.MeasureSpace.volume s) 0) (Eq (MeasureTheory.Measu …
      I : MeasureTheory.Integrable (fun x => HMul.hMul x (ENNReal.toReal 0)) Measure …
      x : Real
      hx : Eq (MeasureTheory.pdf X ℙ MeasureTheory.MeasureSpace.volume x) (0 x)
      ⊢ Eq (HMul.hMul x (ENNReal.toReal 0)) (HMul.hMul x (MeasureTheory.pdf X ℙ Meas …
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    hcs : IsCompact s
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    hnt : Not (Or (Eq (MeasureTheory.MeasureSpace.volume s) 0) (Eq (MeasureTheory. …
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul x (MeasureTheory.pdf X ℙ Measur …
  -/
  simp only [not_or] at hnt
  /-
    case neg
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    hcs : IsCompact s
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    hnt : And (Not (Eq (MeasureTheory.MeasureSpace.volume s) 0)) (Not (Eq (Measure …
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul x (MeasureTheory.pdf X ℙ Measur …
  -/
  have : IsProbabilityMeasure ℙ := isProbabilityMeasure hnt.1 hnt.2 huX
  /-
    case neg
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    hcs : IsCompact s
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    hnt : And (Not (Eq (MeasureTheory.MeasureSpace.volume s) 0)) (Not (Eq (Measure …
    this : MeasureTheory.IsProbabilityMeasure ℙ
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul x (MeasureTheory.pdf X ℙ Measur …
  -/
  constructor
  · exact aestronglyMeasurable_id.mul
      (measurable_pdf X ℙ).aemeasurable.ennreal_toReal.aestronglyMeasurable
  /-
    case neg.right
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    hcs : IsCompact s
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    hnt : And (Not (Eq (MeasureTheory.MeasureSpace.volume s) 0)) (Not (Eq (Measure …
    this : MeasureTheory.IsProbabilityMeasure ℙ
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => HMul.hMul x (MeasureTheory.pdf X ℙ …
  -/
  refine hasFiniteIntegral_mul (pdf_eq hcs.measurableSet huX) ?_
  /-
    case neg.right
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    hcs : IsCompact s
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    hnt : And (Not (Eq (MeasureTheory.MeasureSpace.volume s) 0)) (Not (Eq (Measure …
    this : MeasureTheory.IsProbabilityMeasure ℙ
    ⊢ Ne (MeasureTheory.lintegral MeasureTheory.MeasureSpace.volume fun x => HMul. …
  -/
  set ind := (volume s)⁻¹ • (1 : ℝ → ℝ≥0∞)
  have : ∀ x, ↑‖x‖₊ * s.indicator ind x = s.indicator (fun x => ‖x‖₊ * ind x) x := fun x =>
    (s.indicator_mul_right (fun x => ↑‖x‖₊) ind).symm
  simp only [ind, this, lintegral_indicator hcs.measurableSet, mul_one, Algebra.id.smul_eq_mul,
    Pi.one_apply, Pi.smul_apply]
  /-
    case neg.right
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    hcs : IsCompact s
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    hnt : And (Not (Eq (MeasureTheory.MeasureSpace.volume s) 0)) (Not (Eq (Measure …
    this✝ : MeasureTheory.IsProbabilityMeasure ℙ
    ind : Real → ENNReal := HSMul.hSMul (Inv.inv (MeasureTheory.MeasureSpace.volum …
    this : ∀ (x : Real), Eq (HMul.hMul (↑(NNNorm.nnnorm x)) (s.indicator ind x)) ( …
    ⊢ Ne (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict s) f …
  -/
  rw [lintegral_mul_const _ measurable_nnnorm.coe_nnreal_ennreal]
  exact ENNReal.mul_ne_top (setLIntegral_lt_top_of_isCompact hnt.2 hcs continuous_nnnorm).ne
    (ENNReal.inv_lt_top.2 (pos_iff_ne_zero.mpr hnt.1)).ne


/-- A real uniform random variable `X` with support `s` has expectation
`(λ s)⁻¹ * ∫ x in s, x ∂λ` where `λ` is the Lebesgue measure. -/
                           /-
                             E : Type u_1
                             inst✝ : MeasurableSpace E
                             μ : MeasureTheory.Measure E
                             Ω : Type u_2
                             x✝ : MeasurableSpace Ω
                             ℙ : MeasureTheory.Measure Ω
                             X : Ω → Real
                             s : Set Real
                             ⊢ MeasureTheory.Measure Real
                           -/
theorem integral_eq (huX : IsUniform X s ℙ) :
                           /-
                             🎉 no goals
                           -/
    ∫ x, X x ∂ℙ = (volume s)⁻¹.toReal * ∫ x in s, x := by
  /-
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    ⊢ Eq (MeasureTheory.integral ℙ fun x => X x) (HMul.hMul (Inv.inv (MeasureTheor …
  -/
  rw [← smul_eq_mul, ← integral_smul_measure]
  /-
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    huX : MeasureTheory.pdf.IsUniform X s ℙ MeasureTheory.MeasureSpace.volume
    ⊢ Eq (MeasureTheory.integral ℙ fun x => X x) (MeasureTheory.integral (HSMul.hS …
  -/
  dsimp only [IsUniform, ProbabilityTheory.cond] at huX
  /-
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    huX : Eq (MeasureTheory.Measure.map X ℙ) (HSMul.hSMul (Inv.inv (MeasureTheory. …
    ⊢ Eq (MeasureTheory.integral ℙ fun x => X x) (MeasureTheory.integral (HSMul.hS …
  -/
  rw [← huX]
  /-
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    s : Set Real
    huX : Eq (MeasureTheory.Measure.map X ℙ) (HSMul.hSMul (Inv.inv (MeasureTheory. …
    ⊢ Eq (MeasureTheory.integral ℙ fun x => X x) (MeasureTheory.integral (MeasureT …
  -/
  by_cases hX : AEMeasurable X ℙ
    /-
      case pos
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → Real
      s : Set Real
      huX : Eq (MeasureTheory.Measure.map X ℙ) (HSMul.hSMul (Inv.inv (MeasureTheory. …
      hX : AEMeasurable X ℙ
      ⊢ Eq (MeasureTheory.integral ℙ fun x => X x) (MeasureTheory.integral (MeasureT …
    -/
  · exact (integral_map hX aestronglyMeasurable_id).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → Real
      s : Set Real
      huX : Eq (MeasureTheory.Measure.map X ℙ) (HSMul.hSMul (Inv.inv (MeasureTheory. …
      hX : Not (AEMeasurable X ℙ)
      ⊢ Eq (MeasureTheory.integral ℙ fun x => X x) (MeasureTheory.integral (MeasureT …
    -/
  · rw [map_of_not_aemeasurable hX, integral_zero_measure, integral_non_aestronglyMeasurable]
    /-
      case neg
      Ω : Type u_2
      x✝ : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      X : Ω → Real
      s : Set Real
      huX : Eq (MeasureTheory.Measure.map X ℙ) (HSMul.hSMul (Inv.inv (MeasureTheory. …
      hX : Not (AEMeasurable X ℙ)
      ⊢ Not (MeasureTheory.AEStronglyMeasurable X ℙ)
    -/
    rwa [aestronglyMeasurable_iff_aemeasurable]
    /-
      🎉 no goals
    -/


lemma IsUniform.cond {s : Set E} :
    IsUniform (id : E → E) s (ProbabilityTheory.cond μ s) μ := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    s : Set E
    ⊢ MeasureTheory.pdf.IsUniform id s (ProbabilityTheory.cond μ s) μ
  -/
  unfold IsUniform
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    s : Set E
    ⊢ Eq (MeasureTheory.Measure.map id (ProbabilityTheory.cond μ s)) (ProbabilityT …
  -/
  rw [Measure.map_id]
  /-
    🎉 no goals
  -/


/-- The density of the uniform measure on a set with respect to itself. This allows us to abstract
away the choice of random variable and probability space. -/
def uniformPDF (s : Set E) (x : E) (μ : Measure E := by volume_tac) : ℝ≥0∞ :=
  s.indicator ((μ s)⁻¹ • (1 : E → ℝ≥0∞)) x


/-- Check that indeed any uniform random variable has the uniformPDF. -/
lemma uniformPDF_eq_pdf {s : Set E} (hs : MeasurableSet s) (hu : pdf.IsUniform X s ℙ μ) :
    (fun x ↦ uniformPDF s x μ) =ᵐ[μ] pdf X ℙ μ := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hs : MeasurableSet s
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => MeasureTheory.pdf.uniformPDF s x …
  -/
  unfold uniformPDF
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    Ω : Type u_2
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → E
    s : Set E
    hs : MeasurableSet s
    hu : MeasureTheory.pdf.IsUniform X s ℙ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => s.indicator (HSMul.hSMul (Inv.in …
  -/
  exact Filter.EventuallyEq.trans (pdf.IsUniform.pdf_eq hs hu).symm (ae_eq_refl _)
  /-
    🎉 no goals
  -/


/-- Alternative way of writing the uniformPDF. -/
lemma uniformPDF_ite {s : Set E} {x : E} :
    uniformPDF s x μ = if x ∈ s then (μ s)⁻¹ else 0 := by
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    s : Set E
    x : E
    ⊢ Eq (MeasureTheory.pdf.uniformPDF s x μ) (ite (Membership.mem s x) (Inv.inv ( …
  -/
  unfold uniformPDF
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    s : Set E
    x : E
    ⊢ Eq (s.indicator (HSMul.hSMul (Inv.inv (μ s)) 1) x) (ite (Membership.mem s x) …
  -/
  unfold Set.indicator
  /-
    E : Type u_1
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    s : Set E
    x : E
    ⊢ Eq (ite (Membership.mem s x) (HSMul.hSMul (Inv.inv (μ s)) 1 x) 0) (ite (Memb …
  -/
  simp only [Pi.smul_apply, Pi.one_apply, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


/-- Uniform distribution taking the same non-zero probability on the nonempty finset `s` -/
def uniformOfFinset (s : Finset α) (hs : s.Nonempty) : PMF α := by
  /-
    α : Type u_1
    s : Finset α
    hs : s.Nonempty
    ⊢ PMF α
  -/
  refine ofFinset (fun a => if a ∈ s then s.card⁻¹ else 0) s ?_ ?_
    /-
      case refine_1
      α : Type u_1
      s : Finset α
      hs : s.Nonempty
      ⊢ Eq (s.sum fun a => (fun a => ite (Membership.mem s a) (Inv.inv ↑s.card) 0) a …
    -/
  · simp only [Finset.sum_ite_mem, Finset.inter_self, Finset.sum_const, nsmul_eq_mul]
    have : (s.card : ℝ≥0∞) ≠ 0 := by
      simpa only [Ne, Nat.cast_eq_zero, Finset.card_eq_zero] using
        Finset.nonempty_iff_ne_empty.1 hs
    /-
      case refine_1
      α : Type u_1
      s : Finset α
      hs : s.Nonempty
      this : Ne (↑s.card) 0
      ⊢ Eq (HMul.hMul (↑s.card) (Inv.inv ↑s.card)) 1
    -/
    exact ENNReal.mul_inv_cancel this <| ENNReal.natCast_ne_top s.card
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      s : Finset α
      hs : s.Nonempty
      ⊢ ∀ (a : α), Not (Membership.mem s a) → Eq ((fun a => ite (Membership.mem s a) …
    -/
  · exact fun x hx => by simp only [hx, if_false]
    /-
      🎉 no goals
    -/


@[simp]
theorem uniformOfFinset_apply (a : α) :
    uniformOfFinset s hs a = if a ∈ s then (s.card : ℝ≥0∞)⁻¹ else 0 :=
  rfl


theorem uniformOfFinset_apply_of_mem (ha : a ∈ s) : uniformOfFinset s hs a = (s.card : ℝ≥0∞)⁻¹ := by
  /-
    α : Type u_1
    s : Finset α
    hs : s.Nonempty
    a : α
    ha : Membership.mem s a
    ⊢ Eq ((PMF.uniformOfFinset s hs) a) (Inv.inv ↑s.card)
  -/
  simp [ha]
  /-
    🎉 no goals
  -/


                                                                                         /-
                                                                                           α : Type u_1
                                                                                           s : Finset α
                                                                                           hs : s.Nonempty
                                                                                           a : α
                                                                                           ha : Not (Membership.mem s a)
                                                                                           ⊢ Eq ((PMF.uniformOfFinset s hs) a) 0
                                                                                         -/
theorem uniformOfFinset_apply_of_not_mem (ha : a ∉ s) : uniformOfFinset s hs a = 0 := by simp [ha]
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem support_uniformOfFinset : (uniformOfFinset s hs).support = s :=
  Set.ext
    (by
      /-
        α : Type u_1
        s : Finset α
        hs : s.Nonempty
        ⊢ ∀ (x : α), Iff (Membership.mem (PMF.uniformOfFinset s hs).support x) (Member …
      -/
      let ⟨a, ha⟩ := hs
      /-
        α : Type u_1
        s : Finset α
        hs : s.Nonempty
        a : α
        ha : Membership.mem s a
        ⊢ ∀ (x : α), Iff (Membership.mem (PMF.uniformOfFinset s ⋯).support x) (Members …
      -/
      simp [mem_support_iff, Finset.ne_empty_of_mem ha])
      /-
        🎉 no goals
      -/


theorem mem_support_uniformOfFinset_iff (a : α) : a ∈ (uniformOfFinset s hs).support ↔ a ∈ s := by
  /-
    α : Type u_1
    s : Finset α
    hs : s.Nonempty
    a : α
    ⊢ Iff (Membership.mem (PMF.uniformOfFinset s hs).support a) (Membership.mem s a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toOuterMeasure_uniformOfFinset_apply :
    (uniformOfFinset s hs).toOuterMeasure t = (s.filter (· ∈ t)).card / s.card :=
  calc
    (uniformOfFinset s hs).toOuterMeasure t = ∑' x, if x ∈ t then uniformOfFinset s hs x else 0 :=
      toOuterMeasure_apply (uniformOfFinset s hs) t
    _ = ∑' x, if x ∈ s ∧ x ∈ t then (s.card : ℝ≥0∞)⁻¹ else 0 :=
                              /-
                                α : Type u_1
                                s : Finset α
                                hs : s.Nonempty
                                t : Set α
                                x : α
                                ⊢ Eq (ite (Membership.mem t x) ((PMF.uniformOfFinset s hs) x) 0) (ite (And (Me …
                              -/
      (tsum_congr fun x => by simp_rw [uniformOfFinset_apply, ← ite_and, and_comm])
                              /-
                                🎉 no goals
                              -/
    _ = ∑ x ∈ s.filter (· ∈ t), if x ∈ s ∧ x ∈ t then (s.card : ℝ≥0∞)⁻¹ else 0 :=
      (tsum_eq_sum fun _ hx => if_neg fun h => hx (Finset.mem_filter.2 h))
    _ = ∑ _x ∈ s.filter (· ∈ t), (s.card : ℝ≥0∞)⁻¹ :=
      (Finset.sum_congr rfl fun x hx => by
        /-
          α : Type u_1
          s : Finset α
          hs : s.Nonempty
          t : Set α
          x : α
          hx : Membership.mem (Finset.filter (fun x => Membership.mem t x) s) x
          ⊢ Eq (ite (And (Membership.mem s x) (Membership.mem t x)) (Inv.inv ↑s.card) 0) …
        -/
        let this : x ∈ s ∧ x ∈ t := by simpa using hx
        /-
          α : Type u_1
          s : Finset α
          hs : s.Nonempty
          t : Set α
          x : α
          hx : Membership.mem (Finset.filter (fun x => Membership.mem t x) s) x
          this : And (Membership.mem s x) (Membership.mem t x) := Eq.mp Mathlib.Data.Fin …
          ⊢ Eq (ite (And (Membership.mem s x) (Membership.mem t x)) (Inv.inv ↑s.card) 0) …
        -/
        simp only [this, and_self_iff, if_true])
        /-
          🎉 no goals
        -/
    _ = (s.filter (· ∈ t)).card / s.card := by
        /-
          α : Type u_1
          s : Finset α
          hs : s.Nonempty
          t : Set α
          ⊢ Eq ((Finset.filter (fun x => Membership.mem t x) s).sum fun _x => Inv.inv ↑s …
        -/
        simp only [div_eq_mul_inv, Finset.sum_const, nsmul_eq_mul]
        /-
          🎉 no goals
        -/


@[simp]
theorem toMeasure_uniformOfFinset_apply [MeasurableSpace α] (ht : MeasurableSet t) :
    (uniformOfFinset s hs).toMeasure t = (s.filter (· ∈ t)).card / s.card :=
  (toMeasure_apply_eq_toOuterMeasure_apply _ t ht).trans (toOuterMeasure_uniformOfFinset_apply hs t)


/-- The uniform pmf taking the same uniform value on all of the fintype `α` -/
def uniformOfFintype (α : Type*) [Fintype α] [Nonempty α] : PMF α :=
  uniformOfFinset Finset.univ Finset.univ_nonempty


@[simp]
theorem uniformOfFintype_apply (a : α) : uniformOfFintype α a = (Fintype.card α : ℝ≥0∞)⁻¹ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Nonempty α
    a : α
    ⊢ Eq ((PMF.uniformOfFintype α) a) (Inv.inv ↑(Fintype.card α))
  -/
  simp [uniformOfFintype, Finset.mem_univ, if_true, uniformOfFinset_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_uniformOfFintype (α : Type*) [Fintype α] [Nonempty α] :
    (uniformOfFintype α).support = ⊤ :=
                      /-
                        α : Type u_2
                        inst✝¹ : Fintype α
                        inst✝ : Nonempty α
                        x : α
                        ⊢ Iff (Membership.mem (PMF.uniformOfFintype α).support x) (Membership.mem Top. …
                      -/
  Set.ext fun x => by simp [mem_support_iff]
                      /-
                        🎉 no goals
                      -/


                                                                                      /-
                                                                                        α : Type u_1
                                                                                        inst✝¹ : Fintype α
                                                                                        inst✝ : Nonempty α
                                                                                        a : α
                                                                                        ⊢ Membership.mem (PMF.uniformOfFintype α).support a
                                                                                      -/
theorem mem_support_uniformOfFintype (a : α) : a ∈ (uniformOfFintype α).support := by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem toOuterMeasure_uniformOfFintype_apply :
    (uniformOfFintype α).toOuterMeasure s = Fintype.card s / Fintype.card α := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Nonempty α
    s : Set α
    ⊢ Eq ((PMF.uniformOfFintype α).toOuterMeasure s) (HDiv.hDiv ↑(Fintype.card ↑s) …
  -/
  rw [uniformOfFintype, toOuterMeasure_uniformOfFinset_apply,Fintype.card_ofFinset]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Nonempty α
    s : Set α
    ⊢ Eq (HDiv.hDiv ↑(Finset.filter (fun x => Membership.mem s x) Finset.univ).car …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toMeasure_uniformOfFintype_apply [MeasurableSpace α] (hs : MeasurableSet s) :
    (uniformOfFintype α).toMeasure s = Fintype.card s / Fintype.card α := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : Nonempty α
    s : Set α
    inst✝ : MeasurableSpace α
    hs : MeasurableSet s
    ⊢ Eq ((PMF.uniformOfFintype α).toMeasure s) (HDiv.hDiv ↑(Fintype.card ↑s) ↑(Fi …
  -/
  simp [uniformOfFintype, hs]
  /-
    🎉 no goals
  -/


/-- Given a non-empty multiset `s` we construct the `PMF` which sends `a` to the fraction of
  elements in `s` that are `a`. -/
def ofMultiset (s : Multiset α) (hs : s ≠ 0) : PMF α :=
  ⟨fun a => s.count a / (Multiset.card s),
    ENNReal.summable.hasSum_iff.2
      (calc
        (∑' b : α, (s.count b : ℝ≥0∞) / (Multiset.card s))
          = (Multiset.card s : ℝ≥0∞)⁻¹ * ∑' b, (s.count b : ℝ≥0∞) := by
            /-
              α : Type u_1
              s : Multiset α
              hs : Ne s 0
              ⊢ Eq (tsum fun b => HDiv.hDiv ↑(Multiset.count b s) ↑s.card) (HMul.hMul (Inv.i …
            -/
            simp_rw [ENNReal.div_eq_inv_mul, ENNReal.tsum_mul_left]
            /-
              🎉 no goals
            -/
        _ = (Multiset.card s : ℝ≥0∞)⁻¹ * ∑ b ∈ s.toFinset, (s.count b : ℝ≥0∞) :=
          (congr_arg (fun x => (Multiset.card s : ℝ≥0∞)⁻¹ * x)
            (tsum_eq_sum fun a ha =>
                                       /-
                                         α : Type u_1
                                         s : Multiset α
                                         hs : Ne s 0
                                         a : α
                                         ha : Not (Membership.mem s.toFinset a)
                                         ⊢ Eq (Multiset.count a s) 0
                                       -/
              Nat.cast_eq_zero.2 <| by rwa [Multiset.count_eq_zero, ← Multiset.mem_toFinset]))
                                       /-
                                         🎉 no goals
                                       -/
        _ = 1 := by
          rw [← Nat.cast_sum, Multiset.toFinset_sum_count_eq s,
            ENNReal.inv_mul_cancel (Nat.cast_ne_zero.2 (hs ∘ Multiset.card_eq_zero.1))
              (ENNReal.natCast_ne_top _)]
        )⟩


@[simp]
theorem ofMultiset_apply (a : α) : ofMultiset s hs a = s.count a / (Multiset.card s) :=
  rfl


@[simp]
theorem support_ofMultiset : (ofMultiset s hs).support = s.toFinset :=
              /-
                α : Type u_1
                s : Multiset α
                hs : Ne s 0
                ⊢ ∀ (x : α), Iff (Membership.mem (PMF.ofMultiset s hs).support x) (Membership. …
              -/
  Set.ext (by simp [mem_support_iff, hs])
              /-
                🎉 no goals
              -/


theorem mem_support_ofMultiset_iff (a : α) : a ∈ (ofMultiset s hs).support ↔ a ∈ s.toFinset := by
  /-
    α : Type u_1
    s : Multiset α
    hs : Ne s 0
    a : α
    ⊢ Iff (Membership.mem (PMF.ofMultiset s hs).support a) (Membership.mem s.toFin …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ofMultiset_apply_of_not_mem {a : α} (ha : a ∉ s) : ofMultiset s hs a = 0 := by
  simpa only [ofMultiset_apply, ENNReal.div_eq_zero_iff, Nat.cast_eq_zero, Multiset.count_eq_zero,
    ENNReal.natCast_ne_top, or_false] using ha


@[simp]
theorem toOuterMeasure_ofMultiset_apply :
    (ofMultiset s hs).toOuterMeasure t =
      (∑' x, (s.filter (· ∈ t)).count x : ℝ≥0∞) / (Multiset.card s) := by
  /-
    α : Type u_1
    s : Multiset α
    hs : Ne s 0
    t : Set α
    ⊢ Eq ((PMF.ofMultiset s hs).toOuterMeasure t) (HDiv.hDiv (tsum fun x => ↑(Mult …
  -/
  simp_rw [div_eq_mul_inv, ← ENNReal.tsum_mul_right, toOuterMeasure_apply]
  /-
    α : Type u_1
    s : Multiset α
    hs : Ne s 0
    t : Set α
    ⊢ Eq (tsum fun x => t.indicator (⇑(PMF.ofMultiset s hs)) x) (tsum fun i => HMu …
  -/
  refine tsum_congr fun x => ?_
  /-
    α : Type u_1
    s : Multiset α
    hs : Ne s 0
    t : Set α
    x : α
    ⊢ Eq (t.indicator (⇑(PMF.ofMultiset s hs)) x) (HMul.hMul (↑(Multiset.count x ( …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x ∈ t <;> simp [Set.indicator, hx, div_eq_mul_inv]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem toMeasure_ofMultiset_apply [MeasurableSpace α] (ht : MeasurableSet t) :
    (ofMultiset s hs).toMeasure t = (∑' x, (s.filter (· ∈ t)).count x : ℝ≥0∞) / (Multiset.card s) :=
  (toMeasure_apply_eq_toOuterMeasure_apply _ t ht).trans (toOuterMeasure_ofMultiset_apply hs t)


