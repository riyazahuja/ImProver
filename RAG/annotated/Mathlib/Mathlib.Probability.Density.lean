/-- A random variable `X : Ω → E` is said to have a probability density function (`HasPDF`)
with respect to the measure `ℙ` on `Ω` and `μ` on `E`
if the push-forward measure of `ℙ` along `X` is absolutely continuous with respect to `μ`
and they have a Lebesgue decomposition (`HaveLebesgueDecomposition`). -/
class HasPDF {m : MeasurableSpace Ω} (X : Ω → E) (ℙ : Measure Ω) (μ : Measure E := by volume_tac) :
    Prop where
  protected aemeasurable' : AEMeasurable X ℙ
  protected haveLebesgueDecomposition' : (map X ℙ).HaveLebesgueDecomposition μ
  protected absolutelyContinuous' : map X ℙ ≪ μ


theorem hasPDF_iff :
    HasPDF X ℙ μ ↔ AEMeasurable X ℙ ∧ (map X ℙ).HaveLebesgueDecomposition μ ∧ map X ℙ ≪ μ :=
  ⟨fun ⟨h₁, h₂, h₃⟩ ↦ ⟨h₁, h₂, h₃⟩, fun ⟨h₁, h₂, h₃⟩ ↦ ⟨h₁, h₂, h₃⟩⟩


theorem hasPDF_iff_of_aemeasurable (hX : AEMeasurable X ℙ) :
    HasPDF X ℙ μ ↔ (map X ℙ).HaveLebesgueDecomposition μ ∧ map X ℙ ≪ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    x✝ : MeasurableSpace Ω
    X : Ω → E
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    hX : AEMeasurable X ℙ
    ⊢ Iff (MeasureTheory.HasPDF X ℙ μ) (And ((MeasureTheory.Measure.map X ℙ).HaveL …
  -/
  rw [hasPDF_iff]
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    x✝ : MeasurableSpace Ω
    X : Ω → E
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    hX : AEMeasurable X ℙ
    ⊢ Iff (And (AEMeasurable X ℙ) (And ((MeasureTheory.Measure.map X ℙ).HaveLebesg …
  -/
  simp only [hX, true_and]
  /-
    🎉 no goals
  -/


variable (X ℙ μ) in
@[measurability]
theorem HasPDF.aemeasurable [HasPDF X ℙ μ] : AEMeasurable X ℙ := HasPDF.aemeasurable' μ


instance HasPDF.haveLebesgueDecomposition [HasPDF X ℙ μ] : (map X ℙ).HaveLebesgueDecomposition μ :=
  HasPDF.haveLebesgueDecomposition'


theorem HasPDF.absolutelyContinuous [HasPDF X ℙ μ] : map X ℙ ≪ μ := HasPDF.absolutelyContinuous'


/-- A random variable that `HasPDF` is quasi-measure preserving. -/
theorem HasPDF.quasiMeasurePreserving_of_measurable (X : Ω → E) (ℙ : Measure Ω) (μ : Measure E)
    [HasPDF X ℙ μ] (h : Measurable X) : QuasiMeasurePreserving X ℙ μ :=
  { measurable := h
    absolutelyContinuous := HasPDF.absolutelyContinuous .. }


theorem HasPDF.congr (hXY : X =ᵐ[ℙ] Y) [hX : HasPDF X ℙ μ] : HasPDF Y ℙ μ :=
  ⟨(HasPDF.aemeasurable X ℙ μ).congr hXY, ℙ.map_congr hXY ▸ hX.haveLebesgueDecomposition,
    ℙ.map_congr hXY ▸ hX.absolutelyContinuous⟩


theorem HasPDF.congr_iff (hXY : X =ᵐ[ℙ] Y) : HasPDF X ℙ μ ↔ HasPDF Y ℙ μ :=
  ⟨fun _ ↦ HasPDF.congr hXY, fun _ ↦ HasPDF.congr hXY.symm⟩


@[deprecated (since := "2024-10-28")] alias HasPDF.congr' := HasPDF.congr_iff


/-- X `HasPDF` if there is a pdf `f` such that `map X ℙ = μ.withDensity f`. -/
theorem hasPDF_of_map_eq_withDensity (hX : AEMeasurable X ℙ) (f : E → ℝ≥0∞) (hf : AEMeasurable f μ)
    (h : map X ℙ = μ.withDensity f) : HasPDF X ℙ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    x✝ : MeasurableSpace Ω
    X : Ω → E
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    hX : AEMeasurable X ℙ
    f : E → ENNReal
    hf : AEMeasurable f μ
    h : Eq (MeasureTheory.Measure.map X ℙ) (μ.withDensity f)
    ⊢ MeasureTheory.HasPDF X ℙ μ
  -/
  refine ⟨hX, ?_, ?_⟩ <;> rw [h]
    /-
      case refine_1
      Ω : Type u_1
      E : Type u_2
      inst✝ : MeasurableSpace E
      x✝ : MeasurableSpace Ω
      X : Ω → E
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      hX : AEMeasurable X ℙ
      f : E → ENNReal
      hf : AEMeasurable f μ
      h : Eq (MeasureTheory.Measure.map X ℙ) (μ.withDensity f)
      ⊢ (μ.withDensity f).HaveLebesgueDecomposition μ
    -/
  · rw [withDensity_congr_ae hf.ae_eq_mk]
    /-
      case refine_1
      Ω : Type u_1
      E : Type u_2
      inst✝ : MeasurableSpace E
      x✝ : MeasurableSpace Ω
      X : Ω → E
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      hX : AEMeasurable X ℙ
      f : E → ENNReal
      hf : AEMeasurable f μ
      h : Eq (MeasureTheory.Measure.map X ℙ) (μ.withDensity f)
      ⊢ (μ.withDensity (AEMeasurable.mk f hf)).HaveLebesgueDecomposition μ
    -/
    exact haveLebesgueDecomposition_withDensity μ hf.measurable_mk
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Ω : Type u_1
      E : Type u_2
      inst✝ : MeasurableSpace E
      x✝ : MeasurableSpace Ω
      X : Ω → E
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      hX : AEMeasurable X ℙ
      f : E → ENNReal
      hf : AEMeasurable f μ
      h : Eq (MeasureTheory.Measure.map X ℙ) (μ.withDensity f)
      ⊢ (μ.withDensity f).AbsolutelyContinuous μ
    -/
  · exact withDensity_absolutelyContinuous μ f
    /-
      🎉 no goals
    -/


/-- If `X` is a random variable, then `pdf X ℙ μ`
is the Radon–Nikodym derivative of the push-forward measure of `ℙ` along `X` with respect to `μ`. -/
def pdf {_ : MeasurableSpace Ω} (X : Ω → E) (ℙ : Measure Ω) (μ : Measure E := by volume_tac) :
    E → ℝ≥0∞ :=
  (map X ℙ).rnDeriv μ


theorem pdf_def {_ : MeasurableSpace Ω} {ℙ : Measure Ω} {μ : Measure E} {X : Ω → E} :
    pdf X ℙ μ = (map X ℙ).rnDeriv μ := rfl


theorem pdf_of_not_aemeasurable {_ : MeasurableSpace Ω} {ℙ : Measure Ω} {μ : Measure E}
    {X : Ω → E} (hX : ¬AEMeasurable X ℙ) : pdf X ℙ μ =ᵐ[μ] 0 := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    X : Ω → E
    hX : Not (AEMeasurable X ℙ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0
  -/
  rw [pdf_def, map_of_not_aemeasurable hX]
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    x✝ : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    X : Ω → E
    hX : Not (AEMeasurable X ℙ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.Measure.rnDeriv 0 μ) 0
  -/
  exact rnDeriv_zero μ
  /-
    🎉 no goals
  -/


theorem pdf_of_not_haveLebesgueDecomposition {_ : MeasurableSpace Ω} {ℙ : Measure Ω}
    {μ : Measure E} {X : Ω → E} (h : ¬(map X ℙ).HaveLebesgueDecomposition μ) : pdf X ℙ μ = 0 :=
  rnDeriv_of_not_haveLebesgueDecomposition h


theorem aemeasurable_of_pdf_ne_zero {m : MeasurableSpace Ω} {ℙ : Measure Ω} {μ : Measure E}
    (X : Ω → E) (h : ¬pdf X ℙ μ =ᵐ[μ] 0) : AEMeasurable X ℙ := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    X : Ω → E
    h : Not ((MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0)
    ⊢ AEMeasurable X ℙ
  -/
  contrapose! h
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    X : Ω → E
    h : Not (AEMeasurable X ℙ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0
  -/
  exact pdf_of_not_aemeasurable h
  /-
    🎉 no goals
  -/


theorem hasPDF_of_pdf_ne_zero {m : MeasurableSpace Ω} {ℙ : Measure Ω} {μ : Measure E} {X : Ω → E}
    (hac : map X ℙ ≪ μ) (hpdf : ¬pdf X ℙ μ =ᵐ[μ] 0) : HasPDF X ℙ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    X : Ω → E
    hac : (MeasureTheory.Measure.map X ℙ).AbsolutelyContinuous μ
    hpdf : Not ((MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0)
    ⊢ MeasureTheory.HasPDF X ℙ μ
  -/
  refine ⟨?_, ?_, hac⟩
    /-
      case refine_1
      Ω : Type u_1
      E : Type u_2
      inst✝ : MeasurableSpace E
      m : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      X : Ω → E
      hac : (MeasureTheory.Measure.map X ℙ).AbsolutelyContinuous μ
      hpdf : Not ((MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0)
      ⊢ AEMeasurable X ℙ
    -/
  · exact aemeasurable_of_pdf_ne_zero X hpdf
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Ω : Type u_1
      E : Type u_2
      inst✝ : MeasurableSpace E
      m : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      X : Ω → E
      hac : (MeasureTheory.Measure.map X ℙ).AbsolutelyContinuous μ
      hpdf : Not ((MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0)
      ⊢ (MeasureTheory.Measure.map X ℙ).HaveLebesgueDecomposition μ
    -/
  · contrapose! hpdf
    /-
      case refine_2
      Ω : Type u_1
      E : Type u_2
      inst✝ : MeasurableSpace E
      m : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      X : Ω → E
      hac : (MeasureTheory.Measure.map X ℙ).AbsolutelyContinuous μ
      hpdf : Not ((MeasureTheory.Measure.map X ℙ).HaveLebesgueDecomposition μ)
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0
    -/
    have := pdf_of_not_haveLebesgueDecomposition hpdf
    /-
      case refine_2
      Ω : Type u_1
      E : Type u_2
      inst✝ : MeasurableSpace E
      m : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      X : Ω → E
      hac : (MeasureTheory.Measure.map X ℙ).AbsolutelyContinuous μ
      hpdf : Not ((MeasureTheory.Measure.map X ℙ).HaveLebesgueDecomposition μ)
      this : Eq (MeasureTheory.pdf X ℙ μ) 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.pdf X ℙ μ) 0
    -/
    filter_upwards using congrFun this
    /-
      🎉 no goals
    -/


@[measurability]
theorem measurable_pdf {m : MeasurableSpace Ω} (X : Ω → E) (ℙ : Measure Ω)
    (μ : Measure E := by volume_tac) : Measurable (pdf X ℙ μ) := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    X : Ω → E
    ℙ : MeasureTheory.Measure Ω
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    ⊢ Measurable (MeasureTheory.pdf X ℙ μ)
  -/
  exact measurable_rnDeriv _ _
  /-
    🎉 no goals
  -/


theorem withDensity_pdf_le_map {_ : MeasurableSpace Ω} (X : Ω → E) (ℙ : Measure Ω)
    (μ : Measure E := by volume_tac) : μ.withDensity (pdf X ℙ μ) ≤ map X ℙ :=
  withDensity_rnDeriv_le _ _


theorem setLIntegral_pdf_le_map {m : MeasurableSpace Ω} (X : Ω → E) (ℙ : Measure Ω)
    (μ : Measure E := by volume_tac) (s : Set E) :
    ∫⁻ x in s, pdf X ℙ μ x ∂μ ≤ map X ℙ s := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    X : Ω → E
    ℙ : MeasureTheory.Measure Ω
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    s : Set E
    ⊢ LE.le (MeasureTheory.lintegral (MeasureTheory.Measure.restrict μ s) fun x => …
  -/
  apply (withDensity_apply_le _ s).trans
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    X : Ω → E
    ℙ : MeasureTheory.Measure Ω
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    s : Set E
    ⊢ LE.le ((MeasureTheory.Measure.withDensity μ (MeasureTheory.pdf X ℙ μ)) s) (( …
  -/
  exact withDensity_pdf_le_map _ _ _ s
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_pdf_le_map := setLIntegral_pdf_le_map


theorem map_eq_withDensity_pdf {m : MeasurableSpace Ω} (X : Ω → E) (ℙ : Measure Ω)
    (μ : Measure E := by volume_tac) [hX : HasPDF X ℙ μ] :
    map X ℙ = μ.withDensity (pdf X ℙ μ) := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    X : Ω → E
    ℙ : MeasureTheory.Measure Ω
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    hX : MeasureTheory.HasPDF X ℙ μ
    ⊢ Eq (MeasureTheory.Measure.map X ℙ) (MeasureTheory.Measure.withDensity μ (Mea …
  -/
  rw [pdf_def, withDensity_rnDeriv_eq _ _ hX.absolutelyContinuous]
  /-
    🎉 no goals
  -/


theorem map_eq_setLIntegral_pdf {m : MeasurableSpace Ω} (X : Ω → E) (ℙ : Measure Ω)
    (μ : Measure E := by volume_tac) [hX : HasPDF X ℙ μ] {s : Set E}
    (hs : MeasurableSet s) : map X ℙ s = ∫⁻ x in s, pdf X ℙ μ x ∂μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    X : Ω → E
    ℙ : MeasureTheory.Measure Ω
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    hX : MeasureTheory.HasPDF X ℙ μ
    s : Set E
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.map X ℙ) s) (MeasureTheory.lintegral (MeasureTheo …
  -/
  rw [← withDensity_apply _ hs, map_eq_withDensity_pdf X ℙ μ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias map_eq_set_lintegral_pdf := map_eq_setLIntegral_pdf


protected theorem congr {X Y : Ω → E} (hXY : X =ᵐ[ℙ] Y) : pdf X ℙ μ = pdf Y ℙ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝ : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    X Y : Ω → E
    hXY : (MeasureTheory.ae ℙ).EventuallyEq X Y
    ⊢ Eq (MeasureTheory.pdf X ℙ μ) (MeasureTheory.pdf Y ℙ μ)
  -/
  rw [pdf_def, pdf_def, map_congr hXY]
  /-
    🎉 no goals
  -/


theorem lintegral_eq_measure_univ {X : Ω → E} [HasPDF X ℙ μ] :
    ∫⁻ x, pdf X ℙ μ x ∂μ = ℙ Set.univ := by
  rw [← setLIntegral_univ, ← map_eq_setLIntegral_pdf X ℙ μ MeasurableSet.univ,
    map_apply_of_aemeasurable (HasPDF.aemeasurable X ℙ μ) MeasurableSet.univ, Set.preimage_univ]


theorem eq_of_map_eq_withDensity [IsFiniteMeasure ℙ] {X : Ω → E} [HasPDF X ℙ μ] (f : E → ℝ≥0∞)
    (hmf : AEMeasurable f μ) : map X ℙ = μ.withDensity f ↔ pdf X ℙ μ =ᵐ[μ] f := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    inst✝¹ : MeasureTheory.IsFiniteMeasure ℙ
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    f : E → ENNReal
    hmf : AEMeasurable f μ
    ⊢ Iff (Eq (MeasureTheory.Measure.map X ℙ) (μ.withDensity f)) ((MeasureTheory.a …
  -/
  rw [map_eq_withDensity_pdf X ℙ μ]
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    inst✝¹ : MeasureTheory.IsFiniteMeasure ℙ
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    f : E → ENNReal
    hmf : AEMeasurable f μ
    ⊢ Iff (Eq (μ.withDensity (MeasureTheory.pdf X ℙ μ)) (μ.withDensity f)) ((Measu …
  -/
  apply withDensity_eq_iff (measurable_pdf X ℙ μ).aemeasurable hmf
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    inst✝¹ : MeasureTheory.IsFiniteMeasure ℙ
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    f : E → ENNReal
    hmf : AEMeasurable f μ
    ⊢ Ne (MeasureTheory.lintegral μ fun x => MeasureTheory.pdf X ℙ μ x) Top.top
  -/
  rw [lintegral_eq_measure_univ]
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    inst✝¹ : MeasureTheory.IsFiniteMeasure ℙ
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    f : E → ENNReal
    hmf : AEMeasurable f μ
    ⊢ Ne (ℙ Set.univ) Top.top
  -/
  exact measure_ne_top _ _
  /-
    🎉 no goals
  -/


theorem eq_of_map_eq_withDensity' [SigmaFinite μ] {X : Ω → E} [HasPDF X ℙ μ] (f : E → ℝ≥0∞)
    (hmf : AEMeasurable f μ) : map X ℙ = μ.withDensity f ↔ pdf X ℙ μ =ᵐ[μ] f :=
  map_eq_withDensity_pdf X ℙ μ ▸
    withDensity_eq_iff_of_sigmaFinite (measurable_pdf X ℙ μ).aemeasurable hmf


nonrec theorem ae_lt_top [IsFiniteMeasure ℙ] {μ : Measure E} {X : Ω → E} :
    ∀ᵐ x ∂μ, pdf X ℙ μ x < ∞ :=
  rnDeriv_lt_top (map X ℙ) μ


nonrec theorem ofReal_toReal_ae_eq [IsFiniteMeasure ℙ] {X : Ω → E} :
    (fun x => ENNReal.ofReal (pdf X ℙ μ x).toReal) =ᵐ[μ] pdf X ℙ μ :=
  ofReal_toReal_ae_eq ae_lt_top


/-- **The Law of the Unconscious Statistician** for nonnegative random variables. -/
theorem lintegral_pdf_mul {X : Ω → E} [HasPDF X ℙ μ] {f : E → ℝ≥0∞}
    (hf : AEMeasurable f μ) : ∫⁻ x, pdf X ℙ μ x * f x ∂μ = ∫⁻ x, f (X x) ∂ℙ := by
  rw [pdf_def,
    ← lintegral_map' (hf.mono_ac HasPDF.absolutelyContinuous) (HasPDF.aemeasurable X ℙ μ),
    lintegral_rnDeriv_mul HasPDF.absolutelyContinuous hf]


theorem integrable_pdf_smul_iff [IsFiniteMeasure ℙ] {X : Ω → E} [HasPDF X ℙ μ] {f : E → F}
    (hf : AEStronglyMeasurable f μ) :
    Integrable (fun x => (pdf X ℙ μ x).toReal • f x) μ ↔ Integrable (fun x => f (X x)) ℙ := by
  -- Porting note: using `erw` because `rw` doesn't recognize `(f <| X ·)` as `f ∘ X`
  -- https://github.com/leanprover-community/mathlib4/issues/5164
  erw [← integrable_map_measure (hf.mono_ac HasPDF.absolutelyContinuous)
    (HasPDF.aemeasurable X ℙ μ),
    map_eq_withDensity_pdf X ℙ μ, pdf_def, integrable_rnDeriv_smul_iff HasPDF.absolutelyContinuous]
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝⁴ : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : MeasureTheory.IsFiniteMeasure ℙ
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    f : E → F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Iff (MeasureTheory.Integrable f (MeasureTheory.Measure.map X ℙ)) (MeasureThe …
  -/
  rw [withDensity_rnDeriv_eq _ _ HasPDF.absolutelyContinuous]
  /-
    🎉 no goals
  -/


/-- **The Law of the Unconscious Statistician**: Given a random variable `X` and a measurable
function `f`, `f ∘ X` is a random variable with expectation `∫ x, pdf X x • f x ∂μ`
where `μ` is a measure on the codomain of `X`. -/
theorem integral_pdf_smul [IsFiniteMeasure ℙ] {X : Ω → E} [HasPDF X ℙ μ] {f : E → F}
    (hf : AEStronglyMeasurable f μ) : ∫ x, (pdf X ℙ μ x).toReal • f x ∂μ = ∫ x, f (X x) ∂ℙ := by
  rw [← integral_map (HasPDF.aemeasurable X ℙ μ) (hf.mono_ac HasPDF.absolutelyContinuous),
    map_eq_withDensity_pdf X ℙ μ, pdf_def, integral_rnDeriv_smul HasPDF.absolutelyContinuous,
    withDensity_rnDeriv_eq _ _ HasPDF.absolutelyContinuous]


/-- A random variable that `HasPDF` transformed under a `QuasiMeasurePreserving`
map also `HasPDF` if `(map g (map X ℙ)).HaveLebesgueDecomposition μ`.

`quasiMeasurePreserving_hasPDF` is more useful in the case we are working with a
probability measure and a real-valued random variable. -/
theorem quasiMeasurePreserving_hasPDF (hg : QuasiMeasurePreserving g μ ν)
    (hmap : (map g (map X ℙ)).HaveLebesgueDecomposition ν) : HasPDF (g ∘ X) ℙ ν := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    F : Type u_3
    inst✝¹ : MeasurableSpace F
    ν : MeasureTheory.Measure F
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    g : E → F
    hg : MeasureTheory.Measure.QuasiMeasurePreserving g μ ν
    hmap : (MeasureTheory.Measure.map g (MeasureTheory.Measure.map X ℙ)).HaveLebes …
    ⊢ MeasureTheory.HasPDF (Function.comp g X) ℙ ν
  -/
  have hgm : AEMeasurable g (map X ℙ) := hg.aemeasurable.mono_ac HasPDF.absolutelyContinuous
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    F : Type u_3
    inst✝¹ : MeasurableSpace F
    ν : MeasureTheory.Measure F
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    g : E → F
    hg : MeasureTheory.Measure.QuasiMeasurePreserving g μ ν
    hmap : (MeasureTheory.Measure.map g (MeasureTheory.Measure.map X ℙ)).HaveLebes …
    hgm : AEMeasurable g (MeasureTheory.Measure.map X ℙ)
    ⊢ MeasureTheory.HasPDF (Function.comp g X) ℙ ν
  -/
  rw [hasPDF_iff, ← AEMeasurable.map_map_of_aemeasurable hgm (HasPDF.aemeasurable X ℙ μ)]
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    F : Type u_3
    inst✝¹ : MeasurableSpace F
    ν : MeasureTheory.Measure F
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    g : E → F
    hg : MeasureTheory.Measure.QuasiMeasurePreserving g μ ν
    hmap : (MeasureTheory.Measure.map g (MeasureTheory.Measure.map X ℙ)).HaveLebes …
    hgm : AEMeasurable g (MeasureTheory.Measure.map X ℙ)
    ⊢ And (AEMeasurable (Function.comp g X) ℙ) (And ((MeasureTheory.Measure.map g  …
  -/
  refine ⟨hg.measurable.comp_aemeasurable (HasPDF.aemeasurable _ _ μ), hmap, ?_⟩
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace E
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    μ : MeasureTheory.Measure E
    F : Type u_3
    inst✝¹ : MeasurableSpace F
    ν : MeasureTheory.Measure F
    X : Ω → E
    inst✝ : MeasureTheory.HasPDF X ℙ μ
    g : E → F
    hg : MeasureTheory.Measure.QuasiMeasurePreserving g μ ν
    hmap : (MeasureTheory.Measure.map g (MeasureTheory.Measure.map X ℙ)).HaveLebes …
    hgm : AEMeasurable g (MeasureTheory.Measure.map X ℙ)
    ⊢ (MeasureTheory.Measure.map g (MeasureTheory.Measure.map X ℙ)).AbsolutelyCont …
  -/
  exact (HasPDF.absolutelyContinuous.map hg.1).trans hg.2
  /-
    🎉 no goals
  -/


theorem quasiMeasurePreserving_hasPDF' [SFinite ℙ] [SigmaFinite ν]
    (hg : QuasiMeasurePreserving g μ ν) : HasPDF (g ∘ X) ℙ ν :=
  quasiMeasurePreserving_hasPDF X hg inferInstance


nonrec theorem _root_.Real.hasPDF_iff [SFinite ℙ] :
    /-
      Ω : Type u_1
      E : Type u_2
      inst✝¹ : MeasurableSpace E
      m : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      X : Ω → Real
      inst✝ : MeasureTheory.SFinite ℙ
      ⊢ MeasureTheory.Measure Real
    -/
    HasPDF X ℙ ↔ AEMeasurable X ℙ ∧ map X ℙ ≪ volume := by
    /-
      🎉 no goals
    -/
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    inst✝ : MeasureTheory.SFinite ℙ
    ⊢ Iff (MeasureTheory.HasPDF X ℙ MeasureTheory.MeasureSpace.volume) (And (AEMea …
  -/
  rw [hasPDF_iff, and_iff_right (inferInstance : HaveLebesgueDecomposition _ _)]
  /-
    🎉 no goals
  -/


/-- A real-valued random variable `X` `HasPDF X ℙ λ` (where `λ` is the Lebesgue measure) if and
only if the push-forward measure of `ℙ` along `X` is absolutely continuous with respect to `λ`. -/
nonrec theorem _root_.Real.hasPDF_iff_of_aemeasurable [SFinite ℙ] (hX : AEMeasurable X ℙ) :
    /-
      Ω : Type u_1
      E : Type u_2
      inst✝¹ : MeasurableSpace E
      m : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      X : Ω → Real
      inst✝ : MeasureTheory.SFinite ℙ
      hX : AEMeasurable X ℙ
      ⊢ MeasureTheory.Measure Real
    -/
    HasPDF X ℙ ↔ map X ℙ ≪ volume := by
    /-
      🎉 no goals
    -/
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    inst✝ : MeasureTheory.SFinite ℙ
    hX : AEMeasurable X ℙ
    ⊢ Iff (MeasureTheory.HasPDF X ℙ MeasureTheory.MeasureSpace.volume) ((MeasureTh …
  -/
  rw [Real.hasPDF_iff, and_iff_right hX]
  /-
    🎉 no goals
  -/


/-- If `X` is a real-valued random variable that has pdf `f`, then the expectation of `X` equals
`∫ x, x * f x ∂λ` where `λ` is the Lebesgue measure. -/
                                  /-
                                    Ω : Type u_1
                                    E : Type u_2
                                    inst✝¹ : MeasurableSpace E
                                    m : MeasurableSpace Ω
                                    ℙ : MeasureTheory.Measure Ω
                                    μ : MeasureTheory.Measure E
                                    X : Ω → Real
                                    inst✝ : MeasureTheory.IsFiniteMeasure ℙ
                                    ⊢ MeasureTheory.Measure Real
                                  -/
theorem integral_mul_eq_integral [HasPDF X ℙ] : ∫ x, x * (pdf X ℙ volume x).toReal = ∫ x, X x ∂ℙ :=
                                  /-
                                    🎉 no goals
                                  -/
  calc
                                                 /-
                                                   Ω : Type u_1
                                                   m : MeasurableSpace Ω
                                                   ℙ : MeasureTheory.Measure Ω
                                                   X : Ω → Real
                                                   inst✝¹ : MeasureTheory.IsFiniteMeasure ℙ
                                                   inst✝ : MeasureTheory.HasPDF X ℙ MeasureTheory.MeasureSpace.volume
                                                   ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => HMul.h …
                                                 -/
    _ = ∫ x, (pdf X ℙ volume x).toReal * x := by congr with x; exact mul_comm _ _
                                                               /-
                                                                 🎉 no goals
                                                               -/
    _ = _ := integral_pdf_smul measurable_id.aestronglyMeasurable


                                                               /-
                                                                 Ω : Type u_1
                                                                 E : Type u_2
                                                                 inst✝¹ : MeasurableSpace E
                                                                 m : MeasurableSpace Ω
                                                                 ℙ : MeasureTheory.Measure Ω
                                                                 μ : MeasureTheory.Measure E
                                                                 X : Ω → Real
                                                                 inst✝ : MeasureTheory.IsFiniteMeasure ℙ
                                                                 f : Real → Real
                                                                 g : Real → ENNReal
                                                                 ⊢ MeasureTheory.Measure Real
                                                               -/
theorem hasFiniteIntegral_mul {f : ℝ → ℝ} {g : ℝ → ℝ≥0∞} (hg : pdf X ℙ =ᵐ[volume] g)
                                                               /-
                                                                 🎉 no goals
                                                               -/
    (hgi : ∫⁻ x, ‖f x‖₊ * g x ≠ ∞) :
    /-
      Ω : Type u_1
      E : Type u_2
      inst✝¹ : MeasurableSpace E
      m : MeasurableSpace Ω
      ℙ : MeasureTheory.Measure Ω
      μ : MeasureTheory.Measure E
      X : Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure ℙ
      f : Real → Real
      g : Real → ENNReal
      hg : (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Measur …
      hgi : Ne (MeasureTheory.lintegral MeasureTheory.MeasureSpace.volume fun x => H …
      ⊢ MeasureTheory.Measure Real
    -/
    HasFiniteIntegral fun x => f x * (pdf X ℙ volume x).toReal := by
    /-
      🎉 no goals
    -/
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    inst✝ : MeasureTheory.IsFiniteMeasure ℙ
    f : Real → Real
    g : Real → ENNReal
    hg : (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Measur …
    hgi : Ne (MeasureTheory.lintegral MeasureTheory.MeasureSpace.volume fun x => H …
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => HMul.hMul (f x) (MeasureTheory.pdf …
  -/
  rw [hasFiniteIntegral_iff_nnnorm]
  have : (fun x => ↑‖f x‖₊ * g x) =ᵐ[volume] fun x => ‖f x * (pdf X ℙ volume x).toReal‖₊ := by
    refine ae_eq_trans (Filter.EventuallyEq.mul (ae_eq_refl fun x => (‖f x‖₊ : ℝ≥0∞))
      (ae_eq_trans hg.symm ofReal_toReal_ae_eq.symm)) ?_
    simp_rw [← smul_eq_mul, nnnorm_smul, ENNReal.coe_mul, smul_eq_mul]
    refine Filter.EventuallyEq.mul (ae_eq_refl _) ?_
    simp only [Real.ennnorm_eq_ofReal ENNReal.toReal_nonneg, ae_eq_refl]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    ℙ : MeasureTheory.Measure Ω
    X : Ω → Real
    inst✝ : MeasureTheory.IsFiniteMeasure ℙ
    f : Real → Real
    g : Real → ENNReal
    hg : (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Measur …
    hgi : Ne (MeasureTheory.lintegral MeasureTheory.MeasureSpace.volume fun x => H …
    this : (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (fun  …
    ⊢ LT.lt (MeasureTheory.lintegral MeasureTheory.MeasureSpace.volume fun a => ↑( …
  -/
  rwa [lt_top_iff_ne_top, ← lintegral_congr_ae this]
  /-
    🎉 no goals
  -/


/-- Random variables are independent iff their joint density is a product of marginal densities. -/
theorem indepFun_iff_pdf_prod_eq_pdf_mul_pdf
    [IsFiniteMeasure ℙ] [SigmaFinite μ] [SigmaFinite ν] [HasPDF (fun ω ↦ (X ω, Y ω)) ℙ (μ.prod ν)] :
    IndepFun X Y ℙ ↔
      pdf (fun ω ↦ (X ω, Y ω)) ℙ (μ.prod ν) =ᵐ[μ.prod ν] fun z ↦ pdf X ℙ μ z.1 * pdf Y ℙ ν z.2 := by
  have : HasPDF X ℙ μ := quasiMeasurePreserving_hasPDF' (μ := μ.prod ν) (fun ω ↦ (X ω, Y ω))
    quasiMeasurePreserving_fst
  have : HasPDF Y ℙ ν := quasiMeasurePreserving_hasPDF' (μ := μ.prod ν) (fun ω ↦ (X ω, Y ω))
    quasiMeasurePreserving_snd
  have h₀ : (ℙ.map X).prod (ℙ.map Y) =
      (μ.prod ν).withDensity fun z ↦ pdf X ℙ μ z.1 * pdf Y ℙ ν z.2 :=
    prod_eq fun s t hs ht ↦ by rw [withDensity_apply _ (hs.prod ht), ← prod_restrict,
      lintegral_prod_mul (measurable_pdf X ℙ μ).aemeasurable (measurable_pdf Y ℙ ν).aemeasurable,
      map_eq_setLIntegral_pdf X ℙ μ hs, map_eq_setLIntegral_pdf Y ℙ ν ht]
  rw [indepFun_iff_map_prod_eq_prod_map_map (HasPDF.aemeasurable X ℙ μ) (HasPDF.aemeasurable Y ℙ ν),
    ← eq_of_map_eq_withDensity, h₀]
  exact (((measurable_pdf X ℙ μ).comp measurable_fst).mul
    ((measurable_pdf Y ℙ ν).comp measurable_snd)).aemeasurable


