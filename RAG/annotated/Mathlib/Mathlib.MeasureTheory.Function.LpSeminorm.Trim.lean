theorem eLpNorm'_trim (hm : m ≤ m0) {f : α → E} (hf : StronglyMeasurable[m] f) :
    eLpNorm' f q (μ.trim hm) = eLpNorm' f q μ := by
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (MeasureTheory.eLpNorm' f q (μ.trim hm)) (MeasureTheory.eLpNorm' f q μ)
  -/
  simp_rw [eLpNorm']
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (HPow.hPow (MeasureTheory.lintegral (μ.trim hm) fun a => HPow.hPow (ENorm …
  -/
  congr 1
  /-
    case e_a
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (MeasureTheory.lintegral (μ.trim hm) fun a => HPow.hPow (ENorm.enorm (f a …
  -/
  refine lintegral_trim hm ?_
  /-
    case e_a
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Measurable fun a => HPow.hPow (ENorm.enorm (f a)) q
  -/
  refine @Measurable.pow_const _ _ _ _ _ _ _ m _ (@Measurable.coe_nnreal_ennreal _ m _ ?_) q
  /-
    case e_a
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Measurable fun a => NNNorm.nnnorm (f a)
  -/
  apply @StronglyMeasurable.measurable
  /-
    case e_a.hf
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    q : Real
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable fun a => NNNorm.nnnorm (f a)
  -/
  exact @StronglyMeasurable.nnnorm α m _ _ _ hf
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_trim := eLpNorm'_trim


theorem limsup_trim (hm : m ≤ m0) {f : α → ℝ≥0∞} (hf : Measurable[m] f) :
    limsup f (ae (μ.trim hm)) = limsup f (ae μ) := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (Filter.limsup f (MeasureTheory.ae (μ.trim hm))) (Filter.limsup f (Measur …
  -/
  simp_rw [limsup_eq]
  suffices h_set_eq : { a : ℝ≥0∞ | ∀ᵐ n ∂μ.trim hm, f n ≤ a } = { a : ℝ≥0∞ | ∀ᵐ n ∂μ, f n ≤ a } by
    rw [h_set_eq]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (setOf fun a => Filter.Eventually (fun n => LE.le (f n) a) (MeasureTheory …
  -/
  ext1 a
  suffices h_meas_eq : μ { x | ¬f x ≤ a } = μ.trim hm { x | ¬f x ≤ a } by
    simp_rw [Set.mem_setOf_eq, ae_iff, h_meas_eq]
  /-
    case h
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → ENNReal
    hf : Measurable f
    a : ENNReal
    ⊢ Eq (μ (setOf fun x => Not (LE.le (f x) a))) ((μ.trim hm) (setOf fun x => Not …
  -/
  refine (trim_measurableSet_eq hm ?_).symm
  /-
    case h
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → ENNReal
    hf : Measurable f
    a : ENNReal
    ⊢ MeasurableSet (setOf fun x => Not (LE.le (f x) a))
  -/
  refine @MeasurableSet.compl _ _ m (@measurableSet_le ℝ≥0∞ _ _ _ _ m _ _ _ _ _ hf ?_)
  /-
    case h
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → ENNReal
    hf : Measurable f
    a : ENNReal
    ⊢ Measurable fun x => a
  -/
  exact @measurable_const _ _ _ m _
  /-
    🎉 no goals
  -/


theorem essSup_trim (hm : m ≤ m0) {f : α → ℝ≥0∞} (hf : Measurable[m] f) :
    essSup f (μ.trim hm) = essSup f μ := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (essSup f (μ.trim hm)) (essSup f μ)
  -/
  simp_rw [essSup]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (Filter.limsup f (MeasureTheory.ae (μ.trim hm))) (Filter.limsup f (Measur …
  -/
  exact limsup_trim hm hf
  /-
    🎉 no goals
  -/


theorem eLpNormEssSup_trim (hm : m ≤ m0) {f : α → E} (hf : StronglyMeasurable[m] f) :
    eLpNormEssSup f (μ.trim hm) = eLpNormEssSup f μ :=
  essSup_trim _ (@StronglyMeasurable.ennnorm _ m _ _ _ hf)


@[deprecated (since := "2024-07-27")]
alias snormEssSup_trim := eLpNormEssSup_trim


theorem eLpNorm_trim (hm : m ≤ m0) {f : α → E} (hf : StronglyMeasurable[m] f) :
    eLpNorm f p (μ.trim hm) = eLpNorm f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (MeasureTheory.eLpNorm f p (μ.trim hm)) (MeasureTheory.eLpNorm f p μ)
  -/
  by_cases h0 : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      hm : LE.le m m0
      f : α → E
      hf : MeasureTheory.StronglyMeasurable f
      h0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm f p (μ.trim hm)) (MeasureTheory.eLpNorm f p μ)
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    h0 : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm f p (μ.trim hm)) (MeasureTheory.eLpNorm f p μ)
  -/
  by_cases h_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      hm : LE.le m m0
      f : α → E
      hf : MeasureTheory.StronglyMeasurable f
      h0 : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm f p (μ.trim hm)) (MeasureTheory.eLpNorm f p μ)
    -/
  · simpa only [h_top, eLpNorm_exponent_top] using eLpNormEssSup_trim hm hf
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm f p (μ.trim hm)) (MeasureTheory.eLpNorm f p μ)
  -/
  simpa only [eLpNorm_eq_eLpNorm' h0 h_top] using eLpNorm'_trim hm hf
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_trim := eLpNorm_trim


theorem eLpNorm_trim_ae (hm : m ≤ m0) {f : α → E} (hf : AEStronglyMeasurable f (μ.trim hm)) :
    eLpNorm f p (μ.trim hm) = eLpNorm f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.AEStronglyMeasurable f (μ.trim hm)
    ⊢ Eq (MeasureTheory.eLpNorm f p (μ.trim hm)) (MeasureTheory.eLpNorm f p μ)
  -/
  rw [eLpNorm_congr_ae hf.ae_eq_mk, eLpNorm_congr_ae (ae_eq_of_ae_eq_trim hf.ae_eq_mk)]
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hm : LE.le m m0
    f : α → E
    hf : MeasureTheory.AEStronglyMeasurable f (μ.trim hm)
    ⊢ Eq (MeasureTheory.eLpNorm (MeasureTheory.AEStronglyMeasurable.mk f hf) p (μ. …
  -/
  exact eLpNorm_trim hm hf.stronglyMeasurable_mk
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_trim_ae := eLpNorm_trim_ae


theorem memℒp_of_memℒp_trim (hm : m ≤ m0) {f : α → E} (hf : Memℒp f p (μ.trim hm)) : Memℒp f p μ :=
  ⟨aestronglyMeasurable_of_aestronglyMeasurable_trim hm hf.1,
    (le_of_eq (eLpNorm_trim_ae hm hf.1).symm).trans_lt hf.2⟩


