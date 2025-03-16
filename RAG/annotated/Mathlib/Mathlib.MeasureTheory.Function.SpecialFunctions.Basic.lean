@[measurability]
theorem measurable_exp : Measurable exp :=
  continuous_exp.measurable


@[measurability]
theorem measurable_log : Measurable log :=
  measurable_of_measurable_on_compl_singleton 0 <|
    Continuous.measurable <| continuousOn_iff_continuous_restrict.1 continuousOn_log


lemma measurable_of_measurable_exp {α : Type*} {_ : MeasurableSpace α} {f : α → ℝ}
    (hf : Measurable (fun x ↦ exp (f x))) :
    Measurable f := by
  /-
    α : Type u_1
    x✝ : MeasurableSpace α
    f : α → Real
    hf : Measurable fun x => Real.exp (f x)
    ⊢ Measurable f
  -/
  have : f = fun x ↦ log (exp (f x)) := by ext; rw [log_exp]
  /-
    α : Type u_1
    x✝ : MeasurableSpace α
    f : α → Real
    hf : Measurable fun x => Real.exp (f x)
    this : Eq f fun x => Real.log (Real.exp (f x))
    ⊢ Measurable f
  -/
  rw [this]
  /-
    α : Type u_1
    x✝ : MeasurableSpace α
    f : α → Real
    hf : Measurable fun x => Real.exp (f x)
    this : Eq f fun x => Real.log (Real.exp (f x))
    ⊢ Measurable fun x => Real.log (Real.exp (f x))
  -/
  exact measurable_log.comp hf
  /-
    🎉 no goals
  -/


lemma aemeasurable_of_aemeasurable_exp {α : Type*} {_ : MeasurableSpace α} {f : α → ℝ}
    {μ : MeasureTheory.Measure α} (hf : AEMeasurable (fun x ↦ exp (f x)) μ) :
    AEMeasurable f μ := by
  /-
    α : Type u_1
    x✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    hf : AEMeasurable (fun x => Real.exp (f x)) μ
    ⊢ AEMeasurable f μ
  -/
  have : f = fun x ↦ log (exp (f x)) := by ext; rw [log_exp]
  /-
    α : Type u_1
    x✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    hf : AEMeasurable (fun x => Real.exp (f x)) μ
    this : Eq f fun x => Real.log (Real.exp (f x))
    ⊢ AEMeasurable f μ
  -/
  rw [this]
  /-
    α : Type u_1
    x✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    hf : AEMeasurable (fun x => Real.exp (f x)) μ
    this : Eq f fun x => Real.log (Real.exp (f x))
    ⊢ AEMeasurable (fun x => Real.log (Real.exp (f x))) μ
  -/
  exact measurable_log.comp_aemeasurable hf
  /-
    🎉 no goals
  -/


@[measurability]
theorem measurable_sin : Measurable sin :=
  continuous_sin.measurable


@[measurability]
theorem measurable_cos : Measurable cos :=
  continuous_cos.measurable


@[measurability]
theorem measurable_sinh : Measurable sinh :=
  continuous_sinh.measurable


@[measurability]
theorem measurable_cosh : Measurable cosh :=
  continuous_cosh.measurable


@[measurability]
theorem measurable_arcsin : Measurable arcsin :=
  continuous_arcsin.measurable


@[measurability]
theorem measurable_arccos : Measurable arccos :=
  continuous_arccos.measurable


@[measurability]
theorem measurable_re : Measurable re :=
  continuous_re.measurable


@[measurability]
theorem measurable_im : Measurable im :=
  continuous_im.measurable


@[measurability]
theorem measurable_ofReal : Measurable ((↑) : ℝ → ℂ) :=
  continuous_ofReal.measurable


@[measurability]
theorem measurable_arg : Measurable arg :=
  have A : Measurable fun x : ℂ => Real.arcsin (x.im / Complex.abs x) :=
    Real.measurable_arcsin.comp (measurable_im.div measurable_norm)
  have B : Measurable fun x : ℂ => Real.arcsin ((-x).im / Complex.abs x) :=
    Real.measurable_arcsin.comp ((measurable_im.comp measurable_neg).div measurable_norm)
  Measurable.ite (isClosed_le continuous_const continuous_re).measurableSet A <|
    Measurable.ite (isClosed_le continuous_const continuous_im).measurableSet (B.add_const _)
      (B.sub_const _)


@[measurability]
theorem measurable_log : Measurable log :=
  (measurable_ofReal.comp <| Real.measurable_log.comp measurable_norm).add <|
    (measurable_ofReal.comp measurable_arg).mul_const I


@[measurability]
protected theorem Measurable.exp : Measurable fun x => Real.exp (f x) :=
  Real.measurable_exp.comp hf


@[measurability]
protected theorem Measurable.log : Measurable fun x => log (f x) :=
  measurable_log.comp hf


@[measurability]
protected theorem Measurable.cos : Measurable fun x ↦ cos (f x) := measurable_cos.comp hf


@[measurability]
protected theorem Measurable.sin : Measurable fun x ↦ sin (f x) := measurable_sin.comp hf


@[measurability]
protected theorem Measurable.cosh : Measurable fun x ↦ cosh (f x) := measurable_cosh.comp hf


@[measurability]
protected theorem Measurable.sinh : Measurable fun x ↦ sinh (f x) := measurable_sinh.comp hf


@[measurability]
protected theorem Measurable.sqrt : Measurable fun x => √(f x) := continuous_sqrt.measurable.comp hf


@[measurability, fun_prop]
protected lemma AEMeasurable.exp : AEMeasurable (fun x ↦ exp (f x)) μ :=
  measurable_exp.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.log : AEMeasurable (fun x ↦ log (f x)) μ :=
  measurable_log.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.cos : AEMeasurable (fun x ↦ cos (f x)) μ :=
  measurable_cos.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.sin : AEMeasurable (fun x ↦ sin (f x)) μ :=
  measurable_sin.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.cosh : AEMeasurable (fun x ↦ cosh (f x)) μ :=
  measurable_cosh.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.sinh : AEMeasurable (fun x ↦ sinh (f x)) μ :=
  measurable_sinh.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.sqrt : AEMeasurable (fun x ↦ √(f x)) μ :=
  continuous_sqrt.measurable.comp_aemeasurable hf


@[measurability]
protected theorem Measurable.cexp : Measurable fun x => Complex.exp (f x) :=
  Complex.measurable_exp.comp hf


@[measurability]
protected theorem Measurable.ccos : Measurable fun x => Complex.cos (f x) :=
  Complex.measurable_cos.comp hf


@[measurability]
protected theorem Measurable.csin : Measurable fun x => Complex.sin (f x) :=
  Complex.measurable_sin.comp hf


@[measurability]
protected theorem Measurable.ccosh : Measurable fun x => Complex.cosh (f x) :=
  Complex.measurable_cosh.comp hf


@[measurability]
protected theorem Measurable.csinh : Measurable fun x => Complex.sinh (f x) :=
  Complex.measurable_sinh.comp hf


@[measurability]
protected theorem Measurable.carg : Measurable fun x => arg (f x) :=
  measurable_arg.comp hf


@[measurability]
protected theorem Measurable.clog : Measurable fun x => Complex.log (f x) :=
  measurable_log.comp hf


@[measurability, fun_prop]
protected lemma AEMeasurable.cexp : AEMeasurable (fun x ↦ exp (f x)) μ :=
  measurable_exp.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.ccos : AEMeasurable (fun x ↦ cos (f x)) μ :=
  measurable_cos.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.csin : AEMeasurable (fun x ↦ sin (f x)) μ :=
  measurable_sin.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.ccosh : AEMeasurable (fun x ↦ cosh (f x)) μ :=
  measurable_cosh.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.csinh : AEMeasurable (fun x ↦ sinh (f x)) μ :=
  measurable_sinh.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.carg : AEMeasurable (fun x ↦ arg (f x)) μ :=
  measurable_arg.comp_aemeasurable hf


@[measurability, fun_prop]
protected lemma AEMeasurable.clog : AEMeasurable (fun x ↦ log (f x)) μ :=
  measurable_log.comp_aemeasurable hf


instance Complex.hasMeasurablePow : MeasurablePow ℂ ℂ :=
  ⟨Measurable.ite (measurable_fst (measurableSet_singleton 0))
      (Measurable.ite (measurable_snd (measurableSet_singleton 0)) measurable_one measurable_zero)
      (measurable_fst.clog.mul measurable_snd).cexp⟩


instance Real.hasMeasurablePow : MeasurablePow ℝ ℝ :=
  ⟨Complex.measurable_re.comp <|
      (Complex.measurable_ofReal.comp measurable_fst).pow
        (Complex.measurable_ofReal.comp measurable_snd)⟩


instance NNReal.hasMeasurablePow : MeasurablePow ℝ≥0 ℝ :=
  ⟨(measurable_fst.coe_nnreal_real.pow measurable_snd).subtype_mk⟩


instance ENNReal.hasMeasurablePow : MeasurablePow ℝ≥0∞ ℝ := by
  /-
    ⊢ MeasurablePow ENNReal Real
  -/
  refine ⟨ENNReal.measurable_of_measurable_nnreal_prod ?_ ?_⟩
    /-
      case refine_1
      ⊢ Measurable fun p => HPow.hPow { fst := ↑p.1, snd := p.2 }.1 { fst := ↑p.1, s …
    -/
  · simp_rw [ENNReal.coe_rpow_def]
    /-
      case refine_1
      ⊢ Measurable fun p => ite (And (Eq p.1 0) (LT.lt p.2 0)) Top.top ↑(HPow.hPow p …
    -/
    refine Measurable.ite ?_ measurable_const (measurable_fst.pow measurable_snd).coe_nnreal_ennreal
    exact
      MeasurableSet.inter (measurable_fst (measurableSet_singleton 0))
        (measurable_snd measurableSet_Iio)
    /-
      case refine_2
      ⊢ Measurable fun x => HPow.hPow { fst := Top.top, snd := x }.1 { fst := Top.to …
    -/
  · simp_rw [ENNReal.top_rpow_def]
    /-
      case refine_2
      ⊢ Measurable fun x => ite (LT.lt 0 x) Top.top (ite (Eq x 0) 1 0)
    -/
    refine Measurable.ite measurableSet_Ioi measurable_const ?_
    /-
      case refine_2
      ⊢ Measurable fun x => ite (Eq x 0) 1 0
    -/
    exact Measurable.ite (measurableSet_singleton 0) measurable_const measurable_const
    /-
      🎉 no goals
    -/


